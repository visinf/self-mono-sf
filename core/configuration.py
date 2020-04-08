## Portions of Code from, copyright 2018 Jochen Gast

from __future__ import absolute_import, division, print_function

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import shutil
import random
import fnmatch

from core import logger, tools

from torch.utils.data.sampler import RandomSampler
from datasets.custom_batchsampler import CustomBatchSampler

# ---------------------------------------------------
# Class that contains both the network model and loss
# ---------------------------------------------------
class ModelAndLoss(nn.Module):
    def __init__(self, args, model, training_loss, evaluation_loss=None):
        super(ModelAndLoss, self).__init__()
        self._model = model
        self._training_loss = training_loss
        self._evaluation_loss = evaluation_loss

    @property
    def training_loss(self):
        return self._training_loss

    @property
    def evaluation_loss(self):
        return self._evaluation_loss

    @property
    def model(self):
        return self._model

    def num_parameters(self):
        return sum([p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    # -------------------------------------------------------------
    # Note: We merge inputs and targets into a single dictionary !
    # -------------------------------------------------------------
    def forward(self, example_dict):
        # -------------------------------------
        # Run forward pass
        # -------------------------------------
        output_dict = self._model(example_dict)

        # -------------------------------------
        # Compute losses
        # -------------------------------------
        if self.training:
            loss_dict = self._training_loss(output_dict, example_dict)
        else:
            loss_dict = self._evaluation_loss(output_dict, example_dict)

        # -------------------------------------
        # Return losses and outputs
        # -------------------------------------
        return loss_dict, output_dict


def configure_runtime_augmentations(args):
    with logger.LoggingBlock("Runtime Augmentations", emph=True):

        training_augmentation = None
        validation_augmentation = None

        # ----------------------------------------------------
        # Training Augmentation
        # ----------------------------------------------------
        if args.training_augmentation is not None:
            kwargs = tools.kwargs_from_args(args, "training_augmentation")
            logging.info("training_augmentation: %s" % args.training_augmentation)
            for param, default in sorted(kwargs.items()):
                logging.info("  %s: %s" % (param, default))
            kwargs["args"] = args
            training_augmentation = tools.instance_from_kwargs(
                args.training_augmentation_class, kwargs)
            if args.cuda:
                training_augmentation = training_augmentation.cuda()

        else:
            logging.info("training_augmentation: None")

        # ----------------------------------------------------
        # Validation Augmentation
        # ----------------------------------------------------
        if args.validation_augmentation is not None:
            kwargs = tools.kwargs_from_args(args, "validation_augmentation")
            logging.info("validation_augmentation: %s" % args.validation_augmentation)
            for param, default in sorted(kwargs.items()):
                logging.info("  %s: %s" % (param, default))
            kwargs["args"] = args
            validation_augmentation = tools.instance_from_kwargs(
                args.validation_augmentation_class, kwargs)
            if args.cuda:
                validation_augmentation = validation_augmentation.cuda()

        else:
            logging.info("validation_augmentation: None")

    return training_augmentation, validation_augmentation


def configure_model_and_loss(args):

    # ----------------------------------------------------
    # Dynamically load model and loss class with parameters
    # passed in via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    # ----------------------------------------------------
    with logger.LoggingBlock("Model and Loss", emph=True):

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------
        kwargs = tools.kwargs_from_args(args, "model")
        kwargs["args"] = args
        model = tools.instance_from_kwargs(args.model_class, kwargs)

        # ----------------------------------------------------
        # Training loss
        # ----------------------------------------------------
        training_loss = None
        if args.training_loss is not None:
            kwargs = tools.kwargs_from_args(args, "training_loss")
            kwargs["args"] = args
            training_loss = tools.instance_from_kwargs(args.training_loss_class, kwargs)

        # ----------------------------------------------------
        # Validation loss
        # ----------------------------------------------------
        validation_loss = None
        if args.validation_loss is not None:
            kwargs = tools.kwargs_from_args(args, "validation_loss")
            kwargs["args"] = args
            validation_loss = tools.instance_from_kwargs(args.validation_loss_class, kwargs)

        # ----------------------------------------------------
        # Model and loss
        # ----------------------------------------------------
        model_and_loss = ModelAndLoss(args, model, training_loss, validation_loss)

        # -----------------------------------------------------------
        # If Cuda, transfer model to Cuda and wrap with DataParallel.
        # -----------------------------------------------------------
        if args.cuda:
            model_and_loss = model_and_loss.cuda()

        # ---------------------------------------------------------------
        # Report some network statistics
        # ---------------------------------------------------------------
        logging.info("Batch Size: %i" % args.batch_size)
        logging.info("GPGPU: Cuda") if args.cuda else logging.info("GPGPU: off")
        logging.info("Network: %s" % args.model)
        logging.info("Number of parameters: %i" % tools.x2module(model_and_loss).num_parameters())
        if training_loss is not None:
            logging.info("Training Key: %s" % args.training_key)
            logging.info("Training Loss: %s" % args.training_loss)
        if validation_loss is not None:
            logging.info("Validation Key: %s" % args.validation_key)
            logging.info("Validation Loss: %s" % args.validation_loss)

    return model_and_loss


def configure_random_seed(args):
    with logger.LoggingBlock("Random Seeds", emph=True):
        # python
        seed = args.seed
        random.seed(seed)
        logging.info("Python seed: %i" % seed)
        # numpy
        seed += 1
        np.random.seed(seed)
        logging.info("Numpy seed: %i" % seed)
        # torch
        seed += 1
        torch.manual_seed(seed)
        logging.info("Torch CPU seed: %i" % seed)
        # torch cuda
        seed += 1
        torch.cuda.manual_seed(seed)
        logging.info("Torch CUDA seed: %i" % seed)


# --------------------------------------------------------------------------
# Checkpoint loader/saver.
# --------------------------------------------------------------------------
class CheckpointSaver:
    def __init__(self,
                 prefix="checkpoint",
                 latest_postfix="_latest",
                 best_postfix="_best",
                 model_key="state_dict",
                 extension=".ckpt"):

        self._prefix = prefix
        self._model_key = model_key
        self._latest_postfix = latest_postfix
        self._best_postfix = best_postfix
        self._extension = extension

    # the purpose of rewriting the loading function is we sometimes want to
    # initialize parameters in modules without knowing the dimensions at runtime
    #
    # This function here will resize these parameters to whatever size required.
    #
    def _load_state_dict_into_module(self, state_dict, module, strict=True):
        own_state = module.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].resize_as_(param)
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            elif strict:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def restore(self, filename, model_and_loss, include_params="*", exclude_params=()):
        # -----------------------------------------------------------------------------------------
        # Make sure file exists
        # -----------------------------------------------------------------------------------------
        if not os.path.isfile(filename):
            logging.info("Could not find checkpoint file '%s'!" % filename)
            quit()

        # -----------------------------------------------------------------------------------------
        # Load checkpoint from file including the state_dict
        # -----------------------------------------------------------------------------------------
        checkpoint_with_state = torch.load(filename)

        # -----------------------------------------------------------------------------------------
        # Load filtered state dictionary
        # -----------------------------------------------------------------------------------------
        state_dict = checkpoint_with_state[self._model_key]
        restore_keys = tools.filter_list_of_strings(
            state_dict.keys(),
            include=include_params,
            exclude=exclude_params)
        state_dict = {key: value for key, value in state_dict.items() if key in restore_keys}
        self._load_state_dict_into_module(state_dict, model_and_loss)
        logging.info("  Restore keys:")
        for key in restore_keys:
            logging.info("    %s" % key)

        # -----------------------------------------------------------------------------------------
        # Get checkpoint statistics without the state dict
        # -----------------------------------------------------------------------------------------
        checkpoint_stats = {
            key: value for key, value in checkpoint_with_state.items() if key != self._model_key
        }

        return checkpoint_stats, filename

    def restore_latest(self, directory, model_and_loss, include_params="*", exclude_params=()):
        latest_checkpoint_filename = os.path.join(
            directory, self._prefix + self._latest_postfix + self._extension)
        return self.restore(latest_checkpoint_filename, model_and_loss, include_params, exclude_params)

    def restore_best(self, directory, model_and_loss, include_params="*", exclude_params=()):
        best_checkpoint_filename = os.path.join(
            directory, self._prefix + self._best_postfix + self._extension)
        return self.restore(best_checkpoint_filename, model_and_loss, include_params, exclude_params)

    def save_latest(self, directory, model_and_loss, stats_dict, store_as_best=False):
        # -----------------------------------------------------------------------------------------
        # Make sure directory exists
        # -----------------------------------------------------------------------------------------
        tools.ensure_dir(directory)

        # -----------------------------------------------------------------------------------------
        # Save
        # -----------------------------------------------------------------------------------------
        save_dict = dict(stats_dict)
        save_dict[self._model_key] = model_and_loss.state_dict()

        latest_checkpoint_filename = os.path.join(directory, self._prefix + self._latest_postfix + self._extension)
        torch.save(save_dict, latest_checkpoint_filename)

        # -----------------------------------------------------------------------------------------
        # Possibly store as best
        # -----------------------------------------------------------------------------------------
        if store_as_best:
            best_checkpoint_filename = os.path.join(directory, self._prefix + self._best_postfix + self._extension)

            logging.info("Saved checkpoint as best model..")
            shutil.copyfile(latest_checkpoint_filename, best_checkpoint_filename)


def configure_checkpoint_saver(args, model_and_loss):
    with logger.LoggingBlock("Checkpoint", emph=True):
        checkpoint_saver = CheckpointSaver()
        checkpoint_stats = None

        if args.checkpoint is None:
            logging.info("No checkpoint given.")
            logging.info("Starting from scratch with random initialization.")

        elif os.path.isfile(args.checkpoint):
            checkpoint_stats, filename = checkpoint_saver.restore(
                filename=args.checkpoint,
                model_and_loss=model_and_loss,
                include_params=args.checkpoint_include_params,
                exclude_params=args.checkpoint_exclude_params)

        elif os.path.isdir(args.checkpoint):
            if args.checkpoint_mode in ["resume_from_best"]:
                logging.info("Loading best checkpoint in %s" % args.checkpoint)
                checkpoint_stats, filename = checkpoint_saver.restore_best(
                    directory=args.checkpoint,
                    model_and_loss=model_and_loss,
                    include_params=args.checkpoint_include_params,
                    exclude_params=args.checkpoint_exclude_params)

            elif args.checkpoint_mode in ["resume_from_latest"]:
                logging.info("Loading latest checkpoint in %s" % args.checkpoint)
                checkpoint_stats, filename = checkpoint_saver.restore_latest(
                    directory=args.checkpoint,
                    model_and_loss=model_and_loss,
                    include_params=args.checkpoint_include_params,
                    exclude_params=args.checkpoint_exclude_params)
            else:
                logging.info("Unknown checkpoint_restore '%s' given!" % args.checkpoint_restore)
                quit()
        else:
            logging.info("Could not find checkpoint file or directory '%s'" % args.checkpoint)
            quit()

    return checkpoint_saver, checkpoint_stats


# -------------------------------------------------------------------------------------------------
# Configure data loading
# -------------------------------------------------------------------------------------------------
def configure_data_loaders(args):
    with logger.LoggingBlock("Datasets", emph=True):

        def _sizes_to_str(value):
            if np.isscalar(value):
                return '[1L]'
            else:
                return ' '.join([str([d for d in value.size()])])

        def _log_statistics(dataset, prefix, name):
            with logger.LoggingBlock("%s Dataset: %s" % (prefix, name)):
                example_dict = dataset[0]  # get sizes from first dataset example
                for key, value in sorted(example_dict.items()):
                    if key in ["index", "basename"]:  # no need to display these
                        continue
                    if isinstance(value, str):
                        logging.info("{}: {}".format(key, value))
                    else:
                        logging.info("%s: %s" % (key, _sizes_to_str(value)))
                logging.info("num_examples: %i" % len(dataset))

        # -----------------------------------------------------------------------------------------
        # GPU parameters
        # -----------------------------------------------------------------------------------------
        gpuargs = {"num_workers": args.num_workers, "pin_memory": True} if args.cuda else {}

        train_loader = None
        validation_loader = None
        inference_loader = None

        # -----------------------------------------------------------------------------------------
        # Training dataset
        # -----------------------------------------------------------------------------------------
        if args.training_dataset is not None:

            # ----------------------------------------------
            # Figure out training_dataset arguments
            # ----------------------------------------------
            kwargs = tools.kwargs_from_args(args, "training_dataset")
            kwargs["is_cropped"] = True
            kwargs["args"] = args

            # ----------------------------------------------
            # Create training dataset
            # ----------------------------------------------
            train_dataset = tools.instance_from_kwargs(args.training_dataset_class, kwargs)

            # ----------------------------------------------
            # Create training loader
            # ----------------------------------------------            
            if args.training_dataset == 'KITTI_Comb_Train' or args.training_dataset == 'KITTI_Comb_Full' :
                custom_batch_sampler = CustomBatchSampler([RandomSampler(train_dataset.dataset1), RandomSampler(train_dataset.dataset2)])
                train_loader = DataLoader(dataset=train_dataset, batch_sampler=custom_batch_sampler, **gpuargs)

            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    **gpuargs)

            _log_statistics(train_dataset, prefix="Training", name=args.training_dataset)

        # -----------------------------------------------------------------------------------------
        # Validation dataset
        # -----------------------------------------------------------------------------------------
        if args.validation_dataset is not None:

            # ----------------------------------------------
            # Figure out validation_dataset arguments
            # ----------------------------------------------
            kwargs = tools.kwargs_from_args(args, "validation_dataset")
            kwargs["is_cropped"] = True
            kwargs["args"] = args

            # ----------------------------------------------
            # Create validation dataset
            # ----------------------------------------------
            validation_dataset = tools.instance_from_kwargs(args.validation_dataset_class, kwargs)

            # ----------------------------------------------
            # Create validation loader
            # ----------------------------------------------
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=args.batch_size_val,
                shuffle=False,
                drop_last=False,
                **gpuargs)

            _log_statistics(validation_dataset, prefix="Validation", name=args.validation_dataset)

    return train_loader, validation_loader, inference_loader


# ------------------------------------------------------------
# Generator for trainable parameters by pattern matching
# ------------------------------------------------------------
def _print_trainable_params(model_and_loss, match="*"):
    sum = 0
    for name, p in model_and_loss.named_parameters():
        if fnmatch.fnmatch(name, match):
            if p.requires_grad:
                logging.info(name)
                logging.info(str(p.numel()))
                print(name)
                print(p.numel())
                sum += p.numel()
    logging.info(str(sum))

def _generate_trainable_params(model_and_loss, match="*"):
    for name, p in model_and_loss.named_parameters():
        if fnmatch.fnmatch(name, match):
            if p.requires_grad:
                yield p


def _param_names_and_trainable_generator(model_and_loss, match="*"):
    names = []
    for name, p in model_and_loss.named_parameters():
        if fnmatch.fnmatch(name, match):
            if p.requires_grad:
                names.append(name)

    return names, _generate_trainable_params(model_and_loss, match=match)


# -------------------------------------------------------------------------------------------------
# Build optimizer:
# -------------------------------------------------------------------------------------------------
def configure_optimizer(args, model_and_loss):
    optimizer = None
    with logger.LoggingBlock("Optimizer", emph=True):
        if args.optimizer is not None:
            if model_and_loss.num_parameters() == 0:
                logging.info("No trainable parameters detected.")
                logging.info("Setting optimizer to None.")
            else:
                logging.info(args.optimizer)

                # -------------------------------------------
                # Figure out all optimizer arguments
                # -------------------------------------------
                all_kwargs = tools.kwargs_from_args(args, "optimizer")

                # -------------------------------------------
                # Get the split of param groups
                # -------------------------------------------
                kwargs_without_groups = {
                    key: value for key,value in all_kwargs.items() if key != "group"
                }
                param_groups = all_kwargs["group"]

                # ----------------------------------------------------------------------
                # Print arguments (without groups)
                # ----------------------------------------------------------------------
                for param, default in sorted(kwargs_without_groups.items()):
                    logging.info("%s: %s" % (param, default))

                # ----------------------------------------------------------------------
                # Construct actual optimizer params
                # ----------------------------------------------------------------------
                kwargs = dict(kwargs_without_groups)
                if param_groups is None:
                    # ---------------------------------------------------------
                    # Add all trainable parameters if there is no param groups
                    # ---------------------------------------------------------
                    all_trainable_parameters = _generate_trainable_params(model_and_loss)
                    kwargs["params"] = all_trainable_parameters
                else:
                    # -------------------------------------------
                    # Add list of parameter groups instead
                    # -------------------------------------------
                    trainable_parameter_groups = []
                    dnames, dparams = _param_names_and_trainable_generator(model_and_loss)
                    dnames = set(dnames)
                    dparams = set(list(dparams))
                    with logger.LoggingBlock("parameter_groups:"):
                        for group in param_groups:
                            #  log group settings
                            group_match = group["params"]
                            group_args = {
                                key: value for key, value in group.items() if key != "params"
                            }

                            with logger.LoggingBlock("%s: %s" % (group_match, group_args)):
                                # retrieve parameters by matching name
                                gnames, gparams = _param_names_and_trainable_generator(
                                    model_and_loss, match=group_match)
                                # log all names affected
                                for n in sorted(gnames):
                                    logging.info(n)
                                # set generator for group
                                group_args["params"] = gparams
                                # append parameter group
                                trainable_parameter_groups.append(group_args)
                                # update remaining trainable parameters
                                dnames -= set(gnames)
                                dparams -= set(list(gparams))

                        # append default parameter group
                        trainable_parameter_groups.append({"params": list(dparams)})
                        # and log its parameter names
                        with logger.LoggingBlock("default:"):
                            for dname in sorted(dnames):
                                logging.info(dname)

                    # set params in optimizer kwargs
                    kwargs["params"] = trainable_parameter_groups

                # -------------------------------------------
                # Create optimizer instance
                # -------------------------------------------
                optimizer = tools.instance_from_kwargs(args.optimizer_class, kwargs)

    return optimizer


# -------------------------------------------------------------------------------------------------
# Configure learning rate scheduler
# -------------------------------------------------------------------------------------------------
def configure_lr_scheduler(args, optimizer):
    lr_scheduler = None

    with logger.LoggingBlock("Learning Rate Scheduler", emph=True):
        logging.info("class: %s" % args.lr_scheduler)

        if args.lr_scheduler is not None:

            # ----------------------------------------------
            # Figure out lr_scheduler arguments
            # ----------------------------------------------
            kwargs = tools.kwargs_from_args(args, "lr_scheduler")
            
            # -------------------------------------------
            # Print arguments
            # -------------------------------------------
            for param, default in sorted(kwargs.items()):
                logging.info("%s: %s" % (param, default))

            # -------------------------------------------
            # Add optimizer
            # -------------------------------------------
            kwargs["optimizer"] = optimizer

            # -------------------------------------------
            # Create lr_scheduler instance
            # -------------------------------------------
            lr_scheduler = tools.instance_from_kwargs(args.lr_scheduler_class, kwargs)

    return lr_scheduler
