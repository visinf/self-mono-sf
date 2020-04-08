#!/bin/bash

# datasets
KITTI_COMB_HOME=""
EXPERIMENTS_HOME=""

# model
MODEL=MonoSceneFlow_fullmodel

# save path
ALIAS="-kitti_ft-"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL$ALIAS$TIME"
CHECKPOINT="checkpoints/full_model_kitti/checkpoint_latest.ckpt"

# Loss and Augmentation
Train_Dataset=KITTI_Comb_Train
Train_Augmentation=Augmentation_SceneFlow_Finetuning
Train_Loss_Function=Loss_SceneFlow_SemiSupFinetune

Valid_Dataset=KITTI_Comb_Val
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_SceneFlow_KITTI_Train

# training configuration
python ../main.py \
--batch_size=4 \
--batch_size_val=1 \
--finetuning=True \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[125, 187, 250, 281, 312]" \
--model=$MODEL \
--num_workers=16 \
--optimizer=Adam \
--optimizer_lr=4e-5 \
--save=$SAVE_PATH \
--total_epochs=343 \
--training_augmentation=$Train_Augmentation \
--training_augmentation_photometric=True \
--training_dataset=$Train_Dataset \
--training_dataset_root=$KITTI_COMB_HOME \
--training_loss=$Train_Loss_Function \
--training_key=total_loss \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$KITTI_COMB_HOME \
--validation_key=sf \
--validation_loss=$Valid_Loss_Function \