#!/bin/bash

# experiments and datasets meta
KITTI_RAW_HOME=""
EXPERIMENTS_HOME=""

# model
MODEL=MonoDepth_CamConv

# save path
ALIAS="-noAug-"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL$ALIAS$TIME"
CHECKPOINT=None

# Loss and Augmentation
Train_Dataset=KITTI_Raw_KittiSplit_Train_monodepth
Train_Augmentation=Augmentation_MonoDepthBaseline
Train_Loss_Function=Loss_MonoDepth

Valid_Dataset=KITTI_Raw_KittiSplit_Valid_monodepth
Valid_Loss_Function=Loss_MonoDepth


# training configuration
python ../../main.py \
--batch_size=4 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[30, 40]" \
--model=$MODEL \
--num_workers=16 \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--save=$SAVE_PATH \
--total_epochs=50 \
--training_augmentation=$Train_Augmentation \
--training_augmentation_photometric=True \
--training_dataset=$Train_Dataset \
--training_dataset_root=$KITTI_RAW_HOME \
--training_dataset_preprocessing_crop=False \
--training_dataset_num_examples=-1 \
--training_key=total_loss \
--training_loss=$Train_Loss_Function \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$KITTI_RAW_HOME \
--validation_dataset_preprocessing_crop=False \
--validation_key=total_loss \
--validation_loss=$Valid_Loss_Function
