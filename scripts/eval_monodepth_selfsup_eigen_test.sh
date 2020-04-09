#!/bin/bash

# DATASETS_HOME
KITTI_RAW_HOME=""
CHECKPOINT="checkpoints/full_model_eigen/checkpoint_eigen_split.ckpt"

# model
MODEL=MonoSceneFlow_fullmodel

Valid_Dataset=KITTI_Eigen_Test
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_MonoDepth_Eigen

# training configuration
SAVE_PATH="eval/monod_selfsup_eigen_test"
python ../main.py \
--batch_size=1 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--model=$MODEL \
--evaluation=True \
--num_workers=4 \
--save=$SAVE_PATH \
--start_epoch=1 \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$KITTI_RAW_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=ab_r \
#--save_disp=True \