#!/bin/bash

# DATASETS_HOME
KITTI_HOME=""
CHECKPOINT="checkpoints/abl1_monodepth_augmentation/checkpoint_basic.ckpt"

# model
MODEL=MonoDepth_Baseline

Valid_Dataset=KITTI_2015_Train_Full_monodepth
Valid_Loss_Function=Eval_MonoDepth

# training configuration
SAVE_PATH="eval/abl1_depth/basic"
python ../../main.py \
--batch_size=1 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--model=$MODEL \
--evaluation=True \
--num_workers=4 \
--save=$SAVE_PATH \
--start_epoch=1 \
--validation_dataset=$Valid_Dataset \
--validation_dataset_root=$KITTI_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=ab_r \
# --save_disp=True
