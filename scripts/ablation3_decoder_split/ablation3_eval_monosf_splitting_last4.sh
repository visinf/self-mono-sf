#!/bin/bash

# DATASETS_HOME
KITTI_HOME=""
CHECKPOINT="checkpoints/abl3_monosf_decoder_split/checkpoint_split_last4.ckpt"

# model
MODEL=MonoSceneFlow_Split_Last4

Valid_Dataset=KITTI_2015_Train_Full_mnsf
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_SceneFlow_KITTI_Train

# training configuration
SAVE_PATH="eval/abl3/last4"
python ../../main.py \
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
--validation_dataset_preprocessing_crop=False \
--validation_dataset_root=$KITTI_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=sf \
# --save_disp=True \
# --save_disp2=True \
# --save_flow=True
