from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte
from .common import kitti_crop_image_list, kitti_adjust_intrinsic

## Combining datasets
from .kitti_2015_train import KITTI_2015_MonoSceneFlow
from .kitti_raw_monosf import KITTI_Raw
from torch.utils.data.dataset import ConcatDataset



class KITTI_Raw_for_Finetune(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1,
                 index_file=""):
        super(KITTI_Raw_for_Finetune, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file=index_file)

    def __getitem__(self, index):
        index = index % self._size

        # read images and flow
        img_list_np = [read_image_as_byte(img) for img in self._image_list[index]]

        # example filename
        im_l1_filename = self._image_list[index][0]
        basename = os.path.basename(im_l1_filename)[:6]
        dirname = os.path.dirname(im_l1_filename)[-51:]
        datename = dirname[:10]
        k_l1 = torch.from_numpy(self.intrinsic_dict_l[datename]).float()
        k_r1 = torch.from_numpy(self.intrinsic_dict_r[datename]).float()
        
        # input size
        h_orig, w_orig, _ = img_list_np[0].shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

        # cropping 
        if self._preprocessing_crop:
            # get starting positions
            crop_height = self._crop_size[0]
            crop_width = self._crop_size[1]
            x = np.random.uniform(0, w_orig - crop_width + 1)
            y = np.random.uniform(0, h_orig - crop_height + 1)
            crop_info = [int(x), int(y), int(x + crop_width), int(y + crop_height)]

            # cropping images and adjust intrinsic accordingly
            img_list_np = kitti_crop_image_list(img_list_np, crop_info)
            k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)
        
        # to tensors
        img_list_tensor = [self._to_tensor(img) for img in img_list_np]
        im_l1 = img_list_tensor[0]
        im_l2 = img_list_tensor[1]
        im_r1 = img_list_tensor[2]
        im_r2 = img_list_tensor[3]

        void_tensor1 = im_l1[0:1, :, :] * 0
        void_tensor2 = im_l1[0:2, :, :] * 0
        
        common_dict = {
            "index": index,
            "basename": basename,
            "datename": datename,
            "input_size": input_im_size,
            "target_flow": void_tensor2,
            "target_flow_mask": void_tensor1,
            "target_flow_noc": void_tensor2,
            "target_flow_mask_noc": void_tensor1,
            "target_disp": void_tensor1,
            "target_disp_mask": void_tensor1,
            "target_disp2_occ": void_tensor1,
            "target_disp2_mask_occ": void_tensor1,
            "target_disp_noc": void_tensor1,
            "target_disp_mask_noc": void_tensor1,
            "target_disp2_noc": void_tensor1,
            "target_disp2_mask_noc": void_tensor1
        }

        # random flip
        if self._flip_augmentations is True and torch.rand(1) > 0.5:
            _, _, ww = im_l1.size()
            im_l1_flip = torch.flip(im_l1, dims=[2])
            im_l2_flip = torch.flip(im_l2, dims=[2])
            im_r1_flip = torch.flip(im_r1, dims=[2])
            im_r2_flip = torch.flip(im_r2, dims=[2])

            k_l1[0, 2] = ww - k_l1[0, 2]
            k_r1[0, 2] = ww - k_r1[0, 2]

            example_dict = {
                "input_l1": im_r1_flip,
                "input_r1": im_l1_flip,
                "input_l2": im_r2_flip,
                "input_r2": im_l2_flip,                
                "input_k_l1": k_r1,
                "input_k_r1": k_l1,
                "input_k_l2": k_r1,
                "input_k_r2": k_l1,
            }
            example_dict.update(common_dict)

        else:
            example_dict = {
                "input_l1": im_l1,
                "input_r1": im_r1,
                "input_l2": im_l2,
                "input_r2": im_r2,
                "input_k_l1": k_l1,
                "input_k_r1": k_r1,
                "input_k_l2": k_l1,
                "input_k_r2": k_r1,
            }
            example_dict.update(common_dict)

        return example_dict


class KITTI_Comb_Train(ConcatDataset):  
    def __init__(self, args, root):        
        
        self.dataset1 = KITTI_2015_MonoSceneFlow(
            args, 
            root + '/KITTI_flow/', 
            preprocessing_crop=True, 
            crop_size=[370, 1224], 
            dstype="train")

        self.dataset2 = KITTI_Raw_for_Finetune(
            args, 
            root + '/KITTI_raw_noPCL/',
            flip_augmentations=True,
            preprocessing_crop=True,
            crop_size=[370, 1224],
            num_examples=-1,
            index_file='index_txt/kitti_full.txt')
      
        super(KITTI_Comb_Train, self).__init__(
            datasets=[self.dataset1, self.dataset2])


class KITTI_Comb_Val(KITTI_2015_MonoSceneFlow):
    def __init__(self,
                 args,
                 root,
                 preprocessing_crop=False,
                 crop_size=[370, 1224]):
        super(KITTI_Comb_Val, self).__init__(
            args,
            data_root=root + '/KITTI_flow/',          
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            dstype="valid")



class KITTI_Comb_Full(ConcatDataset):  
    def __init__(self, args, root):        

        self.dataset1 = KITTI_2015_MonoSceneFlow(
            args, 
            root + '/KITTI_flow/', 
            preprocessing_crop=True,
            crop_size=[370, 1224], 
            dstype="full")

        self.dataset2 = KITTI_Raw_for_Finetune(
            args, 
            root + '/KITTI_raw_noPCL/',
            flip_augmentations=True,
            preprocessing_crop=True,
            crop_size=[370, 1224],
            num_examples=-1,
            index_file='index_txt/kitti_raw_all_imgs.txt')

        super(KITTI_Comb_Full, self).__init__(
            datasets=[self.dataset1, self.dataset2])



