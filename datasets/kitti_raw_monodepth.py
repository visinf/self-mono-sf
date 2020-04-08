from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict
from .common import kitti_crop_image_list, kitti_adjust_intrinsic
from .common import intrinsic_scale


class KITTI_Raw(data.Dataset):
    def __init__(self,
                 args,
                 images_root=None,
                 preprocessing_crop=False,                 
                 crop_size=[370, 1224],
                 num_examples=-1,
                 index_file=None):

        self._args = args
        self._seq_len = 1
        self._preprocessing_crop = preprocessing_crop
        self._crop_size = crop_size

        path_dir = os.path.dirname(os.path.realpath(__file__))
        path_index_file = os.path.join(path_dir, index_file)

        if not os.path.exists(path_index_file):
            raise ValueError("Index File '%s' not found!", path_index_file)
        index_file = open(path_index_file, 'r')

        ## loading image -----------------------------------
        if not os.path.isdir(images_root):
            raise ValueError("Image directory '%s' not found!")

        filename_list = [line.rstrip().split(' ') for line in index_file.readlines()]
        self._image_list = []
        view1 = 'image_02/data'
        view2 = 'image_03/data'
        ext = '.jpg'
        for item in filename_list:
            date = item[0][:10]
            scene = item[0]
            idx_src = item[1]
            for ii in range(self._seq_len):
                idx_tgt = '%.10d' % (int(idx_src) + ii + 1)
                name_l1 = os.path.join(images_root, date, scene, view1, idx_src) + ext
                name_r1 = os.path.join(images_root, date, scene, view2, idx_src) + ext
                if os.path.isfile(name_l1) and os.path.isfile(name_r1):
                    self._image_list.append([name_l1, name_r1])

        if num_examples > 0:
            self._image_list = self._image_list[:num_examples]

        self._size = len(self._image_list)

        ## loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}        
        self.intrinsic_dict_l, self.intrinsic_dict_r = read_calib_into_dict(path_dir)

        # ----------------------------------------------------------
        # Image resize only
        # ----------------------------------------------------------
        self._resize_to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.Resize((256, 512)),
            vision_transforms.transforms.ToTensor()
        ])
        self._to_tensor = vision_transforms.Compose([
            vision_transforms.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        index = index % self._size

        im_l1_filename = self._image_list[index][0]
        im_r1_filename = self._image_list[index][1]

        # read float32 images and flow
        im_l1_np = read_image_as_byte(im_l1_filename)
        im_r1_np = read_image_as_byte(im_r1_filename)

        # example filename
        basename = os.path.basename(im_l1_filename)[:6]
        dirname = os.path.dirname(im_l1_filename)[-51:]
        datename = dirname[:10]
        k_l1 = torch.from_numpy(self.intrinsic_dict_l[datename]).float()
        k_r1 = torch.from_numpy(self.intrinsic_dict_r[datename]).float()
        k_l1_orig = k_l1.clone()
        
        h_orig, w_orig, _ = im_l1_np.shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

        # resizing image 
        if self._preprocessing_crop == False:
            # No Geometric Augmentation, Resizing to 256 x 512 here
            # resizing input images
            im_l1 = self._resize_to_tensor(im_l1_np)
            im_r1 = self._resize_to_tensor(im_r1_np)
            # resizing intrinsic matrix            
            k_l1 = intrinsic_scale(k_l1, im_l1.size(1) / h_orig, im_l1.size(2) / w_orig)
            k_r1 = intrinsic_scale(k_r1, im_r1.size(1) / h_orig, im_r1.size(2) / w_orig)
        else:
            # For Geometric Augmentation, first croping the images to 370 x 1224 here, 
            # then do the augmentation in augmentation.py
            # get starting positions
            crop_height = self._crop_size[0]
            crop_width = self._crop_size[1]
            x = np.random.uniform(0, w_orig - crop_width + 1)
            y = np.random.uniform(0, h_orig - crop_height + 1)
            crop_info = [int(x), int(y), int(x + crop_width), int(y + crop_height)]

            # cropping images and adjust intrinsic accordingly
            im_l1_np, im_r1_np = kitti_crop_image_list([im_l1_np, im_r1_np], crop_info)
            im_l1 = self._to_tensor(im_l1_np)
            im_r1 = self._to_tensor(im_r1_np)
            k_l1, k_r1 = kitti_adjust_intrinsic(k_l1, k_r1, crop_info)
        
        # For CamCOnv
        k_r1_flip = k_r1.clone()
        k_r1_flip[0, 2] = im_r1.size(2) - k_r1_flip[0, 2]

        example_dict = {
            "input_l1": im_l1,
            "input_r1": im_r1,
            "index": index,
            "basename": basename,
            "datename": datename,
            "input_k_l1_orig": k_l1_orig,
            "input_k_l1": k_l1,
            "input_k_r1": k_r1,
            "input_k_r1_flip": k_r1_flip,
            "input_size": input_im_size
        }

        return example_dict

    def __len__(self):
        return self._size


class KITTI_Raw_KittiSplit_Train(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 preprocessing_crop=False,
                 crop_size=[370, 1224],
                 num_examples=-1):
        super(KITTI_Raw_KittiSplit_Train, self).__init__(
            args,
            images_root=root,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_train.txt")


class KITTI_Raw_KittiSplit_Valid(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 preprocessing_crop=False,
                 crop_size=[370, 1224],
                 num_examples=-1):
        super(KITTI_Raw_KittiSplit_Valid, self).__init__(
            args,
            images_root=root,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_valid.txt")