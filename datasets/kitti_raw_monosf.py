from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict
from .common import kitti_crop_image_list, kitti_adjust_intrinsic


class KITTI_Raw(data.Dataset):
    def __init__(self,
                 args,
                 images_root=None,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1,
                 index_file=None):

        self._args = args
        self._seq_len = 1
        self._flip_augmentations = flip_augmentations
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
            idx_tgt = '%.10d' % (int(idx_src) + 1)
            name_l1 = os.path.join(images_root, date, scene, view1, idx_src) + ext
            name_l2 = os.path.join(images_root, date, scene, view1, idx_tgt) + ext
            name_r1 = os.path.join(images_root, date, scene, view2, idx_src) + ext
            name_r2 = os.path.join(images_root, date, scene, view2, idx_tgt) + ext

            if os.path.isfile(name_l1) and os.path.isfile(name_l2) and os.path.isfile(name_r1) and os.path.isfile(name_r2):
                self._image_list.append([name_l1, name_l2, name_r1, name_r2])

        if num_examples > 0:
            self._image_list = self._image_list[:num_examples]

        self._size = len(self._image_list)

        ## loading calibration matrix
        self.intrinsic_dict_l = {}
        self.intrinsic_dict_r = {}        
        self.intrinsic_dict_l, self.intrinsic_dict_r = read_calib_into_dict(path_dir)

        self._to_tensor = vision_transforms.Compose([
            vision_transforms.ToPILImage(),
            vision_transforms.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        index = index % self._size

        # read images and flow
        # im_l1, im_l2, im_r1, im_r2
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
       
        common_dict = {
            "index": index,
            "basename": basename,
            "datename": datename,
            "input_size": input_im_size
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

    def __len__(self):
        return self._size



class KITTI_Raw_KittiSplit_Train(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1):
        super(KITTI_Raw_KittiSplit_Train, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_train.txt")


class KITTI_Raw_KittiSplit_Valid(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=False,
                 crop_size=[370, 1224],
                 num_examples=-1):
        super(KITTI_Raw_KittiSplit_Valid, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_valid.txt")


class KITTI_Raw_KittiSplit_Full(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1):
        super(KITTI_Raw_KittiSplit_Full, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/kitti_full.txt")


class KITTI_Raw_EigenSplit_Train(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1):
        super(KITTI_Raw_EigenSplit_Train, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/eigen_train.txt")


class KITTI_Raw_EigenSplit_Valid(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=False,
                 preprocessing_crop=False,
                 crop_size=[370, 1224],
                 num_examples=-1):
        super(KITTI_Raw_EigenSplit_Valid, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/eigen_valid.txt")


class KITTI_Raw_EigenSplit_Full(KITTI_Raw):
    def __init__(self,
                 args,
                 root,
                 flip_augmentations=True,
                 preprocessing_crop=True,
                 crop_size=[370, 1224],
                 num_examples=-1):
        super(KITTI_Raw_EigenSplit_Full, self).__init__(
            args,
            images_root=root,
            flip_augmentations=flip_augmentations,
            preprocessing_crop=preprocessing_crop,
            crop_size=crop_size,
            num_examples=num_examples,
            index_file="index_txt/eigen_full.txt")