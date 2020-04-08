from __future__ import absolute_import, division, print_function

import os.path
import torch
import torch.utils.data as data
import numpy as np

from torchvision import transforms as vision_transforms
from .common import read_image_as_byte, read_calib_into_dict



class KITTI_Eigen_Test(data.Dataset):
    def __init__(self,
                 args,
                 root,
                 num_examples=-1):

        self._args = args
        
        index_file = "index_txt/eigen_text.txt"


        path_dir = os.path.dirname(os.path.realpath(__file__))
        path_index_file = os.path.join(path_dir, index_file)

        if not os.path.exists(path_index_file):
            raise ValueError("Index File '%s' not found!", path_index_file)
        index_file = open(path_index_file, 'r')

        ## loading image -----------------------------------
        if not os.path.isdir(root):
            raise ValueError("Image directory '%s' not found!", root)

        filename_list = [line.rstrip().split(' ') for line in index_file.readlines()]
        self._image_list = []
        
        view1 = 'image_02/data'
        view2 = 'image_03/data'
        ext = '.jpg'
        for item in filename_list:

            name_l1 = root + '/' + item[0]
            name_depth = (root + '/' + item[0]).replace("jpg", "npy").replace("image_02", "projected_depth")
            idx_src = item[0].split('/')[4].split('.')[0]
            idx_tgt = '%.10d' % (int(idx_src) + 1)            
            name_l2 = name_l1.replace(idx_src, idx_tgt)
            if not os.path.isfile(name_l2):
                idx_prev = '%.10d' % (int(idx_src) - 1)
                name_l2 = name_l1.replace(idx_src, idx_prev)            

            if os.path.isfile(name_l1) and os.path.isfile(name_l2) and os.path.isfile(name_depth):
                self._image_list.append([name_l1, name_l2, name_depth])

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

        im_l1_filename = self._image_list[index][0]
        im_l2_filename = self._image_list[index][1]
        depth_filename = self._image_list[index][2]

        # read images and flow
        im_l1_np = read_image_as_byte(im_l1_filename)
        im_l2_np = read_image_as_byte(im_l2_filename)
        im_l1_depth_np = np.load(depth_filename)
        
        # example filename
        basename = os.path.dirname(im_l1_filename).split('/')[-3] + '_' + os.path.basename(im_l1_filename).split('.')[0]
        dirname = os.path.dirname(im_l1_filename)[-51:]
        datename = dirname[:10]

        k_l1 = torch.from_numpy(self.intrinsic_dict_l[datename]).float()
        k_r1 = torch.from_numpy(self.intrinsic_dict_r[datename]).float()

        im_l1 = self._to_tensor(im_l1_np)
        im_l2 = self._to_tensor(im_l2_np)
        im_l1_depth = torch.from_numpy(im_l1_depth_np).unsqueeze(0).float()

        # input size
        h_orig, w_orig, _ = im_l1_np.shape
        input_im_size = torch.from_numpy(np.array([h_orig, w_orig])).float()

    
        example_dict = {
            "input_l1": im_l1,
            "input_l2": im_l2,
            "index": index,
            "basename": basename,
            "datename": datename,
            "input_k_l1": k_l1,
            "input_k_l2": k_l1,
            "input_size": input_im_size,
            "target_depth": im_l1_depth
        }

        return example_dict

    def __len__(self):
        return self._size
