from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np

from utils.interpolation import interpolate2d
from utils.interpolation import Interp2, Meshgrid


class PhotometricAugmentation(nn.Module):
    def __init__(self):
        super(PhotometricAugmentation, self).__init__()

        self._min_gamma = 0.8
        self._max_gamma = 1.2
        self._min_brght = 0.5
        self._max_brght = 2.0
        self._min_shift = 0.8
        self._max_shift = 1.2

        self._intv_gamma = self._max_gamma - self._min_gamma
        self._intv_brght = self._max_brght - self._min_brght
        self._intv_shift = self._max_shift - self._min_shift

    def forward(self, *args):

        _, orig_c, _, _ = args[0].size()
        num_splits = len(args)
        concat_data = torch.cat(args, dim=1)
        d_dtype = concat_data.dtype
        d_device = concat_data.device
        b, c, h, w = concat_data.size()
        num_images = int(c / orig_c)

        rand_gamma = torch.rand([b, 1, 1, 1], dtype=d_dtype, device=d_device, requires_grad=False) * self._intv_gamma + self._min_gamma
        rand_brightness = torch.rand([b, 1, 1, 1], dtype=d_dtype, device=d_device, requires_grad=False) * self._intv_brght + self._min_brght
        rand_shift = torch.rand([b, 3, 1, 1], dtype=d_dtype, device=d_device, requires_grad=False) * self._intv_shift + self._min_shift

        # gamma
        concat_data = concat_data ** rand_gamma.expand(-1, c, h, w)

        # brightness
        concat_data = concat_data * rand_brightness.expand(-1, c, h, w)

        # color shift
        rand_shift = rand_shift.expand(-1, -1, h, w)
        rand_shift = torch.cat([rand_shift for i in range(0, num_images)], dim=1)
        concat_data = concat_data * rand_shift

        # clip
        concat_data = torch.clamp(concat_data, 0, 1)

        split = torch.chunk(concat_data, num_splits, dim=1)

        return split


class _IdentityParams(nn.Module):
    def __init__(self):
        super(_IdentityParams, self).__init__()
        self._batch_size = 0
        self._device = None
        self._o = None
        self._i = None
        self._identity_params = None

    def _update(self, batch_size, device):
        self._o = torch.zeros([batch_size, 1, 1], device=device).float()
        self._i = torch.ones([batch_size, 1, 1], device=device).float()
        r1 = torch.cat([self._i, self._o, self._o], dim=2)
        r2 = torch.cat([self._o, self._i, self._o], dim=2)
        r3 = torch.cat([self._o, self._o, self._i], dim=2)
        return torch.cat([r1, r2, r3], dim=1)

    def forward(self, batch_size, device):
        if self._batch_size != batch_size or self._device != device:
            self._identity_params = self._update(batch_size, device)
            self._batch_size = batch_size
            self._device = device

        return self._identity_params.clone()


def _intrinsic_scale(intrinsic, sx, sy):
    out = intrinsic.clone()
    out[:, 0, 0] *= sx
    out[:, 0, 2] *= sx
    out[:, 1, 1] *= sy
    out[:, 1, 2] *= sy
    return out


def _intrinsic_crop(intrinsic, str_x, str_y):    
    out = intrinsic.clone()
    out[:, 0, 2] -= str_x
    out[:, 1, 2] -= str_y
    return out



######################################################

## MonoDepthBaseline
class Augmentation_MonoDepthBaseline(nn.Module):
    def __init__(self, args, photometric=True):
        super(Augmentation_MonoDepthBaseline, self).__init__()

        # init
        self._args = args
        self._photometric = photometric
        self._photo_augmentation = PhotometricAugmentation()

    def forward(self, example_dict):

        im_l1 = example_dict["input_l1"]
        im_r1 = example_dict["input_r1"]

        if self._photometric and torch.rand(1) > 0.5:
            im_l1, im_r1 = self._photo_augmentation(im_l1, im_r1)

        example_dict["input_l1"] = im_l1
        example_dict["input_r1"] = im_r1

        return example_dict


class Augmentation_ScaleCrop(nn.Module):
    def __init__(self, args, photometric=True, trans=0.07, scale=[0.93, 1.0], resize=[256, 832]):
        super(Augmentation_ScaleCrop, self).__init__()

        # init
        self._args = args
        self._photometric = photometric
        self._photo_augmentation = PhotometricAugmentation()

        self._batch = None
        self._device = None
        self._identity = _IdentityParams()
        self._meshgrid = Meshgrid()

        # Augmentation Parameters
        self._min_scale = scale[0]
        self._max_scale = scale[1]
        self._max_trans = trans
        self._resize = resize

    def compose_params(self, scale, rot, tx, ty):
        return torch.cat([scale, rot, tx, ty], dim=1)

    def decompose_params(self, params):
        return params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]

    def find_invalid(self, img_size, params):

        scale, _, tx, ty = self.decompose_params(params)

        ## Intermediate image
        intm_size_h = torch.floor(img_size[0] * scale)
        intm_size_w = torch.floor(img_size[1] * scale)

        ## 4 representative points of the intermediate images
        hf_h = (intm_size_h - 1.0) / 2.0
        hf_w = (intm_size_w - 1.0) / 2.0        
        hf_h.unsqueeze_(1)
        hf_w.unsqueeze_(1)
        hf_o = torch.zeros_like(hf_h)
        hf_i = torch.ones_like(hf_h)
        pt_mat = torch.cat([torch.cat([hf_w, hf_o, hf_o], dim=2), torch.cat([hf_o, hf_h, hf_o], dim=2), torch.cat([hf_o, hf_o, hf_i], dim=2)], dim=1)
        ref_mat = torch.ones(self._batch, 4, 3, device=self._device)
        ref_mat[:, 1, 1] = -1
        ref_mat[:, 2, 0] = -1
        ref_mat[:, 3, 0] = -1
        ref_mat[:, 3, 1] = -1
        ref_pts = torch.matmul(ref_mat, pt_mat).transpose(1,2)

        ## Perform trainsform
        tform_mat = self._identity(self._batch, self._device)
        tform_mat[:, 0, 2] = tx[:, 0]
        tform_mat[:, 1, 2] = ty[:, 0]   
        pts_tform = torch.matmul(tform_mat, ref_pts)

        ## Check validity: whether the 4 representative points are inside of the original images
        img_hf_h = (img_size[0] - 1.0) / 2.0
        img_hf_w = (img_size[1] - 1.0) / 2.0
        x_tf = pts_tform[:, 0, :]
        y_tf = pts_tform[:, 1, :]

        invalid = (((x_tf <= -img_hf_w) | (y_tf <= -img_hf_h) | (x_tf >= img_hf_w) | (y_tf >= img_hf_h)).sum(dim=1, keepdim=True) > 0).float()

        return invalid

    def calculate_tform_and_grids(self, img_size, resize, params):

        intm_scale, _, tx, ty = self.decompose_params(params)

        ## Intermediate image
        intm_size_h = torch.floor(img_size[0] * intm_scale)
        intm_size_w = torch.floor(img_size[1] * intm_scale)
        scale_x = intm_size_w / resize[1]
        scale_y = intm_size_h / resize[0]

        ## Coord of the resized image
        grid_ww, grid_hh = self._meshgrid(resize[1], resize[0])
        grid_ww = (grid_ww - (resize[1] - 1.0) / 2.0).unsqueeze(0).cuda()
        grid_hh = (grid_hh - (resize[0] - 1.0) / 2.0).unsqueeze(0).cuda()
        grid_pts = torch.cat([grid_ww, grid_hh, torch.ones_like(grid_hh)], dim=0).unsqueeze(0).expand(self._batch, -1, -1, -1)

        ## 1st - scale_tform -> to intermediate image
        scale_tform = self._identity(self._batch, self._device)
        scale_tform[:, 0, 0] = scale_x[:, 0]
        scale_tform[:, 1, 1] = scale_y[:, 0]
        pts_tform = torch.matmul(scale_tform, grid_pts.view(self._batch, 3, -1))

        ## 2st - trans and rotate -> to original image (each pixel contains the coordinates in the original images)
        tr_tform = self._identity(self._batch, self._device)
        tr_tform[:, 0, 2] = tx[:, 0]
        tr_tform[:, 1, 2] = ty[:, 0]
        pts_tform = torch.matmul(tr_tform, pts_tform).view(self._batch, 3, resize[0], resize[1])

        grid_img_ww = pts_tform[:, 0, :, :] / float(img_size[1]) * 2    # x2 is for scaling [-1. 1]
        grid_img_hh = pts_tform[:, 1, :, :] / float(img_size[0]) * 2

        grid_img = torch.cat([grid_img_ww.unsqueeze(3), grid_img_hh.unsqueeze(3)], dim=3)

        return grid_img


    def find_aug_params(self, img_size, resize):

        ## Init
        scale = torch.zeros(self._batch, 1, device=self._device)
        rot = torch.zeros_like(scale)
        tx = torch.zeros_like(scale)
        ty = torch.zeros_like(scale)

        params = self.compose_params(scale, rot, tx, ty)

        invalid = torch.ones_like(scale)
        max_trans = torch.ones_like(scale) * self._max_trans 

        ## find params
        # scale: for the size of intermediate images (original * scale = intermediate image)
        # rot and trans: rotating and translating of the intermedinate image
        # then resize the augmented images into the resize image

        while invalid.sum() > 0:

            scale.uniform_(self._min_scale, self._max_scale)
            max_t = torch.min(torch.ones_like(scale) - scale, max_trans) * 0.5  # 0.5 because the translation rage is [-0.5*trans, 0.5*trans]
            tx = tx.uniform_(-1.0, 1.0) * max_t * img_size[1]
            ty = ty.uniform_(-1.0, 1.0) * max_t * img_size[0]
            params_new = self.compose_params(scale, rot, tx, ty)
            params = invalid * params_new + (1 - invalid) * params

            invalid = self.find_invalid(img_size, params)

        return params

    def augment_intrinsic_matrices(self, intrinsics, num_splits, img_size, resize, params):

        ### Finding the starting pt in the Original Image

        intm_scale, _, tx, ty = self.decompose_params(params)

        ## Intermediate image: finding scale from "Resize" to "Intermediate Image"
        intm_size_h = torch.floor(img_size[0] * intm_scale)
        intm_size_w = torch.floor(img_size[1] * intm_scale)
        scale_x = intm_size_w / resize[1]
        scale_y = intm_size_h / resize[0]

        ## Coord of the resized image
        pt_o = torch.zeros([1, 1]).float()
        grid_ww = (pt_o - (resize[1] - 1.0) / 2.0).unsqueeze(0).cuda()
        grid_hh = (pt_o - (resize[0] - 1.0) / 2.0).unsqueeze(0).cuda()
        grid_pts = torch.cat([grid_ww, grid_hh, torch.ones_like(grid_hh)], dim=0).unsqueeze(0).expand(self._batch, -1, -1, -1)

        ## 1st - scale_tform -> to intermediate image
        scale_tform = self._identity(self._batch, self._device)
        scale_tform[:, 0, 0] = scale_x[:, 0]
        scale_tform[:, 1, 1] = scale_y[:, 0]
        pts_tform = torch.matmul(scale_tform, grid_pts.view(self._batch, 3, -1))

        ## 2st - trans and rotate -> to original image (each pixel contains the coordinates in the original images)
        tr_tform = self._identity(self._batch, self._device)
        tr_tform[:, 0, 2] = tx[:, 0]
        tr_tform[:, 1, 2] = ty[:, 0]
        pts_tform = torch.matmul(tr_tform, pts_tform)
        str_p_ww = pts_tform[:, 0, :] + torch.ones_like(pts_tform[:, 0, :]) * float(img_size[1]) * 0.5 
        str_p_hh = pts_tform[:, 1, :] + torch.ones_like(pts_tform[:, 1, :]) * float(img_size[0]) * 0.5

        ## Cropping
        intrinsics[:, :, 0, 2] -= str_p_ww[:, 0:1].expand(-1, num_splits)
        intrinsics[:, :, 1, 2] -= str_p_hh[:, 0:1].expand(-1, num_splits)

        ## Scaling        
        intrinsics[:, :, 0, 0] = intrinsics[:, :, 0, 0] / scale_x
        intrinsics[:, :, 1, 1] = intrinsics[:, :, 1, 1] / scale_y
        intrinsics[:, :, 0, 2] = intrinsics[:, :, 0, 2] / scale_x
        intrinsics[:, :, 1, 2] = intrinsics[:, :, 1, 2] / scale_y

        return intrinsics


class Augmentation_SceneFlow(Augmentation_ScaleCrop):
    def __init__(self, args, photometric=True, trans=0.07, scale=[0.93, 1.0], resize=[256, 832]):
        super(Augmentation_SceneFlow, self).__init__(
            args, 
            photometric=photometric, 
            trans=trans, 
            scale=scale, 
            resize=resize)


    def forward(self, example_dict):

        # --------------------------------------------------------
        # Param init
        # --------------------------------------------------------
        im_l1 = example_dict["input_l1"]
        im_l2 = example_dict["input_l2"]
        im_r1 = example_dict["input_r1"]
        im_r2 = example_dict["input_r2"]
        k_l1 = example_dict["input_k_l1"].clone()
        k_l2 = example_dict["input_k_l2"].clone()
        k_r1 = example_dict["input_k_r1"].clone()
        k_r2 = example_dict["input_k_r2"].clone()
        self._batch, _, h_orig, w_orig = im_l1.size()
        self._device = im_l1.device

        ## Finding out augmentation parameters
        params = self.find_aug_params([h_orig, w_orig], self._resize)
        coords = self.calculate_tform_and_grids([h_orig, w_orig], self._resize, params)
        params_scale, _, _, _ = self.decompose_params(params)

        ## Augment images
        im_l1 = tf.grid_sample(im_l1, coords)
        im_l2 = tf.grid_sample(im_l2, coords)
        im_r1 = tf.grid_sample(im_r1, coords)        
        im_r2 = tf.grid_sample(im_r2, coords)

        ## Augment intrinsic matrix         
        k_list = [k_l1.unsqueeze(1), k_l2.unsqueeze(1), k_r1.unsqueeze(1), k_r2.unsqueeze(1)]
        num_splits = len(k_list)
        intrinsics = torch.cat(k_list, dim=1)
        intrinsics = self.augment_intrinsic_matrices(intrinsics, num_splits, [h_orig, w_orig], self._resize, params)
        k_l1, k_l2, k_r1, k_r2 = torch.chunk(intrinsics, num_splits, dim=1)
        k_l1 = k_l1.squeeze(1)
        k_l2 = k_l2.squeeze(1)
        k_r1 = k_r1.squeeze(1)
        k_r2 = k_r2.squeeze(1)


        if self._photometric and torch.rand(1) > 0.5:
            im_l1, im_l2, im_r1, im_r2 = self._photo_augmentation(im_l1, im_l2, im_r1, im_r2)

        ## construct updated dictionaries        
        example_dict["input_coords"] = coords
        example_dict["input_aug_scale"] = params_scale
        
        example_dict["input_l1_aug"] = im_l1
        example_dict["input_l2_aug"] = im_l2
        example_dict["input_r1_aug"] = im_r1
        example_dict["input_r2_aug"] = im_r2
        
        example_dict["input_k_l1_aug"] = k_l1
        example_dict["input_k_l2_aug"] = k_l2
        example_dict["input_k_r1_aug"] = k_r1
        example_dict["input_k_r2_aug"] = k_r2
    
        k_l1_flip = k_l1.clone()
        k_l2_flip = k_l2.clone()
        k_r1_flip = k_r1.clone()
        k_r2_flip = k_r2.clone()
        k_l1_flip[:, 0, 2] = im_l1.size(3) - k_l1_flip[:, 0, 2]
        k_l2_flip[:, 0, 2] = im_l2.size(3) - k_l2_flip[:, 0, 2]
        k_r1_flip[:, 0, 2] = im_r1.size(3) - k_r1_flip[:, 0, 2]
        k_r2_flip[:, 0, 2] = im_r2.size(3) - k_r2_flip[:, 0, 2]
        example_dict["input_k_l1_flip_aug"] = k_l1_flip
        example_dict["input_k_l2_flip_aug"] = k_l2_flip
        example_dict["input_k_r1_flip_aug"] = k_r1_flip
        example_dict["input_k_r2_flip_aug"] = k_r2_flip

        aug_size = torch.zeros_like(example_dict["input_size"])
        aug_size[:, 0] = self._resize[0]
        aug_size[:, 1] = self._resize[1]
        example_dict["aug_size"] = aug_size

        return example_dict


class Augmentation_MonoDepth(Augmentation_ScaleCrop):
    def __init__(self, args, photometric=True, trans=0.1, scale=[0.9, 1.0], resize=[256, 512]):
        super(Augmentation_MonoDepth, self).__init__(
            args, 
            photometric=photometric, 
            trans=trans, 
            scale=scale, 
            resize=resize)

    def forward(self, example_dict):

        # --------------------------------------------------------
        # Param init
        # --------------------------------------------------------

        im_l1 = example_dict["input_l1"]
        im_r1 = example_dict["input_r1"]
        k_l1 = example_dict["input_k_l1"].clone()
        k_r1 = example_dict["input_k_r1"].clone()

        self._batch, _, h_orig, w_orig = im_l1.size()
        self._device = im_l1.device

        ## Finding out augmentation parameters
        params = self.find_aug_params([h_orig, w_orig], self._resize)
        coords = self.calculate_tform_and_grids([h_orig, w_orig], self._resize, params)
        params_scale, _, _, _ = self.decompose_params(params)

        ## Augment images
        im_l1 = tf.grid_sample(im_l1, coords)
        im_r1 = tf.grid_sample(im_r1, coords)

        ## Augment intrinsic matrix
        k_list = [k_l1.unsqueeze(1), k_r1.unsqueeze(1)]
        num_splits = len(k_list)
        intrinsics = torch.cat(k_list, dim=1)
        intrinsics = self.augment_intrinsic_matrices(intrinsics, num_splits, [h_orig, w_orig], self._resize, params)
        k_l1, k_r1 = torch.chunk(intrinsics, num_splits, dim=1)
        k_l1 = k_l1.squeeze(1)
        k_r1 = k_r1.squeeze(1)

        if self._photometric and torch.rand(1) > 0.5:
            im_l1, im_r1 = self._photo_augmentation(im_l1, im_r1)

        k_r1_flip = k_r1.clone()
        k_r1_flip[:, 0, 2] = im_r1.size(3) - k_r1_flip[:, 0, 2]

        example_dict["input_l1"] = im_l1
        example_dict["input_r1"] = im_r1
        example_dict["input_k_l1"] = k_l1
        example_dict["input_k_r1"] = k_r1
        example_dict["input_k_r1_flip"] = k_r1_flip

        return example_dict

## Only for finetuning. Because the sparse GT cannot be interpolated, we just use cropping
class Augmentation_SceneFlow_Finetuning(nn.Module):
    def __init__(self, args, photometric=True, imgsize=[256, 832]):
        super(Augmentation_SceneFlow_Finetuning, self).__init__()

        # init
        self._args = args
        self._photometric = photometric
        self._photo_augmentation = PhotometricAugmentation()
        self._imgsize = imgsize


    def cropping(self, img, str_x, str_y, end_x, end_y):

        return img[:, :, str_y:end_y, str_x:end_x]

    def kitti_random_crop(self, example_dict):

        im_l1 = example_dict["input_l1"]
        _, _, height, width = im_l1.size()
        
        scale = np.random.uniform(0.94, 1.00)        
        crop_height = int(scale * height)
        crop_width = int(scale * width)

        # get starting positions
        x = np.random.uniform(0, width - crop_width + 1)
        y = np.random.uniform(0, height - crop_height + 1)
        str_x = int(x)
        str_y = int(y)
        end_x = int(x + crop_width)
        end_y = int(y + crop_height)

        ## Cropping
        example_dict["input_l1"] = self.cropping(example_dict["input_l1"], str_x, str_y, end_x, end_y)
        example_dict["input_l2"] = self.cropping(example_dict["input_l2"], str_x, str_y, end_x, end_y)
        example_dict["input_r1"] = self.cropping(example_dict["input_r1"], str_x, str_y, end_x, end_y)
        example_dict["input_r2"] = self.cropping(example_dict["input_r2"], str_x, str_y, end_x, end_y)

        example_dict["target_flow"] = self.cropping(example_dict["target_flow"], str_x, str_y, end_x, end_y)
        example_dict["target_flow_mask"] = self.cropping(example_dict["target_flow_mask"], str_x, str_y, end_x, end_y)
        example_dict["target_flow_noc"] = self.cropping(example_dict["target_flow_noc"], str_x, str_y, end_x, end_y)
        example_dict["target_flow_mask_noc"] = self.cropping(example_dict["target_flow_mask_noc"], str_x, str_y, end_x, end_y)
        
        example_dict["target_disp"] = self.cropping(example_dict["target_disp"], str_x, str_y, end_x, end_y)
        example_dict["target_disp_mask"] = self.cropping(example_dict["target_disp_mask"], str_x, str_y, end_x, end_y)
        example_dict["target_disp2_occ"] = self.cropping(example_dict["target_disp2_occ"], str_x, str_y, end_x, end_y)
        example_dict["target_disp2_mask_occ"] = self.cropping(example_dict["target_disp2_mask_occ"], str_x, str_y, end_x, end_y)

        example_dict["target_disp_noc"] = self.cropping(example_dict["target_disp_noc"], str_x, str_y, end_x, end_y)
        example_dict["target_disp_mask_noc"] = self.cropping(example_dict["target_disp_mask_noc"], str_x, str_y, end_x, end_y)
        example_dict["target_disp2_noc"] = self.cropping(example_dict["target_disp2_noc"], str_x, str_y, end_x, end_y)
        example_dict["target_disp2_mask_noc"] = self.cropping(example_dict["target_disp2_mask_noc"], str_x, str_y, end_x, end_y)

        example_dict["input_k_l1"] = _intrinsic_crop(example_dict["input_k_l1"], str_x, str_y)
        example_dict["input_k_l2"] = _intrinsic_crop(example_dict["input_k_l2"], str_x, str_y)
        example_dict["input_k_r1"] = _intrinsic_crop(example_dict["input_k_r1"], str_x, str_y)
        example_dict["input_k_r2"] = _intrinsic_crop(example_dict["input_k_r2"], str_x, str_y)

        input_size = example_dict["input_size"].clone()
        input_size[:, 0] = crop_height
        input_size[:, 1] = crop_width
        example_dict["input_size"] = input_size

        return 


    def forward(self, example_dict):

        ## KITTI Random Crop
        self.kitti_random_crop(example_dict)

        # Image resizing
        im_l1 = interpolate2d(example_dict["input_l1"], self._imgsize)
        im_l2 = interpolate2d(example_dict["input_l2"], self._imgsize)
        im_r1 = interpolate2d(example_dict["input_r1"], self._imgsize)
        im_r2 = interpolate2d(example_dict["input_r2"], self._imgsize)

        # Focal length rescaling
        _, _, hh, ww = example_dict["input_l1"].size()
        sy = self._imgsize[0] / hh
        sx = self._imgsize[1] / ww

        k_l1 = _intrinsic_scale(example_dict["input_k_l1"], sx, sy)
        k_l2 = _intrinsic_scale(example_dict["input_k_l2"], sx, sy)
        k_r1 = _intrinsic_scale(example_dict["input_k_r1"], sx, sy)
        k_r2 = _intrinsic_scale(example_dict["input_k_r2"], sx, sy)

        if self._photometric and torch.rand(1) > 0.5:
            im_l1, im_l2, im_r1, im_r2 = self._photo_augmentation(im_l1, im_l2, im_r1, im_r2)

        example_dict["input_l1_aug"] = im_l1
        example_dict["input_l2_aug"] = im_l2
        example_dict["input_r1_aug"] = im_r1
        example_dict["input_r2_aug"] = im_r2

        example_dict["input_k_l1_aug"] = k_l1
        example_dict["input_k_l2_aug"] = k_l2
        example_dict["input_k_r1_aug"] = k_r1
        example_dict["input_k_r2_aug"] = k_r2

        k_l1_flip = k_l1.clone()
        k_l2_flip = k_l2.clone()
        k_r1_flip = k_r1.clone()
        k_r2_flip = k_r2.clone()
        k_l1_flip[:, 0, 2] = im_l1.size(3) - k_l1_flip[:, 0, 2]
        k_l2_flip[:, 0, 2] = im_l2.size(3) - k_l2_flip[:, 0, 2]
        k_r1_flip[:, 0, 2] = im_r1.size(3) - k_r1_flip[:, 0, 2]
        k_r2_flip[:, 0, 2] = im_r2.size(3) - k_r2_flip[:, 0, 2]
        example_dict["input_k_l1_flip_aug"] = k_l1_flip
        example_dict["input_k_l2_flip_aug"] = k_l2_flip
        example_dict["input_k_r1_flip_aug"] = k_r1_flip
        example_dict["input_k_r2_flip_aug"] = k_r2_flip

        aug_size = torch.zeros_like(example_dict["input_size"])
        aug_size[:, 0] = self._imgsize[0]
        aug_size[:, 1] = self._imgsize[1]
        example_dict["aug_size"] = aug_size

        return example_dict


class Augmentation_Resize_Only(nn.Module):
    def __init__(self, args, photometric=False, imgsize=[256, 832]):
        super(Augmentation_Resize_Only, self).__init__()

        # init
        self._args = args
        self._imgsize = imgsize
        self._isRight = False
        self._photometric = photometric
        self._photo_augmentation = PhotometricAugmentation()

    def forward(self, example_dict):

        if ('input_r1' in example_dict) and ('input_r2' in example_dict):
            self._isRight = True

        # Focal length rescaling
        _, _, hh, ww = example_dict["input_l1"].size()
        sy = self._imgsize[0] / hh
        sx = self._imgsize[1] / ww

        # Image resizing
        im_l1 = interpolate2d(example_dict["input_l1"], self._imgsize)
        im_l2 = interpolate2d(example_dict["input_l2"], self._imgsize)
        k_l1 = _intrinsic_scale(example_dict["input_k_l1"], sx, sy)
        k_l2 = _intrinsic_scale(example_dict["input_k_l2"], sx, sy)

        if self._isRight:
            im_r1 = interpolate2d(example_dict["input_r1"], self._imgsize)
            im_r2 = interpolate2d(example_dict["input_r2"], self._imgsize)
            k_r1 = _intrinsic_scale(example_dict["input_k_r1"], sx, sy)
            k_r2 = _intrinsic_scale(example_dict["input_k_r2"], sx, sy)


        if self._photometric and torch.rand(1) > 0.5:
            if self._isRight:
                im_l1, im_l2, im_r1, im_r2 = self._photo_augmentation(im_l1, im_l2, im_r1, im_r2)
            else:
                im_l1, im_l2 = self._photo_augmentation(im_l1, im_l2)


        example_dict["input_l1_aug"] = im_l1
        example_dict["input_l2_aug"] = im_l2
        example_dict["input_k_l1_aug"] = k_l1
        example_dict["input_k_l2_aug"] = k_l2

        if self._isRight:
            example_dict["input_r1_aug"] = im_r1
            example_dict["input_r2_aug"] = im_r2
            example_dict["input_k_r1_aug"] = k_r1
            example_dict["input_k_r2_aug"] = k_r2

        k_l1_flip = k_l1.clone()
        k_l2_flip = k_l2.clone()
        k_l1_flip[:, 0, 2] = im_l1.size(3) - k_l1_flip[:, 0, 2]
        k_l2_flip[:, 0, 2] = im_l2.size(3) - k_l2_flip[:, 0, 2]
        example_dict["input_k_l1_flip_aug"] = k_l1_flip
        example_dict["input_k_l2_flip_aug"] = k_l2_flip

        if self._isRight:
            k_r1_flip = k_r1.clone()
            k_r2_flip = k_r2.clone()
            k_r1_flip[:, 0, 2] = im_r1.size(3) - k_r1_flip[:, 0, 2]
            k_r2_flip[:, 0, 2] = im_r2.size(3) - k_r2_flip[:, 0, 2]
            example_dict["input_k_r1_flip_aug"] = k_r1_flip
            example_dict["input_k_r2_flip_aug"] = k_r2_flip

        aug_size = torch.zeros_like(example_dict["input_size"])
        aug_size[:, 0] = self._imgsize[0]
        aug_size[:, 1] = self._imgsize[1]
        example_dict["aug_size"] = aug_size

        return example_dict

