from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf


class CamConvModule(nn.Module):
    def __init__(self, norm_const=256.0):
        super(CamConvModule, self).__init__()

        self._inputimg_size = None
        self._norm_const = norm_const

        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None

        self._grid_w = None
        self._grid_h = None
        self._norm_coord = None
        self._centered_coord = None
        self._fov_maps = None

    # Unsqueeze and Expand as
    def ue_as(self, input_tensor, target_as):
        return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(target_as.size()).clone()

    def interpolate2d(self, inputs, h, w, mode="bilinear"):
        return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)

    def calculate_CoordConv(self, x):

        grid_w = torch.linspace(0, x.size(3) - 1, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
        grid_h = torch.linspace(0, x.size(2) - 1, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))        
        self._grid_w = grid_w.float().requires_grad_(False).cuda()
        self._grid_h = grid_h.float().requires_grad_(False).cuda()
        norm_grid_w = self._grid_w / (x.size(3) - 1) * 2 - 1
        norm_grid_h = self._grid_h / (x.size(2) - 1) * 2 - 1
        self._norm_coord = torch.cat((norm_grid_w, norm_grid_h), dim=1)

        return None

    def calculate_CamConv(self):

        ## Centered coordinates    
        centered_coord_w = self._grid_w - self.ue_as(self._cx, self._grid_w) + 0.5
        centered_coord_h = self._grid_h - self.ue_as(self._cy, self._grid_h) + 0.5
        self._centered_coord = torch.cat((centered_coord_w / self._norm_const, centered_coord_h / self._norm_const), dim=1)

        ## 3) FOV maps
        fov_xx_channel = torch.atan(centered_coord_w / self.ue_as(self._fx, self._grid_w))
        fov_yy_channel = torch.atan(centered_coord_h / self.ue_as(self._fy, self._grid_h))
        self._fov_maps = torch.cat((fov_xx_channel, fov_yy_channel), dim=1)

        return None

    def initialize(self, intrinsic, input_img):

        self._fx = intrinsic[:, 0, 0]
        self._fy = intrinsic[:, 1, 1]
        self._cx = intrinsic[:, 0, 2]
        self._cy = intrinsic[:, 1, 2]
        self.calculate_CoordConv(input_img)
        self.calculate_CamConv()

        return None

    def forward(self, input_tensor, input_img=None, intrinsic=None):

        if input_img is not None:
            self.initialize(intrinsic, input_img)

        _, _, hh_t, ww_t = input_tensor.size()
        cam_conv_tensor = torch.cat((self._norm_coord, self._centered_coord, self._fov_maps), dim=1)
        cam_conv_tensor = self.interpolate2d(cam_conv_tensor, hh_t, ww_t, mode="bilinear")


        return torch.cat((cam_conv_tensor.detach_(), input_tensor), dim=1)
