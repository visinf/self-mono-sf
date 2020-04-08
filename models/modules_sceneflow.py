from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging 

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms

def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    grids_cuda = grid.float().requires_grad_(False).cuda()
    return grids_cuda


class WarpingLayer_Flow(nn.Module):
    def __init__(self):
        super(WarpingLayer_Flow, self).__init__()

    def forward(self, x, flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(x.size(3) - 1, 1)
        flo_h = flow[:, 1] * 2 / max(x.size(2) - 1, 1)
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)        
        x_warp = tf.grid_sample(x, grid)

        mask = torch.ones(x.size(), requires_grad=False).cuda()
        mask = tf.grid_sample(mask, grid)
        mask = (mask >= 1.0).float()

        return x_warp * mask


class WarpingLayer_SF(nn.Module):
    def __init__(self):
        super(WarpingLayer_SF, self).__init__()
 
    def forward(self, x, sceneflow, disp, k1, input_size):

        _, _, h_x, w_x = x.size()
        disp = interpolate2d_as(disp, x) * w_x

        local_scale = torch.zeros_like(input_size)
        local_scale[:, 0] = h_x
        local_scale[:, 1] = w_x

        pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
        _, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

        grid = coord1.transpose(1, 2).transpose(2, 3)
        x_warp = tf.grid_sample(x, grid)

        mask = torch.ones_like(x, requires_grad=False)
        mask = tf.grid_sample(mask, grid)
        mask = (mask >= 1.0).float()

        return x_warp * mask


def initialize_msra(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass


def upsample_outputs_as(input_list, ref_list):
    output_list = []
    for ii in range(0, len(input_list)):
        output_list.append(interpolate2d_as(input_list[ii], ref_list[ii]))

    return output_list


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        return self.conv1(x)


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class MonoSceneFlowDecoder(nn.Module):
    def __init__(self, ch_in):
        super(MonoSceneFlowDecoder, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_sf = conv(32, 3, isReLU=False)
        self.conv_d1 = conv(32, 1, isReLU=False)

    def forward(self, x):
        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out)

        return x_out, sf, disp1


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1)
        )
        self.conv_sf = conv(32, 3, isReLU=False)
        self.conv_d1 = nn.Sequential(
            conv(32, 1, isReLU=False), 
            torch.nn.Sigmoid()
        )

    def forward(self, x):

        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out) * 0.3

        return sf, disp1
