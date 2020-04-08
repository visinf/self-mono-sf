from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from .correlation_package.correlation import Correlation

from .modules_sceneflow import get_grid, WarpingLayer_SF
from .modules_sceneflow import initialize_msra, upsample_outputs_as
from .modules_sceneflow import conv, upconv
from .modules_sceneflow import FeatureExtractor, MonoSceneFlowDecoder

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, post_processing



class ContextNetwork_separate_base(nn.Module):
    def __init__(self, ch_in1, ch_in2):
        super(ContextNetwork_separate_base, self).__init__()

        self.convs_sf = nn.Sequential(
            conv(ch_in1, 89, 3, 1, 1),
            conv(89, 89, 3, 1, 2),
            conv(89, 89, 3, 1, 4),
            conv(89, 67, 3, 1, 8),
            conv(67, 44, 3, 1, 16),
            conv(44, 22, 3, 1, 1),
            conv(22, 3, isReLU=False)
        )

        self.convs_dp = nn.Sequential(
            conv(ch_in2, 89, 3, 1, 1),
            conv(89, 89, 3, 1, 2),
            conv(89, 89, 3, 1, 4),
            conv(89, 67, 3, 1, 8),
            conv(67, 44, 3, 1, 16),
            conv(44, 22, 3, 1, 1),
            conv(22, 1, isReLU=False),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x_sf, x_dp):

        sf = self.convs_sf(x_sf)
        dp = self.convs_dp(x_dp)
        return sf, dp

class ContextNetwork_separate(nn.Module):
    def __init__(self, ch_in1, ch_in2):
        super(ContextNetwork_separate, self).__init__()

        self.convs_sf = nn.Sequential(
            conv(ch_in1, 91, 3, 1, 1),
            conv(91, 91, 3, 1, 2),
            conv(91, 91, 3, 1, 4),
            conv(91, 69, 3, 1, 8),
            conv(69, 46, 3, 1, 16),
            conv(46, 23, 3, 1, 1),
            conv(23, 3, isReLU=False)
        )

        self.convs_dp = nn.Sequential(
            conv(ch_in2, 91, 3, 1, 1),
            conv(91, 91, 3, 1, 2),
            conv(91, 91, 3, 1, 4),
            conv(91, 69, 3, 1, 8),
            conv(69, 46, 3, 1, 16),
            conv(46, 23, 3, 1, 1),
            conv(23, 1, isReLU=False),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x_sf, x_dp):

        sf = self.convs_sf(x_sf)
        dp = self.convs_dp(x_dp)
        return sf, dp


class SplitDec1(nn.Module):
    def __init__(self, ch_in):
        super(SplitDec1, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 68)
        )
        self.conv_sf = conv(68, 16)
        self.conv_dp = conv(68, 16)

        self.conv_sf_out = conv(16, 3, isReLU=False)
        self.conv_dp_out = conv(16, 1, isReLU=False)

    def forward(self, x):
        x_out = self.convs(x)

        sf_out = self.conv_sf(x_out)
        dp_out = self.conv_dp(x_out)

        sf = self.conv_sf_out(sf_out)
        dp = self.conv_dp_out(dp_out)

        return sf_out, sf, dp_out, dp

class SplitDec2(nn.Module):
    def __init__(self, ch_in):
        super(SplitDec2, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 81)            
        )
        self.conv_sf = nn.Sequential(
            conv(81, 54),
            conv(54, 16)
        )
        self.conv_dp = nn.Sequential(
            conv(81, 54),
            conv(54, 16)
        )

        self.conv_sf_out = conv(16, 3, isReLU=False)
        self.conv_dp_out = conv(16, 1, isReLU=False)

    def forward(self, x):
        x_out = self.convs(x)

        sf_out = self.conv_sf(x_out)
        dp_out = self.conv_dp(x_out)

        sf = self.conv_sf_out(sf_out)
        dp = self.conv_dp_out(dp_out)

        return sf_out, sf, dp_out, dp

class SplitDec3(nn.Module):
    def __init__(self, ch_in):
        super(SplitDec3, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 101)            
        )
        self.conv_sf = nn.Sequential(
            conv(101, 75),
            conv(75, 51),
            conv(51, 16)
        )
        self.conv_dp = nn.Sequential(
            conv(101, 75),
            conv(75, 51),
            conv(51, 16)
        )

        self.conv_sf_out = conv(16, 3, isReLU=False)
        self.conv_dp_out = conv(16, 1, isReLU=False)

    def forward(self, x):
        x_out = self.convs(x)

        sf_out = self.conv_sf(x_out)
        dp_out = self.conv_dp(x_out)

        sf = self.conv_sf_out(sf_out)
        dp = self.conv_dp_out(dp_out)

        return sf_out, sf, dp_out, dp

class SplitDec4(nn.Module):
    def __init__(self, ch_in):
        super(SplitDec4, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 100),            
        )
        self.conv_sf = nn.Sequential(
            conv(100, 100),
            conv(100, 75),
            conv(75, 52),
            conv(52, 16)
        )
        self.conv_dp = nn.Sequential(
            conv(100, 100),
            conv(100, 75),
            conv(75, 52),
            conv(52, 16)
        )

        self.conv_sf_out = conv(16, 3, isReLU=False)
        self.conv_dp_out = conv(16, 1, isReLU=False)

    def forward(self, x):
        x_out = self.convs(x)

        sf_out = self.conv_sf(x_out)
        dp_out = self.conv_dp(x_out)

        sf = self.conv_sf_out(sf_out)
        dp = self.conv_dp_out(dp_out)

        return sf_out, sf, dp_out, dp

class SplitDec5(nn.Module):
    def __init__(self, ch_in):
        super(SplitDec5, self).__init__()
        
        self.conv_sf = nn.Sequential(
            conv(ch_in, 80),
            conv(80, 80),
            conv(80, 62),
            conv(62, 41),
            conv(41, 16)
        )
        self.conv_dp = nn.Sequential(
            conv(ch_in, 80),
            conv(80, 80),
            conv(80, 62),
            conv(62, 41),
            conv(41, 16)
        )

        self.conv_sf_out = conv(16, 3, isReLU=False)
        self.conv_dp_out = conv(16, 1, isReLU=False)

    def forward(self, x):

        sf_out = self.conv_sf(x)
        dp_out = self.conv_dp(x)

        sf = self.conv_sf_out(sf_out)
        dp = self.conv_dp_out(dp_out)

        return sf_out, sf, dp_out, dp

class SplitDec6(nn.Module):
    def __init__(self, ch_in1, ch_in2):
        super(SplitDec6, self).__init__()
        
        self.conv_sf = nn.Sequential(
            conv(ch_in1, 88),
            conv(88, 88),
            conv(88, 67),
            conv(67, 44),
            conv(44, 16)
        )
        self.conv_dp = nn.Sequential(
            conv(ch_in2, 88),
            conv(88, 88),
            conv(88, 67),
            conv(67, 44),
            conv(44, 16)
        )

        self.conv_sf_out = conv(16, 3, isReLU=False)
        self.conv_dp_out = conv(16, 1, isReLU=False)

    def forward(self, x_sf, x_dp):

        sf_out = self.conv_sf(x_sf)
        dp_out = self.conv_dp(x_dp)

        sf = self.conv_sf_out(sf_out)
        dp = self.conv_dp_out(dp_out)

        return sf_out, sf, dp_out, dp


class SceneFlow_pwcnet_split_base(nn.Module):
    def __init__(self, args):
        super(SceneFlow_pwcnet_split_base, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_SF()
        
        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1
                self.upconv_layers.append(upconv(32, 32, 3, 2))

            layer_disp = MonoSceneFlowDecoder(num_ch_in)            
            self.flow_estimators.append(layer_disp)            

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}
        self.context_networks = ContextNetwork_separate_base(32+3, 32+1)
        self.sigmoid = torch.nn.Sigmoid()

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
    
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        flows_f = []
        flows_b = []
        disps_l1 = []
        disps_l2 = []


        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)
                x2_warp = self.warping_layer_sf(x2, flow_f, disp_l1, k1, input_dict['aug_size'])  # becuase K can be changing when doing augmentation
                x1_warp = self.warping_layer_sf(x1, flow_b, disp_l2, k2, input_dict['aug_size'])

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # flow estimator
            if l == 0:
                x1_out, flow_f, disp_l1 = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x2_out, flow_b, disp_l2 = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
            else:
                x1_out, flow_f_res, disp_l1 = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1], dim=1))
                x2_out, flow_b_res, disp_l2 = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2], dim=1))
                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                flows_f.append(flow_f)
                flows_b.append(flow_b)                
                disps_l1.append(disp_l1)
                disps_l2.append(disp_l2)
            else:
                flow_res_f, disp_l1 = self.context_networks(torch.cat([x1_out, flow_f], dim=1), torch.cat([x1_out, disp_l1], dim=1))
                flow_res_b, disp_l2 = self.context_networks(torch.cat([x2_out, flow_b], dim=1), torch.cat([x2_out, disp_l2], dim=1)) 
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b
                flows_f.append(flow_f)
                flows_b.append(flow_b)
                disps_l1.append(disp_l1)
                disps_l2.append(disp_l2)                
                break


        x1_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(flows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(flows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_l1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_l2[::-1], x1_rev)
                
        return output_dict

    def forward(self, input_dict):

        output_dict = {}

        ## Left
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])
        
        ## Right
        if not self._args.evaluation:
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])

            output_dict['output_dict_r'] = output_dict_r


        ## Eval
        if self._args.evaluation:
            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict


class SceneFlow_pwcnet_split_n(nn.Module):
    def __init__(self, args):
        super(SceneFlow_pwcnet_split_n, self).__init__()

        self._args = args
        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer_sf = WarpingLayer_SF()
        
        self.sf_estimators = nn.ModuleList()
        self.upconv_layers_sf = nn.ModuleList()
        self.upconv_layers_dp = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1
                self.upconv_layers_sf.append(upconv(16, 16, 3, 2))
                self.upconv_layers_dp.append(upconv(16, 16, 3, 2))

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.sigmoid = torch.nn.Sigmoid()

        self.context_networks = ContextNetwork_separate(16+3, 16+1)

        initialize_msra(self.modules())

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}        
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        flows_f = []
        flows_b = []
        disps_l1 = []
        disps_l2 = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                sf_f = interpolate2d_as(sf_f, x1, mode="bilinear")
                sf_b = interpolate2d_as(sf_b, x1, mode="bilinear")
                dp_f = interpolate2d_as(dp_f, x1, mode="bilinear")
                dp_b = interpolate2d_as(dp_b, x1, mode="bilinear")

                sf_f_out = self.upconv_layers_sf[l-1](sf_f_out)
                sf_b_out = self.upconv_layers_sf[l-1](sf_b_out)
                dp_f_out = self.upconv_layers_dp[l-1](dp_f_out)
                dp_b_out = self.upconv_layers_dp[l-1](dp_b_out)

                x2_warp = self.warping_layer_sf(x2, sf_f, dp_f, k1, input_dict['aug_size'])  # becuase K can be changing when doing augmentation
                x1_warp = self.warping_layer_sf(x1, sf_b, dp_b, k2, input_dict['aug_size'])

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # flow estimator
            if l == 0:                
                sf_f_out, sf_f, dp_f_out, dp_f = self.sf_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                sf_b_out, sf_b, dp_b_out, dp_b = self.sf_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
            else:
                sf_f_out, sf_f_res, dp_f_out, dp_f = self.sf_estimators[l](torch.cat([out_corr_relu_f, x1, sf_f_out, sf_f, dp_f_out, dp_f], dim=1))
                sf_b_out, sf_b_res, dp_b_out, dp_b = self.sf_estimators[l](torch.cat([out_corr_relu_b, x2, sf_b_out, sf_b, dp_b_out, dp_b], dim=1))
                sf_f = sf_f + sf_f_res
                sf_b = sf_b + sf_b_res

            # upsampling or post-processing
            if l != self.output_level:
                dp_f = self.sigmoid(dp_f) * 0.3
                dp_b = self.sigmoid(dp_b) * 0.3
                flows_f.append(sf_f)
                flows_b.append(sf_b)
                disps_l1.append(dp_f)
                disps_l2.append(dp_b)
            else:
                sf_res_f, dp_f = self.context_networks(torch.cat([sf_f_out, sf_f], dim=1), torch.cat([dp_f_out, dp_f], dim=1))
                sf_res_b, dp_b = self.context_networks(torch.cat([sf_b_out, sf_b], dim=1), torch.cat([dp_b_out, dp_b], dim=1))
                sf_f = sf_f + sf_res_f
                sf_b = sf_b + sf_res_b
                flows_f.append(sf_f)
                flows_b.append(sf_b)
                disps_l1.append(dp_f)
                disps_l2.append(dp_b)
                break

        x1_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(flows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(flows_b[::-1], x1_rev)
        output_dict['disp_l1'] = upsample_outputs_as(disps_l1[::-1], x1_rev)
        output_dict['disp_l2'] = upsample_outputs_as(disps_l2[::-1], x1_rev)
        
        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])
        
        ## Right
        if not self._args.evaluation:
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])
                output_dict_r['disp_l1'][ii] = torch.flip(output_dict_r['disp_l1'][ii], [3])
                output_dict_r['disp_l2'][ii] = torch.flip(output_dict_r['disp_l2'][ii], [3])

            output_dict['output_dict_r'] = output_dict_r


        ## Eval
        if self._args.evaluation:
            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []

            for ii in range(0, len(output_dict_flip['flow_f'])):

                flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp

        return output_dict


class SceneFlow_pwcnet_split1(SceneFlow_pwcnet_split_n):
    def __init__(self, args):
        super(SceneFlow_pwcnet_split1, self).__init__(args)

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1

            self.sf_estimators.append(SplitDec1(num_ch_in))


class SceneFlow_pwcnet_split2(SceneFlow_pwcnet_split_n):
    def __init__(self, args):
        super(SceneFlow_pwcnet_split2, self).__init__(args)

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1

            self.sf_estimators.append(SplitDec2(num_ch_in))


class SceneFlow_pwcnet_split3(SceneFlow_pwcnet_split_n):
    def __init__(self, args):
        super(SceneFlow_pwcnet_split3, self).__init__(args)

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1

            self.sf_estimators.append(SplitDec3(num_ch_in))


class SceneFlow_pwcnet_split4(SceneFlow_pwcnet_split_n):
    def __init__(self, args):
        super(SceneFlow_pwcnet_split4, self).__init__(args)

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1

            self.sf_estimators.append(SplitDec4(num_ch_in))


class SceneFlow_pwcnet_split5(SceneFlow_pwcnet_split_n):
    def __init__(self, args):
        super(SceneFlow_pwcnet_split5, self).__init__(args)

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break

            if l == 0:
                num_ch_in = self.dim_corr + ch 
            else:
                num_ch_in = self.dim_corr + ch + 32 + 3 + 1

            self.sf_estimators.append(SplitDec5(num_ch_in))
