from __future__ import absolute_import, division, print_function

import torch
from torch import nn
import torch.nn.functional as tf


def disp_post_processing(disp):
    b, _, h, w = disp.shape
    b_h = int(b/2)

    l_disp = disp[0:b_h, :, :, :]
    r_disp = torch.flip(disp[b_h:, :, :, :], [3])
    m_disp = 0.5 * (l_disp + r_disp)
    grid_l = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w).float().requires_grad_(False).cuda()
    l_mask = 1.0 - torch.clamp(20 * (grid_l - 0.05), 0, 1)
    r_mask = torch.flip(l_mask, [3])
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def compute_errors(gt, pred):
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_d1_all(gt_disps, disp_t, gt_mask):
    disp_diff = torch.abs(gt_disps[gt_mask] - disp_t[gt_mask])
    bad_pixels = (disp_diff >= 3) & ((disp_diff / gt_disps[gt_mask]) >= 0.05)
    d1_all = 100.0 * bad_pixels.sum().float() / gt_mask.sum().float()

    return d1_all
