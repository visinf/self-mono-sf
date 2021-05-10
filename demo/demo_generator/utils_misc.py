import numpy as np
import torch
import skimage.io as io
import cv2

TAG_CHAR = np.array([202021.25], np.float32)
UNKNOWN_FLOW_THRESH = 1e7


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def flow_to_png_middlebury(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """

    flow = flow.transpose([1, 2, 0])
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad)) 
    # maxrad = 4

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def numpy2torch(array):
    assert(isinstance(array, np.ndarray))
    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array.copy()).float()


def get_pixelgrid(b, h, w, flow=None):

    grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
    grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w)    
    ones = torch.ones_like(grid_h)

    if flow is None:
        pixelgrid = torch.cat((grid_h, grid_v, ones), dim=1).float().requires_grad_(False)
    else:
        pixelgrid = torch.cat((grid_h + flow[:, 0:1, :, :], grid_v + flow[:, 1:2, :, :], ones), dim=1).float().requires_grad_(False)
    
    return pixelgrid


def pixel2pts(depth, intrinsic, flow=None):
    
    b, _, h, w = depth.size()    
    pixelgrid = get_pixelgrid(b, h, w, flow)

    depth_mat = depth.view(b, 1, -1)    
    pixel_mat = pixelgrid.view(b, 3, -1)

    pts_mat = torch.matmul(torch.inverse(intrinsic), pixel_mat) * depth_mat

    pts = pts_mat.view(b, -1, h, w)

    return pts, pixelgrid

def disp2depth_kitti(pred_disp, focal_length):
    pred_depth = focal_length * 0.54 / pred_disp
    pred_depth = torch.clamp(pred_depth, 1e-3, 80)
    return pred_depth

def pixel2pts_ms(output_disp, intrinsic, flow=None):
    focal_length = intrinsic[:, 0, 0]
    output_depth = disp2depth_kitti(output_disp, focal_length)
    pts, _ = pixel2pts(output_depth, intrinsic, flow)
    return pts


def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    grids_cuda = grid.float().requires_grad_(False)
    return grids_cuda


def read_png_disp(disp_file):
    disp_np = io.imread(disp_file).astype(np.uint16) / 256.0
    disp_np = np.expand_dims(disp_np, axis=2)
    mask_disp = (disp_np > 0).astype(np.float64)
    return disp_np

def read_png_flow(flow_file):
    flow = cv2.imread(flow_file, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)[:,:,::-1].astype(np.float64)
    flow, valid = flow[:, :, :2], flow[:, :, 2:]
    flow = (flow - 2**15) / 64.0
    return flow


