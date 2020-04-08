from __future__ import absolute_import, division, print_function

import os.path
import torch
import numpy as np
import skimage.io as io
import png

width_to_date = dict()
width_to_date[1242] = '2011_09_26'
width_to_date[1224] = '2011_09_28'
width_to_date[1238] = '2011_09_29'
width_to_date[1226] = '2011_09_30'
width_to_date[1241] = '2011_10_03'


def get_date_from_width(width):
    return width_to_date[width]


def list_flatten(input_list):
    return [img for sub_list in input_list for img in sub_list]


def intrinsic_scale(mat, sy, sx):
    out = mat.clone()
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def kitti_adjust_intrinsic(k_l1, k_r1, crop_info):
    str_x = crop_info[0]
    str_y = crop_info[1]
    k_l1[0, 2] -= str_x
    k_l1[1, 2] -= str_y
    k_r1[0, 2] -= str_x
    k_r1[1, 2] -= str_y
    return k_l1, k_r1

def kitti_crop_image_list(img_list, crop_info):    
    str_x = crop_info[0]
    str_y = crop_info[1]
    end_x = crop_info[2]
    end_y = crop_info[3]

    transformed = [img[str_y:end_y, str_x:end_x, :] for img in img_list]

    return transformed


def numpy2torch(array):
    assert(isinstance(array, np.ndarray))
    if array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        array = np.expand_dims(array, axis=0)
    return torch.from_numpy(array.copy()).float()


def read_image_as_byte(filename):
    return io.imread(filename)


def read_png_flow(flow_file):
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow[:, :, 0:2], (1 - invalid_idx * 1)[:, :, None]


def read_png_disp(disp_file):
    disp_np = io.imread(disp_file).astype(np.uint16) / 256.0
    disp_np = np.expand_dims(disp_np, axis=2)
    mask_disp = (disp_np > 0).astype(np.float64)
    return disp_np, mask_disp

        
def read_raw_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def read_calib_into_dict(path_dir):

    calibration_file_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
    intrinsic_dict_l = {}
    intrinsic_dict_r = {}

    for ii, date in enumerate(calibration_file_list):
        file_name = "cam_intrinsics/calib_cam_to_cam_" + date + '.txt'
        file_name_full = os.path.join(path_dir, file_name)
        file_data = read_raw_calib_file(file_name_full)
        P_rect_02 = np.reshape(file_data['P_rect_02'], (3, 4))
        P_rect_03 = np.reshape(file_data['P_rect_03'], (3, 4))
        intrinsic_dict_l[date] = P_rect_02[:3, :3]
        intrinsic_dict_r[date] = P_rect_03[:3, :3]

    return intrinsic_dict_l, intrinsic_dict_r