import numpy as np
import os
from glob import glob
from godard_evaluation_utils import *


data_path = '/fastdata/jhur/KITTI_raw/'
all_images = glob(data_path + '*/*/image_02/data/*.jpg')

for ii in range(len(all_images)):   
    all_images[ii] = all_images[ii].replace(data_path, "")

# num_samples = 697
# test_files = read_text_lines('test_files_eigen.txt')
gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(all_images, data_path)

# num_test = len(im_files)
num_samples = len(gt_files)
print(num_samples)
# for t_id in range(0, 2):    
for t_id in range(num_samples):    
    print(t_id)
    camera_id = cams[t_id]  # 2 is left, 3 is right
    depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
    # need to convert from disparity to depth
    focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
    
    npy_file_name = gt_files[t_id].replace("KITTI_raw", "KITTI_raw_depth").replace(".bin", ".npy").replace("velodyne_points", "projected_depth")
    npy_file_dir = os.path.dirname(npy_file_name)
    if not os.path.exists(npy_file_dir):
        os.makedirs(npy_file_dir)

    np.save(npy_file_name, depth)
