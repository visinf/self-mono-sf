import os

import skimage.io as io
from skimage.color import rgb2gray
# from skimage.color import lab2rgb

import open3d as o3d
import numpy as np
import torch
import math

from utils_misc import flow_to_png_middlebury, read_png_flow, read_png_disp
from utils_misc import numpy2torch, pixel2pts_ms

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[1226] = 707.0912

cam_center_dict = dict()
cam_center_dict[1242] = [6.095593e+02, 1.728540e+02]
cam_center_dict[1241] = [6.071928e+02, 1.852157e+02]
cam_center_dict[1224] = [6.040814e+02, 1.805066e+02]
cam_center_dict[1238] = [6.003891e+02, 1.815122e+02]
cam_center_dict[1226] = [6.018873e+02, 1.831104e+02]


########
sampling = [4,20,25,35,40]
imgflag = 1 # 0 is image, 1 is flow
########



def get_pcd(img_idx, image_dir, result_dir, tt):

    idx_curr = '%06d' % (img_idx)

    im1_np0 = (io.imread(os.path.join(image_dir, "image_2/" + idx_curr + "_10.png")) / np.float32(255.0))[110:, :, :]
    flo_f_np0 = read_png_flow(os.path.join(result_dir, "flow/" + idx_curr + "_10.png"))[110:, :, :]
    disp1_np0 = read_png_disp(os.path.join(result_dir, "disp_0/" + idx_curr + "_10.png"))[110:, :, :]
    disp2_np0 = read_png_disp(os.path.join(result_dir, "disp_1/" + idx_curr + "_10.png"))[110:, :, :]

    im1 = numpy2torch(im1_np0).unsqueeze(0)
    disp1 = numpy2torch(disp1_np0).unsqueeze(0)
    disp_diff = numpy2torch(disp2_np0).unsqueeze(0)
    flo_f = numpy2torch(flo_f_np0).unsqueeze(0)

    _, _, hh, ww = im1.size()

    ## Intrinsic
    focal_length = width_to_focal[ww]
    cx = cam_center_dict[ww][0]
    cy = cam_center_dict[ww][1]

    k1_np = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    k1 = numpy2torch(k1_np)

    # Forward warping Pts1 using disp_change and flow
    pts1 = pixel2pts_ms(disp1, k1)
    pts1_warp = pixel2pts_ms(disp_diff, k1, flo_f)
    sf = pts1_warp - pts1

    ## Composing Image
    im1_np0_g = np.repeat(np.expand_dims(rgb2gray(im1_np0), axis=2), 3, axis=2)
    flow = torch.cat((sf[:, 0:1, :, :], sf[:, 2:3, :, :]), dim=1).data.cpu().numpy()[0, :, :, :]
    flow_img = flow_to_png_middlebury(flow) / np.float32(255.0)
    
    if imgflag == 0:
        flow_img = im1_np0
    else:
        flow_img = (flow_img * 0.75 + im1_np0_g * 0.25)
    
    ## Crop
    max_crop = (60, 0.7, 82)
    min_crop = (-60, -20, 0)

    x1 = -60
    x2 = 60
    y1 = 0.7
    y2 = -20
    z1 = 80
    z2 = 0
    pp1 = np.array([[x1, y1, z1]])
    pp2 = np.array([[x1, y1, z2]])
    pp3 = np.array([[x1, y2, z1]])
    pp4 = np.array([[x1, y2, z2]])
    pp5 = np.array([[x2, y1, z1]])
    pp6 = np.array([[x2, y1, z2]])
    pp7 = np.array([[x2, y2, z1]])
    pp8 = np.array([[x2, y2, z2]])
    bb_pts = np.concatenate((pp1, pp2, pp3, pp4, pp5, pp6, pp7, pp8), axis=0)
    wp = np.array([[1.0, 1.0, 1.0]])
    bb_colors = np.concatenate((wp, wp, wp, wp, wp, wp, wp, wp), axis=0)

    ## Open3D Vis
    pts1_tform = pts1 + sf*tt
    pts1_np = np.transpose(pts1_tform[0].view(3, -1).data.numpy(), (1, 0))
    pts1_np = np.concatenate((pts1_np, bb_pts), axis=0)
    pts1_color = np.reshape(flow_img, (hh * ww, 3))
    pts1_color = np.concatenate((pts1_color, bb_colors), axis=0)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts1_np)
    pcd1.colors = o3d.utility.Vector3dVector(pts1_color)

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_crop, max_crop)
    pcd1 = pcd1.crop(bbox)

    return pcd1


def custom_vis(imglist, kitti_data_dir, result_dir, vis_dir):

    custom_vis.index = 0
    custom_vis.trajectory = o3d.io.read_pinhole_camera_trajectory("cam_pose.json")
    custom_vis.vis = o3d.visualization.Visualizer()

    img_id = imglist[custom_vis.index]
    init_pcd = get_pcd(img_id, kitti_data_dir, result_dir, 0)
    custom_vis.prev_pcd = init_pcd

    def move_forward(vis):

        glb = custom_vis

        ## Capture
        depth = vis.capture_depth_float_buffer(False)
        image = vis.capture_screen_float_buffer(False)
        save_id = imglist[glb.index-1]
        file_name = ""

        if imgflag == 0:
            file_name = os.path.join(vis_dir, "{:06d}_{:02d}.png".format(save_id, glb.index))
        else:
            file_name = os.path.join(vis_dir, "{:06d}_{:02d}.png".format(save_id, glb.index))

        print(' ' + str(glb.index) + ' '+ str(save_id) + ' '+ file_name)
        io.imsave(file_name, np.asarray(image), check_contrast=False)

        ## Rendering
        max_d_x = 13
        max_d_y = 4
        
        if glb.index < sampling[0]:
            tt = 0
            rx = 0
            ry = 0
        elif glb.index < sampling[1]: # only rotation
            tt = 0 
            rad = 2 * 3.14159265359 / (sampling[1] - sampling[0]) * (glb.index - sampling[0])
            rx = max_d_x * math.sin(rad)
            ry = (max_d_y * math.cos(rad) - max_d_y)
        elif glb.index < sampling[2]:
            tt = 0
            rx = 0
            ry = 0
        elif glb.index < sampling[3]:
            tt = (glb.index - sampling[2]) / (sampling[3] - sampling[2]) 
            rx = 0
            ry = 0
        else:
            tt = 1
            rx = 0
            ry = 0

        img_id = imglist[glb.index]
        pcd = get_pcd(img_id, kitti_data_dir, result_dir, tt)
        glb.index = glb.index + 1

        vis.clear_geometries()
        vis.add_geometry(pcd)
        glb.prev_pcd = pcd

        ctr = vis.get_view_control()
        ctr.scale(-24)

        ctr.rotate(rx, 980.0  + ry, 0, 0)
        ctr.translate(-5, 0)

        if not glb.index < len(imglist):
            custom_vis.vis.register_animation_callback(None)

        return False

    vis = custom_vis.vis
    vis.create_window()
    vis.add_geometry(init_pcd)

    ctr = vis.get_view_control()
    ctr.scale(-24)
    ctr.rotate(0, 980.0, 0, 0)
    ctr.translate(-5, 0)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()

########################################################################

kitti_data_dir = "kitti_img"    ## raw KITTI image
result_dir = "results"          ## disp_0, disp_1, flow
vis_dir = "vis"                 ## visualization output folder

imglist = []

for ii in range(0, sampling[-1]):
    imglist.append(139)

custom_vis(imglist, kitti_data_dir, result_dir, vis_dir)