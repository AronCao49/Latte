from copy import deepcopy
import os

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from PIL import Image

np.random.seed(42)
CUDA_VISIBLE_DEVICES = -1

label_rgb_mapping = [
    [0, 0, 0],          # 0
    [70, 130, 180],     # 1
    [70, 70, 70],       # 2
    [128, 64, 128],     # 3
    [244, 35, 232],     # 4
    [64, 64, 128],      # 5
    [107, 142, 35],     # 6
    [153, 153, 153],    # 7
    [0, 0, 142],        # 8
    [220, 220, 0],      # 9
    [220, 20, 60],      # 10
    [119, 11, 32],      # 11
    [0, 0, 230],        # 12
    [250, 170, 160],    # 13
    [128, 64, 64],      # 14
    [250, 170, 30],     # 15
    [152, 251, 152],    # 16
    [255, 0, 0],        # 17
    [0, 0, 70],         # 18
    [0, 60, 100],       # 19
    [0, 80, 100],       # 20
    [102, 102, 156],    # 21
    [102, 102, 156]     # 22
    ]

label_rgb_mapping = np.asarray(label_rgb_mapping)

# rgb_label_mapping = {np.asarray(v): k for k, v in label_rgb_mapping.items()}

def select_points_in_frustum(points_2d, x1, y1, x2, y2):
    """
    Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
    :param points_2d: point cloud projected into 2D
    :param points_3d: point cloud
    :param x1: left bound
    :param y1: upper bound
    :param x2: right bound
    :param y2: lower bound
    :return: points (2D and 3D) that are in the frustum
    """
    keep_ind = (points_2d[:, 0] > x1) * \
                (points_2d[:, 1] > y1) * \
                (points_2d[:, 0] < x2) * \
                (points_2d[:, 1] < y2)

    return keep_ind


def rgb_to_label(rgb_array):
    rgb_int_array = rgb_array
    # checking if all rgb values is included in mapping table
    rgb_unique = np.unique(rgb_int_array, axis=0)
    for i in range(rgb_unique.shape[0]):
        # print(rgb_unique[i])
        assert list(rgb_unique[i]) in label_rgb_mapping.values()
    
    # assign label based on rgb value
    label_array = np.zeros((rgb_array.shape[0], 1))
    for label, rgb in label_rgb_mapping.items():
        index = rgb_int_array == np.asarray(rgb)
        index = np.logical_and(np.logical_and(index[:,0], index[:,1]),
                               index[:,2])
        label_array[index] = label

    return label_array


def world_to_img(pc_array, intrin_mtx, img_size, return_idx=False):
    points_hcoords = np.concatenate([pc_array, np.ones([pc_array.shape[0], 1])], axis=1)
    img_points = (intrin_mtx @ points_hcoords.T).T
    img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)
    keep_index = select_points_in_frustum(img_points, 0, 0, *img_size)

    out = [pc_array[keep_index], np.fliplr(img_points[keep_index])]
    
    if return_idx:
        out.append(keep_index)
    
    return out


def depth2lidar(depth_dir, clsgt_dir, lidar_dir, label_dir, y_fov=30, total_points=20000):
    # create output dir
    if not os.path.exists(lidar_dir):
        os.makedirs(lidar_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # loop over the depth images in depth dir
    for depth_img_name in tqdm(sorted(os.listdir(depth_dir))):
        # filter out non-image file
        if ".png" not in depth_img_name:
            continue
        # preparing input and output path
        depth_img_pth = os.path.join(depth_dir, depth_img_name)
        lidar_name = depth_img_name.replace(".png", ".pcd")
        lidar_pth = os.path.join(lidar_dir, lidar_name)

        clsgt_img_pth = os.path.join(clsgt_dir, depth_img_name)
        label_name = depth_img_name.replace(".png", ".npy")
        label_pth = os.path.join(label_dir, label_name)

        # extract lidar from depth, by LiDAR Simulation sampling
        intr_mtx = o3d.camera.PinholeCameraIntrinsic()
        intr_mtx.set_intrinsics(1280, 760, 532.740352, 532.740352, 640.0, 380.0)
        depth_array =  np.asarray(cv2.imread(depth_img_pth, cv2.IMREAD_ANYDEPTH))
        depth_img = o3d.geometry.Image(depth_array.astype(np.uint16))
        label_array = np.asarray(cv2.imread(clsgt_img_pth, cv2.IMREAD_ANYDEPTH))
        # label_array = label_array // 1000

        pc = o3d.geometry.PointCloud.create_from_depth_image(depth_img, intr_mtx, depth_scale=100, depth_trunc=100)
        # compute the distance of points, then filter
        # pc_array = np.asarray(pc.points) / 100
        pc_array = np.asarray(pc.points)
        pc_dist = np.sqrt(np.square(pc_array[:,0]) +\
                        np.square(pc_array[:,1]) +\
                        np.square(pc_array[:,2]))
        index = np.where(pc_dist <= 100)
        pc_array = pc_array[index]
        pc_dist = pc_dist[index]

        # --------------------------------------------------------------------
        # Sampling strategy 1: LiDAR Simulation sample
        # --------------------------------------------------------------------
        xz_angle = np.rad2deg(np.arctan2(pc_array[:,1], 
                            np.sqrt(np.square(pc_array[:,0]) +\
                                    np.square(pc_array[:,2]))))
        # camera_fov_x = np.rad2deg(2 * np.arctan(1742/2/725.0087))
        camera_fov_y = np.rad2deg(2 * np.arctan(760/2/640))
        # select points spliting along y direction
        lidar_fov = y_fov
        assert lidar_fov <= camera_fov_y
        xz_angle_gap = np.around(lidar_fov / 63, decimals=1)
        loop_start = np.around(-lidar_fov/2, decimals=1)
        xz_angle = np.around(xz_angle, decimals=1)
        # print(xz_angle_min, xz_angle_max)
        select_points = []
        for i in range(63+1):
            curr_xz_angle = loop_start
            # print(curr_xz_angle)
            index = np.where(xz_angle == curr_xz_angle)
            select_points.append(pc_array[index])
            loop_start += xz_angle_gap
        pc_array = np.concatenate(select_points, axis=0)
        # print(pc_array.shape)

        rd_index = np.random.choice(pc_array.shape[0], total_points, replace=False)
        pc_array = pc_array[rd_index]

        # project 3d points back to 2d img
        intrin_mtx = np.array([[532.740352, 0, 640.000, 0],
                               [0, 532.740352, 380.000, 0],
                               [0, 0, 1, 0]
                              ])
        pc_array, img_points = world_to_img(pc_array, intrin_mtx, (label_array.shape[1], label_array.shape[0]))
        img_indices = img_points.astype(np.int32)
        label_array = label_array[img_indices[:,0], img_indices[:,1]]
        rgb_array = label_rgb_mapping[label_array]
        
        # --------------------------------------------------------------------
        # Sampling strategy 2: LiDAR Simulation sample
        # --------------------------------------------------------------------
        # index = np.random.choice(raw_pc_array.shape[0], 15000, replace=False)
        # pc_array = raw_pc_array[index]

        # construct pcd and store
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_array)
        pcd.colors = o3d.utility.Vector3dVector((rgb_array / 255).astype(np.float64))

        o3d.io.write_point_cloud(lidar_pth, pcd)
        np.save(label_pth, label_array)
        

def pcd2bin(lidar_dir, bin_dir):
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)
    
    for scan in sorted(os.listdir(lidar_dir)):
        pcd = o3d.io.read_point_cloud(os.path.join(lidar_dir, scan))
        points = np.asarray(pcd.points)[:, :3].astype(np.float32)
        points.tofile(os.path.join(bin_dir, scan.replace(".pcd", ".bin")))


if __name__ == "__main__":
    scene_list = ['RAND_CITYSCAPES']
    prefix = "latte/datasets/synthia"
    
    # loop over all depth in synthia
    for scene in scene_list:
        print("Processing data in {}".format(scene))
        depth_dir = os.path.join(prefix, scene, 'Depth/Depth')
        clsgt_dir = os.path.join(prefix, scene, 'parsed_LABELS')
        lidar_dir = os.path.join(prefix, scene, 'Lidar')
        label_dir = os.path.join(prefix, scene, 'Label')
        depth2lidar(depth_dir=depth_dir, clsgt_dir=clsgt_dir, lidar_dir=lidar_dir, label_dir=label_dir)
        
        lidar_dir = os.path.join(prefix, scene, 'Lidar')
        bin_dir = os.path.join(prefix, scene, 'bin')
        pcd2bin(lidar_dir, bin_dir)
    