import matplotlib.pyplot as plt
import numpy as np
from latte.data.utils.turbo_cmap import interpolate_or_clip, turbo_colormap_data
import open3d as o3d
import copy


NUSCENES_LIDARSEG_COLOR_PALETTE = [  # RGB.
        (112, 128, 144),  # Slategrey
        (220, 20, 60),  # Crimson
        (255, 127, 80),  # Coral
        (255, 158, 0),  # Orange
        (233, 150, 70),  # Darksalmon
        (255, 61, 99),  # Red
        (0, 0, 230),  # Blue
        (47, 79, 79),  # Darkslategrey
        (255, 140, 0),  # Darkorange
        (255, 99, 71),  # Tomato
        (0, 207, 191),  # nuTonomy green
        (175, 0, 75),
        (75, 0, 75),
        (112, 180, 60),
        (222, 184, 135),  # Burlywood
        (0, 175, 0),  # Green
        (0, 0, 0),  # Black.
    ]

# all classes
NUSCENES_COLOR_PALETTE = [
    (255, 158, 0),  # car
    (255, 158, 0),  # truck
    (255, 158, 0),  # bus
    (255, 158, 0),  # trailer
    (255, 158, 0),  # construction_vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # motorcycle
    (255, 61, 99),  # bicycle
    (0, 0, 0),  # traffic_cone
    (0, 0, 0),  # barrier
    (200, 200, 200),  # background
]

# classes after merging (as used in xMUDA)
NUSCENES_COLOR_PALETTE_SHORT = [
    (255, 158, 0),  # vehicle
    (0, 0, 230),  # pedestrian
    (255, 61, 99),  # bike
    (0, 0, 0),  # traffic boundary
    (200, 200, 200),  # background
]

# all classes
A2D2_COLOR_PALETTE_SHORT = [
    (255, 0, 0),  # car
    (255, 128, 0),  # truck
    (182, 89, 6),  # bike
    (204, 153, 255),  # person
    (255, 0, 255),  # road
    (150, 150, 200),  # parking
    (180, 150, 200),  # sidewalk
    (241, 230, 255),  # building
    (147, 253, 194),  # nature
    (255, 246, 143),  # other-objects
    (0, 0, 0)  # ignore
]

# colors as defined in https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
SEMANTIC_KITTI_ID_TO_BGR = {  # bgr
  0: [0, 0, 0],
  1: [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0],
}
SEMANTIC_KITTI_COLOR_PALETTE = [SEMANTIC_KITTI_ID_TO_BGR[id] if id in SEMANTIC_KITTI_ID_TO_BGR.keys() else [0, 0, 0]
                                for id in range(list(SEMANTIC_KITTI_ID_TO_BGR.keys())[-1] + 1)]


# classes after merging (as used in xMUDA)
SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [180, 30, 80],  # truck
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [255, 150, 255],  # parking
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [150, 240, 255], # pole
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
SEMANTIC_KITTI_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in SEMANTIC_KITTI_COLOR_PALETTE_SHORT_BGR]

SYNTHIA_KITTI_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [150, 240, 255], # pole
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
SYNTHIA_KITTI_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in SYNTHIA_KITTI_COLOR_PALETTE_SHORT_BGR]

# SPVNAS color map
SEMANTIC_KITTI_COLOR_PALETTE_LONG_BGR = np.array([
    [245, 150, 100],
    [245, 230, 100],
    [150, 60, 30],
    [180, 30, 80],
    [255, 0, 0],
    [30, 30, 255],
    [200, 40, 255],
    [90, 30, 150],
    [255, 0, 255],
    [255, 150, 255],
    [75, 0, 75],
    [75, 0, 175],
    [0, 200, 255],
    [50, 120, 255],
    [0, 175, 0],
    [0, 60, 135],
    [80, 240, 150],
    [150, 240, 255],
    [0, 0, 255],
])
SEMANTIC_KITTI_COLOR_PALETTE_LONG = SEMANTIC_KITTI_COLOR_PALETTE_LONG_BGR[:, [2, 1, 0]]  # convert bgra to rgba

# classes after merging (as used in xMUDA)
WAYMO_COLOR_PALETTE_SHORT_BGR = [
    [245, 150, 100],  # car
    [180, 30, 80],  # truck
    [150, 60, 30],  # bike
    [30, 30, 255],  # person
    [255, 0, 255],  # road
    [75, 0, 75],  # sidewalk
    [0, 200, 255],  # building
    [0, 175, 0],  # nature
    [150, 240, 255], # pole
    [0, 60, 135],   # tree trunk
    [0, 0, 255],    # traffi-sign
    [255, 255, 50],  # other-objects
    [0, 0, 0],  # ignore
]
WAYMO_COLOR_PALETTE_SHORT = [(c[2], c[1], c[0]) for c in WAYMO_COLOR_PALETTE_SHORT_BGR]


def draw_points_image_labels(img, img_indices, seg_labels, show=True, color_palette_type='NuScenes', point_size=0.5, save=None):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Waymo':
        color_palette = WAYMO_COLOR_PALETTE_SHORT
    elif color_palette_type == 'NuScenesLidarSeg':
        color_palette = NUSCENES_LIDARSEG_COLOR_PALETTE
    elif color_palette_type == 'Synthia':
        color_palette = SYNTHIA_KITTI_COLOR_PALETTE_SHORT
    else:
        raise NotImplementedError('Color palette type not supported: {}'.format(color_palette_type))
    color_palette = np.array(color_palette) / 255.
    seg_labels[seg_labels == -100] = len(color_palette) - 1
    colors = color_palette[seg_labels]

    plt.imshow(img)
    plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    plt.axis('off')

    if show:
        plt.show()

    if save is not None:
        plt.savefig(save)
    
    plt.close()

def draw_range_image_labels(proj_labels, show=True, save=False, color_palette_type='NuScenes'):
    if color_palette_type == 'NuScenes':
        color_palette = NUSCENES_COLOR_PALETTE_SHORT
    elif color_palette_type == 'A2D2':
        color_palette = A2D2_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    elif color_palette_type == 'SemanticKITTI_long':
        color_palette = SEMANTIC_KITTI_COLOR_PALETTE
    elif color_palette_type == 'Waymo':
        color_palette = WAYMO_COLOR_PALETTE_SHORT
    else:
        raise NotImplementedError('Color palette type not supported')
    color_palette = np.array(color_palette) / 255.
    proj_labels[proj_labels < 0] = len(color_palette) - 1
    proj_seg_colors = color_palette[proj_labels]
    plt.imshow(proj_seg_colors)
    
    if show:
        plt.show()
    if save:
        import cv2
        save_img = (proj_seg_colors * 255.).astype(np.uint8)
        save_img = np.stack((save_img[:,:,2], 
                             save_img[:,:,1],
                             save_img[:,:,0]), axis=2)
        print(save_img.shape)
        cv2.imwrite('latte/samples/fig.jpg', save_img)

def draw_registered_point(pc_ls, color_ls, vis=True, save=False):
    """
    Function to visualize register points
    Args:
        pc_ls: list of pc (np.array), which have been transformed in the same pose
        color_ls: list of color (ls), which has the same length of pc_ls
    Return:
        None
    """
    t_sample_ls = []
    for i in range(0, len(pc_ls)):
        source_temp = copy.deepcopy(pc_ls[i-1])
        # transform np.ndarray to PointCloud
        curr_pc = o3d.geometry.PointCloud()
        curr_pc.points = o3d.utility.Vector3dVector(source_temp[:, :3])
        # curr_pc = curr_pc.random_down_sample(0.04)
        curr_pc.estimate_normals()
        t_sample_ls.append(curr_pc)

    for i in range(len(t_sample_ls)):
        t_sample_ls[i].paint_uniform_color(color_ls[i])
    if save:
        for i in range(len(t_sample_ls)):
            o3d.io.write_point_cloud("latte/samples/temp/{:05d}.pcd".format(i), t_sample_ls[i])
    if vis:
        o3d.visualization.draw_geometries(t_sample_ls,
                                        zoom=0.4459,
                                        front=[0.9288, -0.2951, -0.2242],
                                        lookat=[1.6784, 2.0612, 1.4451],
                                        up=[-0.3402, -0.9189, -0.1996])


def normalize_depth(depth, d_min, d_max):
    # normalize linearly between d_min and d_max
    data = np.clip(depth, d_min, d_max)
    return (data - d_min) / (d_max - d_min)

def depth_color(val, min_d=0, max_d=120):
    """ 
    print Color(HSV's H value) corresponding to distance(m) 
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val) # max distance is 120m but usually not usual
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8) 

def grep_depth_color(val, min_d=0, max_d=50):
    """ 
    print Color(HSV's H value) corresponding to distance(m) 
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val,) # max distance is 120m but usually not usual

    return (((max_d - val) / (max_d - min_d)) * (255))

def draw_range_image_depth(depth, save=False):
    # depth = normalize_depth(depth, d_min=3., d_max=50.)
    # depth = normalize_depth(depth, d_min=depth.min(), d_max=depth.max())
    # colors = []
    # for depth_val in depth.tolist():
    #     colors.append(interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val))
    grey_colors = grep_depth_color(depth)
    # ax5.imshow(np.full_like(img, 255))
    # plt.imshow(img)
    # plt.scatter(img_indices[:, 1], img_indices[:, 0], c=colors, alpha=0.5, s=point_size)

    # plt.axis('off')

    # if show:
    #     plt.show()

    if save:
        from PIL import Image
        im = Image.fromarray(np.uint8(grey_colors), 'L')
        # save_img = grey_colors.astype(np.uint8)
        # save_img = np.stack((save_img[:,:,2], 
        #                      save_img[:,:,1],
        #                      save_img[:,:,0]), axis=2)
        print(grey_colors.shape)
        im.save('latte/samples/grep_fig.jpg')

def draw_bird_eye_view(coords, full_scale=4096):
    plt.scatter(coords[:, 0], coords[:, 1], s=0.1)
    plt.xlim([0, full_scale])
    plt.ylim([0, full_scale])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
def colored_point_viz(
    pc: np.ndarray,
    save_pth: str,
    pc_color: np.ndarray = None,
) -> int:
    """Function for debug visualization
    Args:
        pc (np.ndarray): point cloud to be visualized (N, 3)
        pc_color (np.ndarray): RGB colors [0, 1] of each points (N, 3)
        save_pth (str): path to save the visualization
    Returns:
        int: simple return
    """
    device = o3d.core.Device("CPU:0")
    pc_o3d = o3d.t.geometry.PointCloud(device)
    pc_o3d.point['positions'] = o3d.core.Tensor(pc[:, :3], o3d.core.float32, device)
    if pc_color is not None:
        pc_o3d.point['colors'] = o3d.core.Tensor(pc_color, o3d.core.float32, device)
    o3d.t.io.write_point_cloud(save_pth, pc_o3d)
    
    return 0
