import os
import os.path as osp
from sys import prefix
import open3d as o3d
import pickle
from PIL import Image
import imageio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T

from latte.data.utils.refine_pseudo_labels import refine_pseudo_labels
from latte.data.utils.augmentation_3d import augment_and_scale_3d
from latte.data.synthia.preprocess import world_to_img


class SynthiaBase(Dataset):
    """SemanticKITTI dataset"""

    # https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    class_name_to_id = {
        "void":             0,
        "sky":              1,
        "building":         2, 
        "road":             3,
        "sidewalk":         4,
        "fence":            5, 
        "vegetation":       6,
        "pole":             7,
        "car":              8,
        "traffic-sign":     9,
        "pedestrian":       10,
        "bicycle":          11,
        "motorcycle":       12,
        "parking-slot":     13,     # 0 points
        "road-work":        14,
        "traffic-light":    15,
        "terrain":          16,     # 0 points
        "rider":            17,
        "truck":            18,     # 0 points
        "bus":              19,
        "train":            20,     # 0 points
        "wall":             21, 
        "lanemarking":      22, 
    }

    id_to_class_name = {v: k for k, v in class_name_to_id.items()}

    # use those categories if merge_classes == True (common with Semantic-KITTI)
    categories = {
        'car': ['car', 'bus', 'truck'],
        'bike': ['bicycle', 'motorcycle', 'rider'],  # riders are labeled as bikes in Audi dataset
        'person': ['pedestrian'],
        'road': ['road', 'lanemarking', 'parking-slot'],
        'sidewalk': ['sidewalk'],
        'building': ['building', 'wall'],
        'nature': ['vegetation', 'terrain'],
        'pole': ['pole'],
        # 'traffic-sign': ['traffic-sign'],
        'other-objects': ['fence', 'traffic-sign', 'traffic-light'],
    }


    def __init__(self,
                 synthia_dir,
                 merge_classes=True,
                 ):

        self.split_prefix_pth = synthia_dir
        self.rgb_file_list = sorted(os.listdir(osp.join(self.split_prefix_pth, "RGB")))
        # rotation mtx for self-generated pc
        rot_y = np.array([[0,0,-1], [0,1,0], [1,0,0]])
        rot_x = np.array([[1,0,0], [0,0,-1], [0,1,0]])
        self.rot_mtx = np.matmul(rot_y, rot_x)
        # read calib file
        calib_path = 'latte/data/synthia/calib.txt'
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])
        self.intrin_mtx = calib_all['Ps'].reshape(3,4)

        if merge_classes:
            highest_id = list(self.id_to_class_name.keys())[-1]
            self.label_mapping = -100 * np.ones(highest_id + 2, dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_name_to_id[class_name]] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None
            self.class_names = list(self.class_name_to_id.keys())

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SynthiaSCN(SynthiaBase):
    def __init__(self,
                 synthia_dir,
                 merge_classes=True,
                 scale=20,
                 full_scale=4096,
                 image_normalizer=None,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_y=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 resize=(),
                 bottom_crop=tuple(),  # 2D augmentation (also effects 3D)
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 output_orig=False,
                 backbone="SCN"
                 ):
        super().__init__(synthia_dir,
                         merge_classes=merge_classes)

        self.output_orig = output_orig

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_y = flip_y
        self.rot_z = rot_z
        self.transl = transl

        # image parameters
        self.image_normalizer = image_normalizer
        # 2D augmentation
        self.bottom_crop = bottom_crop
        self.resize = resize
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None

        # SPVCNN or not
        self.backbone = backbone

        print("Initialize Synthia dataloader of length {}".format(len(self.rgb_file_list)))

    def __len__(self):
        return len(self.rgb_file_list)

    def pc_bottome_crop(
        self, 
        cp_wd: list,
        points_img: np.ndarray,
        keep_idx: np.ndarray
    ) -> np.ndarray:
        """
        Point cloud bottom crop to map image.
        Args:
            cp_wd:
                Cropping windows: [left, right, top, bottom]
            points_img:
                Point-wise pixel locations computed before.
            keep_idx:
                Point-wise index indicating preserving or discarding.
        Return:
            cp_points_img:
                Cropped version of input points_img
            keep_idx:
                Updated version of input keep_idx.
        """
        # self.bottom_crop is a tuple (crop_width, crop_height)
        left, right, top, bottom = cp_wd
        # update image points
        keep_idx = points_img[:, 0] >= top
        keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
        keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
        keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)
        # crop image
        cp_points_img = points_img[keep_idx]
        cp_points_img[:, 0] -= top
        cp_points_img[:, 1] -= left
        
        return cp_points_img, keep_idx
    
    def __getitem__(self, index):
        # defined selected sample name and path
        rgb_file= self.rgb_file_list[index]
        lidar_file = rgb_file.replace('.png', '.bin')
        label_file = rgb_file.replace('.png', '.npy')

        rgb_pth = osp.join(self.split_prefix_pth, 'RGB', rgb_file)
        lidar_pth = osp.join(self.split_prefix_pth, 'bin', lidar_file)
        label_pth = osp.join(self.split_prefix_pth, 'Label', label_file)

        # load rgb, point cloud, and semantic label
        image = Image.open(rgb_pth)
        points = np.fromfile(lidar_pth, dtype=np.float32).reshape(-1, 3)
        # points = points * 100
        seg_label = np.load(label_pth).astype(np.int64)

        # create 2D point image from 3D point cloud
        points, points_img, keep_idx = world_to_img(points, self.intrin_mtx, image.size, return_idx=True)

        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]
        seg_label = seg_label[keep_idx]

        out_dict = {}
        keep_idx = np.ones(len(points), dtype=bool)

        # 2D Augmentation: Resize
        if self.resize:
            # Mode 1: single resize for TENT
            # check if we do not enlarge downsized images
            assert not image.size == self.resize
            assert image.size[0] > self.resize[0]
            # scale image points
            points_img[:, 0] = float(self.resize[1]) / image.size[1] * np.floor(points_img[:, 0])
            points_img[:, 1] = float(self.resize[0]) / image.size[0] * np.floor(points_img[:, 1])
            # resize image
            image = image.resize(self.resize, Image.BILINEAR)
        # append original image & points_img at the end of list
        # image_ls.append(image)
        # points_img_ls.append(points_img)

        # 2D Augmentation: Bottome crop
        if self.bottom_crop:
            # self.bottom_crop is a tuple (crop_width, crop_height)
            left = int(np.random.rand() * (image.size[0] + 1 - self.bottom_crop[0]))
            right = left + self.bottom_crop[0]
            top = image.size[1] - self.bottom_crop[1]
            bottom = image.size[1]
            # crop image
            image = image.crop((left, top, right, bottom))
            # update point cloud
            points_img, keep_idx = self.pc_bottome_crop(
                [left, right, top, bottom],
                points_img,
                keep_idx
            )
        
        img_indices = points_img.astype(np.int64)
        
        # 2D Augmentation: Random Flip
        flip_idx = np.random.rand()
        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        # 2D augmentation
        if flip_idx < self.fliplr:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]
        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std
        image = np.moveaxis(image, -1, 0)

        # Mode 1: if rot_z_list, then perform multi z rotation based on the element value of rot_z_list
        # Mode 2: if not rot_z_list but rot_z, then perform random z rotation [0, rot_z]
        # Mode 3: if not rot_z_list and not rot_z, then do not perform z rotation
        # 3D data augmentation and scaling from points to voxel indices
        # Kitti lidar coordinates: x (front), y (left), z (up)
        points = np.matmul(points, self.rot_mtx)
        coords, points = augment_and_scale_3d(points, self.scale, self.full_scale,
                                                    noisy_rot=self.noisy_rot, flip_y=self.flip_y,
                                                    rot_z=self.rot_z, transl=self.transl)
        feats_ls = []
        if self.backbone.upper() == "SPVCNN" or self.backbone.upper() == "SPVCNN_BASE":
            # cast to integer
            coords = coords.astype(np.int64)

            # only use voxels inside receptive field
            idxs = np.ones(coords.shape[0], dtype=np.bool8)
            assert coords[idxs].shape[0] == coords.shape[0]

            coords = coords[idxs]
            if self.backbone.upper() == "SCN":
                feats_ls.append(np.ones([coords[idxs].shape[0], 1], np.float32))  # simply use 1 as feature
            elif self.backbone.upper() == "SPVCNN" or self.backbone.upper() == "SPVCNN_BASE":
                import torch
                from torchsparse import SparseTensor
                from torchsparse.utils.quantize import sparse_quantize
                pc_ = coords[idxs]
                # print(pc_)
                _, inds, inverse = sparse_quantize(pc_,
                                                   1 / self.scale,
                                                   return_index=True,
                                                   return_inverse=True)
                coords = pc_
                feats = np.zeros((points[inds].shape[0], 1))
                lidar = SparseTensor(
                    feats=torch.cat((
                            torch.from_numpy(points[inds]).float(),
                            torch.from_numpy(feats).view(-1,1).float()
                        ), dim=1),
                    coords=torch.from_numpy(pc_[inds]).int())

        out_dict['img'] = image                      # list of multi-resolution images
        out_dict['img_indices'] = img_indices            # list of corresponding img -> point indices
        out_dict['coords'] = coords                      # list of multi-z-rotation points
        out_dict['feats'] = feats                       # list of multi-z-rotation feats
        out_dict['points'] = points
        out_dict['seg_label'] = seg_label[idxs]             # single seg label
        out_dict['pc2img_idx'] = keep_idx
        if self.backbone.upper() == "SPVCNN" or self.backbone.upper() == "SPVCNN_BASE":
            out_dict['lidar'] = lidar
            out_dict['indices'] = inds
            out_dict['inverse_map'] = inverse
        # out_dict['img_indices'] = out_dict['img_indices'][idxs]

        # if self.pselab_data is not None:
        #     out_dict.update({
        #         'pseudo_label_2d': self.pselab_data[index]['pseudo_label_2d'][keep_idx][idxs],
        #         'pseudo_label_3d': self.pselab_data[index]['pseudo_label_3d'][keep_idx][idxs]
        #     })

        if self.output_orig:
            out_dict.update({
                'orig_seg_label': seg_label,
                'orig_points_idx': idxs,
            })

        return out_dict


def test_SynthiaSCN():
    from latte.common.utils.checkpoint import CheckpointerV2
    from latte.models.xmuda_arch import Net3DSeg
    from latte.data.utils.visualize import \
        draw_points_image_labels, draw_bird_eye_view, colored_point_viz, SEMANTIC_KITTI_COLOR_PALETTE_SHORT
    import torch
    synthia_dir = "latte/datasets/synthia/RAND_CITYSCAPES"
    dataset = SynthiaSCN(
        synthia_dir,
        backbone="SPVCNN",
        resize=(640, 380),
        bottom_crop=(350, 350),
        fliplr=0.5)
    color_palette = np.asarray(SEMANTIC_KITTI_COLOR_PALETTE_SHORT)

    for i in range(0, len(dataset), 100):
        data = dataset[i]
        coords = data['coords']
        pc = data['points']
        pc2img_idx = data['pc2img_idx']
        seg_label = data['seg_label']
        img_indices = data['img_indices']
        img = np.moveaxis(data['img'], 0, -1)
        seg_label[seg_label < 0] = np.max(seg_label) + 1
        
        draw_points_image_labels(img, img_indices, seg_label[pc2img_idx], show=False, color_palette_type="SemanticKITTI", save='latte/viz_samples/synthia_test.png',)
        
        color_palette = np.array(color_palette) / 255.
        seg_label[seg_label == -100] = len(color_palette) - 1
        colors = color_palette[seg_label]
        colored_point_viz(pc, "latte/viz_samples/synthia_pc.pcd", colors)
        input("Press Enter to continue....")


def compute_class_weights():
    from tqdm import tqdm
    
    synthia_dir = "latte/datasets/synthia/RAND_CITYSCAPES"
    dataset = SynthiaSCN(synthia_dir, backbone="SPVCNN", merge_classes=True)
    # compute points per class over whole dataset
    num_classes = len(dataset.class_names)
    points_per_class = np.zeros(num_classes, int)
    for i, data in tqdm(enumerate(dataset)):
        # print('{}/{}'.format(i, len(dataset)))
        labels = data['seg_label']
        points_per_class += np.bincount(labels[labels != -100], minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    class_weights = class_weights / class_weights.min()
    class_weights_str = ""
    for i in range(class_weights.shape[0]):
        class_weights_str += "{}, ".format(class_weights[i])
    print('log smoothed class weights: [{}]'.format(class_weights_str))


if __name__ == '__main__':
    # test_SynthiaSCN()
    compute_class_weights()