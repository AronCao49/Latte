import os.path as osp
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

from latte.data.utils.refine_pseudo_labels import refine_pseudo_labels
from latte.data.utils.augmentation_3d import augment_and_scale_3d


class NuScenesBase(Dataset):
    """NuScenes dataset"""

    class_names = [
        # 'ignore',     # ignore during result printing
        'barrier',
        'bicycle',
        'bus',
        'car',
        'construction_vehicle',
        'motorcycle',
        'pedestrian',
        'traffic_cone',
        'trailer',
        'truck',
        'driveable_surface',
        'other_flat',
        'sidewalk',
        'terrain',
        'manmade',
        'vegetation'
     ]

    # use those categories if merge_classes == True
    #! Disabled by default
    categories = {
        "vehicle": ["bicycle", "bus", "car", "construction_vehicle", "motorcycle", "trailer", "truck"],
        "driveable_surface": ["driveable_surface"],
        "sidewalk": ["sidewalk"],
        "terrain": ["terrain"],
        "manmade": ["manmade"],
        "vegetation": ["vegetation"],
        # "ignore": ["ignore", "barrier", "pedestrian", "traffic_cone", "other_flat"],
    }
    
    # Raw label map (fine to coarse)
    raw_label_map = np.array([ 0,  0,  7,  7,  7,  0,  7,  0,  0,  1,  0,  0,  8,  0,  2,  3,  3,
        4,  5,  0,  0,  6,  9, 10, 11, 12, 13, 14, 15,  0, 16,  0])

    def __init__(self,
                 split,
                 nuscenes_dir,
                 preprocess_dir,
                 gt_poses,
                 merge_classes=False,
                 pselab_paths=None
                 ):

        self.nuscenes_dir = nuscenes_dir
        self.split = split
        self.preprocess_dir = preprocess_dir
        self.gt_poses = gt_poses

        print("Initialize Nuscenes dataloader")

        assert isinstance(split, tuple)
        print('Load', split)
        self.data = []
        for curr_split in split:
            with open(osp.join(self.preprocess_dir, curr_split + '.pkl'), 'rb') as f:
                self.data.extend(pickle.load(f))
        
        # Load kiss-icp poses
        for i, sample in enumerate(self.data):
            pose_file_suffix = "gt_kitti" if self.gt_poses else "poses_kitti"
            scene_name, scan_ID = sample['scene_name'], sample['scan_ID']
            pose_file = osp.join(
                self.nuscenes_dir, "poses", scene_name, 
                "{}_{}.txt".format(scene_name.split("-")[-1], pose_file_suffix)
                )
            with open(pose_file, 'r') as f:
                pose = f.readlines()[int(scan_ID)].strip('\n').split(' ')
                pose_array = np.identity(4)
                pose_array[:3, :4] = np.asarray(pose).reshape(3,4)
            self.data[i]['pose'] = pose_array

        #TODO: change to individual load scan-wise pseudo label
        self.pselab_data = None
        if pselab_paths:
            assert isinstance(pselab_paths, tuple)
            print('Load pseudo label data ', pselab_paths)
            self.pselab_data = []
            for curr_split in pselab_paths:
                self.pselab_data.extend(np.load(curr_split, allow_pickle=True))

            # check consistency of data and pseudo labels
            assert len(self.pselab_data) == len(self.data)
            for i in range(len(self.pselab_data)):
                assert len(self.pselab_data[i]['pseudo_label_2d']) == len(self.data[i]['seg_labels'])

            # refine 2d pseudo labels
            probs2d = np.concatenate([data['probs_2d'] for data in self.pselab_data])
            pseudo_label_2d = np.concatenate([data['pseudo_label_2d'] for data in self.pselab_data]).astype(np.int)
            pseudo_label_2d = refine_pseudo_labels(probs2d, pseudo_label_2d)

            # refine 3d pseudo labels
            # fusion model has only one final prediction saved in probs_2d
            if self.pselab_data[0]['probs_3d'] is not None:
                probs3d = np.concatenate([data['probs_3d'] for data in self.pselab_data])
                pseudo_label_3d = np.concatenate([data['pseudo_label_3d'] for data in self.pselab_data]).astype(np.int)
                pseudo_label_3d = refine_pseudo_labels(probs3d, pseudo_label_3d)
            else:
                pseudo_label_3d = None

            # undo concat
            left_idx = 0
            for data_idx in range(len(self.pselab_data)):
                right_idx = left_idx + len(self.pselab_data[data_idx]['probs_2d'])
                self.pselab_data[data_idx]['pseudo_label_2d'] = pseudo_label_2d[left_idx:right_idx]
                if pseudo_label_3d is not None:
                    self.pselab_data[data_idx]['pseudo_label_3d'] = pseudo_label_3d[left_idx:right_idx]
                else:
                    self.pselab_data[data_idx]['pseudo_label_3d'] = None
                left_idx = right_idx

        if merge_classes:
            self.label_mapping = -100 * np.ones(len(self.class_names), dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_names.index(class_name)] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class NuScenesSCN(NuScenesBase):
    def __init__(self,
                 split,
                 preprocess_dir,
                 nuscenes_dir='',
                 gt_poses=False,    # use GT poses or not
                 pselab_paths=None,
                 merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 front_only=False,
                 resize=(400, 225),
                 bottom_crop=tuple(),
                 image_normalizer=None,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_x=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 output_orig=False,
                 use_feats=True,
                 backbone="SPVCNN",
                 ):
        super().__init__(split,
                         nuscenes_dir,
                         preprocess_dir,
                         gt_poses,
                         merge_classes=merge_classes,
                         pselab_paths=pselab_paths)

        self.output_orig = output_orig
        self.use_feats = use_feats
        self.backbone = backbone

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        self.front_only = front_only
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_x = flip_x
        self.rot_z = rot_z
        self.transl = transl

        # image parameters
        self.resize = resize
        self.bottom_crop = bottom_crop
        self.image_normalizer = image_normalizer

        # data augmentation
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None

    # def load_nusc_pose(self, data_dict):
    #     scene_name = 
    
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
        data_dict = self.data[index]

        # Data laoding
        points = np.fromfile(
            osp.join(self.nuscenes_dir, data_dict['lidar_path']), 
            dtype=np.float32,
            count=-1).reshape(-1, 5)[:, :4]
        feats = points[:, 3].reshape(-1, 1)
        points = points[:, :3]
        seg_label = np.fromfile(data_dict['seg_labels_pth'], dtype=np.uint8)
        
        # Raw label mapping
        seg_label = self.raw_label_map[seg_label]
        seg_label -= 1
        seg_label[seg_label < 0] = -100
        if self.label_mapping is not None:
            seg_label[seg_label >= 0] = self.label_mapping[seg_label[seg_label >= 0]]

        out_dict = {}

        # Load 2D/3D raw correpondence
        points_img = data_dict['points_img'].copy()
        pc2img_idx = data_dict['pc2img_idx'].copy()
        img_path = osp.join(self.nuscenes_dir, data_dict['camera_path'])
        image = Image.open(img_path)
        ori_img_size = image.size

        if self.resize:
            if not image.size == self.resize:
                # check if we do not enlarge downsized images
                assert image.size[0] > self.resize[0]

                # scale image points
                points_img[:, 0] = float(self.resize[1]) / image.size[1] * np.floor(points_img[:, 0])
                points_img[:, 1] = float(self.resize[0]) / image.size[0] * np.floor(points_img[:, 1])

                # resize image
                image = image.resize(self.resize, Image.BILINEAR)

        # 2D/3D bottom crop
        if self.bottom_crop:
            keep_idx = np.ones(len(points_img), dtype=np.bool_)
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
            pc2img_idx[pc2img_idx] = keep_idx
        img_indices = points_img.astype(np.int64)
        
        
        img_indices = points_img.astype(np.int64)

        assert np.all(img_indices[:, 0] >= 0)
        assert np.all(img_indices[:, 1] >= 0)
        assert np.all(img_indices[:, 0] < image.size[1])
        assert np.all(img_indices[:, 1] < image.size[0])

        # 2D augmentation
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.
        # 2D augmentation
        if np.random.rand() < self.fliplr:
            image = np.ascontiguousarray(np.fliplr(image))
            img_indices[:, 1] = image.shape[1] - 1 - img_indices[:, 1]

        # normalize image
        if self.image_normalizer:
            mean, std = self.image_normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std

        out_dict['img'] = np.moveaxis(image, -1, 0)
        out_dict['img_indices'] = img_indices

        # 3D data augmentation and scaling from points to voxel indices
        # nuscenes lidar coordinates: x (right), y (front), z (up)
        ori_points = points.copy()
        coords, aug_points = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_x=self.flip_x, rot_z=self.rot_z, transl=self.transl)

        # cast to integer
        coords = coords.astype(np.int64)
        if self.front_only:
            coords, points = coords[pc2img_idx], points[pc2img_idx]
            ori_points, seg_label = ori_points[pc2img_idx], seg_label[pc2img_idx]
            feats = feats[pc2img_idx]
            aug_points = aug_points[pc2img_idx]
        out_dict['pc2img_idx'] = pc2img_idx if not self.front_only else \
            np.ones(img_indices.shape[0], dtype=np.bool8)

        # only use voxels inside receptive field
        # idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)
        idxs = np.ones([points.shape[0]], dtype=np.bool8)

        out_dict['aug_points'] = aug_points[idxs]
        out_dict['ori_points'] = ori_points[idxs]
        
        # For SCN
        if self.backbone == "SCN":
            out_dict['coords'] = coords[idxs]
            out_dict['feats'] = np.ones([len(idxs), 1], np.float32) if not self.use_feats else feats[idxs]
        # For SPVCNN
        elif self.backbone == "SPVCNN":
            pc_ = coords[idxs]
            # print(pc_)
            _, inds, inverse = sparse_quantize(
                pc_,
                1 / self.scale,
                return_index=True,
                return_inverse=True
            )
            out_dict['coords'] = pc_
            out_dict['feats'] = feats[idxs]
            out_dict['lidar'] = SparseTensor(
                feats=torch.cat((
                    torch.from_numpy(points[inds]).float(), 
                    torch.from_numpy(out_dict['feats'][inds]).view(-1,1).float()
                    ), dim=1), 
                coords=torch.from_numpy(pc_[inds]).int()
            )
            out_dict['indices'] = inds
            out_dict['inverse_map'] = inverse
        
        out_dict['seg_label'] = seg_label[idxs]
        out_dict['pc2img_idx'] = out_dict['pc2img_idx'][idxs]
        out_dict['scene_name'] = data_dict['scene_name']
        out_dict['scan_ID'] = data_dict['scan_ID']
        out_dict['pose'] = data_dict['pose']
        out_dict['proj_matrix'] = data_dict['proj_matrix']
        out_dict['ori_img_size'] = ori_img_size

        if self.pselab_data is not None:
            out_dict['pseudo_label_2d'] = self.pselab_data[index]['pseudo_label_2d'][idxs]
            if self.pselab_data[index]['pseudo_label_3d'] is None:
                out_dict['pseudo_label_3d'] = None
            else:
                out_dict['pseudo_label_3d'] = self.pselab_data[index]['pseudo_label_3d'][idxs]

        if self.output_orig:
            out_dict.update({
                'orig_seg_label': seg_label,
                'orig_points_idx': idxs[out_dict['pc2img_idx']],
            })

        return out_dict


def test_NuScenesSCN():
    from latte.data.utils.visualize import \
        draw_points_image_labels, draw_bird_eye_view, colored_point_viz, NUSCENES_LIDARSEG_COLOR_PALETTE
    preprocess_dir = 'latte/datasets/nuscenes/preprocess_all/preprocess'
    nuscenes_dir = 'latte/datasets/nuscenes'
    # # split = ('train_singapore',)
    # # pselab_paths = ('/home/docker_user/workspace/outputs/xmuda/nuscenes/usa_singapore/xmuda/pselab_data/train_singapore.npy',)
    # # split = ('train_night',)
    # # pselab_paths = ('/home/docker_user/workspace/outputs/xmuda/nuscenes/day_night/xmuda/pselab_data/train_night.npy',)
    # split = ('val_night',)
    split = ('test_singapore',)
    dataset = NuScenesSCN(split=split,
                                  preprocess_dir=preprocess_dir,
                                  nuscenes_dir=nuscenes_dir,
                                  # pselab_paths=pselab_paths,
                                  merge_classes=False,
                                  noisy_rot=0.1,
                                  flip_x=0.5,
                                  rot_z=2*np.pi,
                                  transl=True,
                                  fliplr=0.5,
                                  color_jitter=(0.4, 0.4, 0.4)
                                  )
    for i in range(len(dataset)):
        data = dataset[i]
        coords = data['coords']
        aug_points = data['aug_points']
        seg_label = data['seg_label']
        pc2img_idx = data['pc2img_idx']
        img = np.moveaxis(data['img'], 0, 2)
        img_indices = data['img_indices']
        draw_points_image_labels(img, img_indices, seg_label[pc2img_idx], 
                                 color_palette_type='NuScenesLidarSeg', point_size=1, 
                                 save="latte/samples/nusc_point2img.png")
        color_platte = np.array(NUSCENES_LIDARSEG_COLOR_PALETTE) / 255.
        seg_label += 1
        seg_label[seg_label < 0] = 0
        pc_color = color_platte[seg_label]
        colored_point_viz(aug_points, 'latte/samples/nusc_labeled_points.pcd', pc_color)
        # pseudo_label_2d = data['pseudo_label_2d']
        # draw_points_image_labels(img, img_indices, pseudo_label_2d, color_palette_type='NuScenes', point_size=3)
        # draw_bird_eye_view(coords)
        print("Current scene: {}, scan ID: {}".format(data['scene_name'], data['scan_ID']))
        if i % 5 == 0:
            print('Number of points:', len(coords))
            input("Press Enter to continue...")

def compute_class_weights():
    from tqdm import tqdm
    preprocess_dir = 'latte/datasets/nuscenes/preprocess_all/preprocess'
    nuscenes_dir = 'latte/datasets/nuscenes'
    split = ('train_usa',)  # nuScenes-lidarseg USA/Singapore
    # split = ('train_day', 'test_day')  # nuScenes-lidarseg Day/Night
    # split = ('train_singapore', 'test_singapore')  # nuScenes-lidarseg Singapore/USA
    # split = ('train_night', 'test_night')  # nuScenes-lidarseg Night/Day
    # split = ('train_singapore_labeled',)  # SSDA: nuScenes-lidarseg Singapore labeled
    dataset = NuScenesSCN(split=split,
                        preprocess_dir=preprocess_dir,
                        nuscenes_dir=nuscenes_dir,
                        # pselab_paths=pselab_paths,
                        merge_classes=True,
                        noisy_rot=0.1,
                        flip_x=0.5,
                        rot_z=2*np.pi,
                        transl=True,
                        fliplr=0.5,
                        color_jitter=(0.4, 0.4, 0.4)
                        )
    # compute points per class over whole dataset
    num_classes = len(dataset.categories)
    points_per_class = np.zeros(num_classes, int)
    for i, data in tqdm(enumerate(dataset)):
        # print('{}/{}'.format(i, len(dataset)))
        labels = data['seg_label']
        points_per_class += np.bincount(labels[labels != -100], minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ')
    print([weight / class_weights.min() for weight in class_weights])


def compute_img_normal():
    from tqdm import tqdm
    preprocess_dir = 'latte/datasets/nuscenes/preprocess_all/preprocess'
    nuscenes_dir = 'latte/datasets/nuscenes'
    split = ('train_usa', 'val_usa')  # nuScenes-lidarseg USA/Singapore
    # split = ('train_day', 'test_day')  # nuScenes-lidarseg Day/Night
    # split = ('train_singapore', 'test_singapore')  # nuScenes-lidarseg Singapore/USA
    # split = ('train_night', 'test_night')  # nuScenes-lidarseg Night/Day
    # split = ('train_singapore_labeled',)  # SSDA: nuScenes-lidarseg Singapore labeled
    dataset = NuScenesSCN(split=split,
                        preprocess_dir=preprocess_dir,
                        nuscenes_dir=nuscenes_dir,
                        # pselab_paths=pselab_paths,
                        resize=(800, 450)
                        )
    # compute points per class over whole dataset
    channels_sum, channels_squared_sum = 0, 0
    num_batches = len(dataset)

    for i, data in enumerate(dataset):
        print('{}/{}'.format(i, len(dataset)))
        img = data['img']
        img = img.reshape(img.shape[0], -1)
        channels_sum += (np.mean(img, axis=1) / num_batches)
        channels_squared_sum += (np.mean(img**2, axis=1) / num_batches)
    
    mean = channels_sum
    std = (channels_squared_sum - mean**2)**0.5

    print('Image statistics: ')
    print("Mean: {},\tStd: {}".format(mean, std))
    # Mean: [0.41836196 0.4219197  0.41423622],       Std: [0.20544815 0.20367281 0.21079224]


def compute_stats():
    preprocess_dir = 'path/to/data/nuscenes_lidarseg_preprocess/preprocess'
    nuscenes_dir = 'path/to/data/nuscenes'
    outdir = 'path/to/data/nuscenes_lidarseg_preprocess/stats'
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = ('train_day', 'test_day', 'train_night', 'val_night', 'test_night',
              'train_usa', 'test_usa', 'train_singapore', 'val_singapore', 'test_singapore')
    for split in splits:
        dataset = NuScenesSCN(
            split=(split,),
            preprocess_dir=preprocess_dir,
            nuscenes_dir=nuscenes_dir
        )
        # compute points per class over whole dataset
        num_classes = len(dataset.class_names)
        points_per_class = np.zeros(num_classes, int)
        for i, data in enumerate(dataset.data):
            print('{}/{}'.format(i, len(dataset)))
            points_per_class += np.bincount(data['seg_label'], minlength=num_classes)

        plt.barh(dataset.class_names, points_per_class)
        plt.grid(axis='x')

        # add values right to the bar plot
        for i, value in enumerate(points_per_class):
            x_pos = value
            y_pos = i
            if dataset.class_names[i] == 'driveable_surface':
                x_pos -= 0.25 * points_per_class.max()
                y_pos += 0.75
            plt.text(x_pos + 0.02 * points_per_class.max(), y_pos - 0.25, f'{value:,}', color='blue', fontweight='bold')
        plt.title(split)
        plt.tight_layout()
        # plt.show()
        plt.savefig(outdir / f'{split}.png')
        plt.close()


if __name__ == '__main__':
    test_NuScenesSCN()
    # compute_img_normal()
    # compute_class_weights()
    # compute_stats()