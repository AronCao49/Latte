import glob
import os.path as osp
import pickle
from PIL import Image, ImageFile
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

from latte.data.utils.refine_pseudo_labels import refine_pseudo_labels
from latte.data.utils.augmentation_3d import augment_and_scale_3d
# use individual load dataloader
# from latte.data.semantic_kitti.semantic_kitti_range_dataloader import SemanticKITTIBase
from latte.data.semantic_kitti import splits

ImageFile.LOAD_TRUNCATED_IMAGES = True



class SemanticKITTIBase(Dataset):
    """
    SemanticKITTI base dataset that store loading paths for lidar and images
    
    """

    # https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    id_to_class_name = {
        0: "unlabeled",
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }

    class_name_to_id = {v: k for k, v in id_to_class_name.items()}

    # use those categories if merge_classes == True (common with A2D2)
    categories_s = {
        'car': ['car', 'moving-car'],
        'truck': ['truck', 'moving-truck'],
        'bike': ['bicycle', 'motorcycle', 'bicyclist', 'motorcyclist',
                 'moving-bicyclist', 'moving-motorcyclist'],  # riders are labeled as bikes in Audi dataset
        'person': ['person', 'moving-person'],
        'road': ['road', 'lane-marking'],
        'parking': ['parking'],
        'sidewalk': ['sidewalk'],
        'building': ['building'],
        'nature': ['vegetation', 'trunk', 'terrain'],
        'other-objects': ['fence', 'traffic-sign', 'other-object', 'pole'],
    }
    
    categories_synthia = {
        'car': ['car', 'moving-car', 'truck', 'moving-truck'],
        'bike': ['bicycle', 'motorcycle', 'bicyclist', 'motorcyclist',
                 'moving-bicyclist', 'moving-motorcyclist'],  # riders are labeled as bikes in Audi dataset
        'person': ['person', 'moving-person'],
        'road': ['road', 'lane-marking', 'parking'],
        'sidewalk': ['sidewalk'],
        'building': ['building'],
        'nature': ['vegetation', 'trunk', 'terrain'],
        'pole': ['pole'],
        'other-objects': ['fence', 'traffic-sign', 'other-object'],
    }

    categories_l = {
        'car': ['car', 'moving-car'],
        'truck': ['truck', 'moving-truck'],
        'bike': ['bicycle', 'motorcycle', 'bicyclist', 'motorcyclist',
                 'moving-bicyclist', 'moving-motorcyclist'],  # riders are labeled as bikes in Audi dataset
        'person': ['person', 'moving-person'],
        'road': ['road', 'lane-marking'],
        'parking': ['parking'],
        'sidewalk': ['sidewalk'],
        'building': ['building'],
        'nature': ['vegetation', 'trunk', 'terrain'],
        'pole': ['pole'],
        'other-objects': ['fence', 'traffic-sign', 'other-object'],
    }

    def __init__(self,
                 split,
                 root_dir,
                 merge_classes=False,
                 ps_label_dir=None,
                 cat_type="s",
                 use_pc_mm=False,
                 obj_name_ls=[],
                 obj_root_dir=None,
                 ):

        self.split = split
        # point mix-match
        self.obj_pc_dict = {}
        self.use_pc_mm = use_pc_mm
        self.obj_name_ls = obj_name_ls
        self.obj_root_dir = obj_root_dir
        scenes = []
        print("Initialize SemanticKITTI dataloader")
        assert isinstance(split, tuple)
        print('Load', split)

        # specify mapping table
        self.cat_type = cat_type
        if cat_type == "s":
            self.categories = self.categories_s
        elif cat_type == "l":
            self.categories = self.categories_l
        elif cat_type == "synthia":
            self.categories = self.categories_synthia
        else:
            raise IndexError("The desired category type {} is not supported!".format(cat_type))

        # retrieve scenes of specified split
        for single_split in self.split:
            scenes.extend(getattr(splits, single_split))
        self.root_dir = root_dir
        self.data = []

        # pseudo label dir
        self.ps_label_dir = ps_label_dir
        self.pselab_data = None
        
        # retrieve loading paths
        self.glob_frames(scenes)

        if merge_classes:
            highest_id = list(self.id_to_class_name.keys())[-1]
            self.label_mapping = -100 * np.ones(highest_id + 2, dtype=int)
            for cat_idx, cat_list in enumerate(self.categories.values()):
                for class_name in cat_list:
                    self.label_mapping[self.class_name_to_id[class_name]] = cat_idx
            self.class_names = list(self.categories.keys())
        else:
            self.label_mapping = None

    def glob_frames(self, scenes):
        for scene in scenes:
            glob_path = osp.join(self.root_dir, 'dataset', 'sequences', scene, 'image_2', '*.png')
            cam_paths = sorted(glob.glob(glob_path))
            # load calibration
            calib = self.read_calib(osp.join(self.root_dir, 'dataset', 'sequences', scene, 'calib.txt'))
            proj_matrix = calib['P2'] @ calib['Tr']
            proj_matrix = proj_matrix.astype(np.float32)
            # load gt poses
            gt_pose_f = open(osp.join(self.root_dir, 'dataset', 'sequences', scene, 'poses.txt'), 'r')
            gt_poses = gt_pose_f.readlines()
            gt_pose_f.close()
            # load estimated poses
            et_pose_f = open(osp.join(self.root_dir, 'poses', scene, '{}_poses_kitti.txt'.format(scene)), 'r')
            et_poses = et_pose_f.readlines()
            et_pose_f.close()
            # pre-defined pseudo label pth if required
            ps_label_prefix = osp.join(self.root_dir, 'ps_label', self.ps_label_dir, scene) \
                if self.ps_label_dir is not None else None

            for cam_path in cam_paths:
                basename = osp.basename(cam_path)
                frame_id = osp.splitext(basename)[0]
                assert frame_id.isdigit()

                # index pose from gt poses
                gt_pose = gt_poses[int(frame_id)].strip('\n').split(' ')
                gt_pose_array = np.identity(4)
                gt_pose_array[:3, :4] = np.asarray(gt_pose).reshape(3,4)
                # transform pose from camera to lidar
                gt_pose_array = np.linalg.inv(calib['Tr']) @ gt_pose_array @ calib['Tr']
                
                # index pose from estimated poses
                pose = et_poses[int(frame_id)].strip('\n').split(' ')
                pose_array = np.identity(4)
                pose_array[:3, :4] = np.asarray(pose).reshape(3,4)
                pose_array = np.linalg.inv(calib['Tr']) @ pose_array @ calib['Tr']

                data = {
                    'camera_path': cam_path,
                    'lidar_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'velodyne',
                                           frame_id + '.bin'),
                    'label_path': osp.join(self.root_dir, 'dataset', 'sequences', scene, 'labels',
                                           frame_id + '.label'),
                    'proj_matrix': proj_matrix,
                    'gt_pose': gt_pose_array,
                    'pose': pose_array, 
                    'scene': scene,
                    'frame_id': int(frame_id)
                }
                
                # update path to load pseudo label if required
                if ps_label_prefix is not None:
                    pslabel_path = osp.join(ps_label_prefix, frame_id + '.npy')
                    data['pslabel_path'] = pslabel_path

                for k, v in data.items():
                    if isinstance(v, str) and k != 'scene':
                        if not osp.exists(v):
                            raise IOError('File not found {}'.format(v))
                self.data.append(data)

        # load object pc
        if self.use_pc_mm:
            for obj_class in self.obj_name_ls:
                glob_path = osp.join(self.obj_root_dir, obj_class, "*.bin")
                obj_paths = sorted(glob.glob(glob_path))
                self.obj_pc_dict[obj_class] = obj_paths
    
    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)
        return calib_out

    @staticmethod
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

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class SemanticKITTISCN(SemanticKITTIBase):
    def __init__(self,
                 split,
                 root_dir,
                 ps_label_dir=None,
                 merge_classes=False,
                 scale=20,
                 full_scale=4096,
                 image_normalizer=None,
                 noisy_rot=0.0,  # 3D augmentation
                 flip_y=0.0,  # 3D augmentation
                 rot_z=0.0,  # 3D augmentation
                 transl=False,  # 3D augmentation
                 bottom_crop=tuple(),  # 2D augmentation (also effects 3D)
                 fliplr=0.0,  # 2D augmentation
                 color_jitter=None,  # 2D augmentation
                 output_orig=False,
                 backbone="SCN",
                 cat_type='s',
                 front_only=False
                 ):
        super().__init__(split,
                         root_dir,
                         merge_classes=merge_classes,
                         ps_label_dir=ps_label_dir,
                         cat_type=cat_type)

        self.output_orig = output_orig

        # point cloud parameters
        self.scale = scale
        self.full_scale = full_scale
        self.front_only = front_only
        # 3D augmentation
        self.noisy_rot = noisy_rot
        self.flip_y = flip_y
        self.rot_z = rot_z
        self.transl = transl
        self.backbone = backbone

        # image parameters
        self.image_normalizer = image_normalizer
        # 2D augmentation
        self.bottom_crop = bottom_crop
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None

    def data_extraction(self, index):
        """
        Function to extract image and lidar from data_list based on index num
        Args:
            index: index number indicating the sample to be extracted
        Return;
            data_dict: dict containing related variables (e.g., point, image, label).
        """
        data_dict = self.data[index].copy()
        scan = np.fromfile(data_dict['lidar_path'], dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, :3]
        feats = scan[:, 3]
        label = np.fromfile(data_dict['label_path'], dtype=np.uint32)
        label = label.reshape((-1))
        label = label & 0xFFFF  # get lower half for semantics
        
        # record seq name and scan ID
        seq_name = data_dict['lidar_path'].split('/')[-3]
        seq_ID = int(data_dict['lidar_path'].split('/')[-1].split('.')[0])

        z_idx = np.logical_and(points[:, 2] > -3, points[:, 2] < 10)
        points = points[z_idx]
        feats = feats[z_idx]
        label = label[z_idx]

        # load pseudo label if needed
        if 'pslabel_path' in data_dict.keys():
            ps_data = np.load(data_dict['pslabel_path'], allow_pickle=True).tolist()

        # load image
        image = Image.open(data_dict['camera_path'])
        
        data_dict['image'] = image
        data_dict['feats'] = feats
        data_dict['points'] = points
        data_dict['seg_labels'] = label.astype(np.int16)
        # scan info
        data_dict['seq_name'] = seq_name
        data_dict['seq_ID'] = seq_ID
        # pseudo label
        if 'pslabel_path' in data_dict.keys():
            data_dict['pseudo_label_2d'] = ps_data['pseudo_label_2d']
            data_dict['pseudo_label_3d'] = ps_data['pseudo_label_3d']
            data_dict['probs_2d'] = ps_data['probs_2d']
            data_dict['probs_3d'] = ps_data['probs_3d']
            # to avoid non-determinstic
            data_dict['ori_keep_idx'] = ps_data['ori_keep_idx']
            data_dict['ori_img_points'] = ps_data['ori_img_points']

        return data_dict

    def preprocess(self, data_dict):
        points = data_dict['points']
        image_size = data_dict['image'].size

        # preserve the half of pc
        keep_idx = points[:, 0] > 0

        # ps_label refinement
        if 'pseudo_label_3d' in data_dict.keys():
            ps_label_2d = refine_pseudo_labels(data_dict['probs_2d'],
                                               data_dict['pseudo_label_2d'].astype(np.int32))
            ps_label_3d = refine_pseudo_labels(data_dict['probs_3d'],
                                               data_dict['pseudo_label_3d'].astype(np.int32))
            data_dict.update({
                'pseudo_label_2d': ps_label_2d,
                'pseudo_label_3d': ps_label_3d,
            })
            # directly load keep_idx and img_points from ps_data
            # to avoid non-deterministic
            keep_idx = data_dict['ori_keep_idx']
            img_points = data_dict['ori_img_points']
            
            data_dict.update({
                'points': points,
                'feats': data_dict['feats'].reshape(-1,1),
                'seg_labels': data_dict['seg_labels'],
                'points_img': img_points
            })
        else:
            points_hcoords = np.concatenate([points[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
            img_points = np.matmul(data_dict['proj_matrix'].astype(np.float32), points_hcoords.T, dtype=np.float32).T
            img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
            img_points = np.around(img_points, decimals=2)
            keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image_size)
            keep_idx[keep_idx] = keep_idx_img_pts
            # fliplr so that indexing is row, col and not col, row
            img_points = np.fliplr(img_points)
            img_points = img_points[keep_idx_img_pts]
            data_dict.update({
                'ori_keep_idx': keep_idx,
                'ori_img_points': img_points
            })
            data_dict.update({
                'points': points,
                'feats': data_dict['feats'].reshape(-1,1),
                'seg_labels': data_dict['seg_labels'],
                'points_img': img_points
            })

        return data_dict
    
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
        data_dict = self.data_extraction(index)

        # preprocess
        data_dict = self.preprocess(data_dict)
        seg_label = data_dict['seg_labels']

        # mapping label
        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]
        out_dict = {}
        
        # retrieve point and feats
        points = data_dict['points']
        ori_points = points.copy()
        feats = data_dict['feats']
        pc2img_idx = data_dict['ori_keep_idx']
        points_img = data_dict['points_img'].copy()
            
        if 'pseudo_label_2d' in data_dict.keys():
            ps_label_2d = data_dict['pseudo_label_2d'].copy()
            ps_label_3d = data_dict['pseudo_label_3d'].copy()
        
        keep_idx = np.ones(len(points_img), dtype=np.bool_)
        image = data_dict['image'].copy()
        ori_img_size = image.size
        
        # 2D/3D bottom crop
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
            pc2img_idx[pc2img_idx] = keep_idx
        img_indices = points_img.astype(np.int64)

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

        # Variable to outputs
        out_dict['img'] = np.moveaxis(image, -1, 0)
        out_dict['img_indices'] = img_indices
        out_dict['scene_name'] = data_dict['seq_name']
        out_dict['scan_ID'] = data_dict['seq_ID']
        out_dict['pose'] = data_dict['pose']
        out_dict['lidar_path'] = data_dict['lidar_path']
        out_dict['pc2img_idx'] = pc2img_idx if not self.front_only else \
            np.ones(img_indices.shape[0], dtype=np.bool8)
        out_dict['proj_matrix'] = data_dict['proj_matrix']
        
        # 3D data augmentation and scaling from points to voxel indices
        coords, points = augment_and_scale_3d(points, self.scale, self.full_scale, noisy_rot=self.noisy_rot,
                                      flip_y=self.flip_y, rot_z=self.rot_z, transl=self.transl)
        coords = coords.astype(np.int64)
        if self.front_only:
            coords, points = coords[pc2img_idx], points[pc2img_idx]
            ori_points, seg_label = ori_points[pc2img_idx], seg_label[pc2img_idx]
            feats = feats[pc2img_idx]
        # idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)
        idxs = np.ones(coords.shape[0], dtype=np.bool8)
        out_dict['coords'] = coords[idxs]
        out_dict['points'] = points[idxs]    
        out_dict['seg_label'] = seg_label[idxs]
        out_dict['ori_points'] = ori_points[idxs]
        out_dict['ori_img_size'] = ori_img_size
        
        if self.backbone.upper() == "SCN":
            out_dict['feats'] = np.ones([out_dict['coords'].shape[0], 1], np.float32)  # simply use 1 as feature
        elif self.backbone.upper() == "SPVCNN" or self.backbone.upper() == "SPVCNN_BASE":
            pc_ = coords[idxs]
            # print(pc_)
            _, inds, inverse = sparse_quantize(
                pc_,
                return_index=True,
                return_inverse=True
            )
            out_dict['coords'] = pc_
            out_dict['feats'] =  np.zeros(pc_.shape[0]) if self.cat_type == "synthia" else feats[idxs]
            out_dict['lidar'] = SparseTensor(
                feats=torch.cat((
                    torch.from_numpy(points[inds]).float(), 
                    torch.from_numpy(out_dict['feats'][inds]).view(-1,1).float()
                    ), dim=1), 
                coords=torch.from_numpy(pc_[inds]).int()
            )
            out_dict['indices'] = inds
            out_dict['inverse_map'] = inverse

        if self.output_orig:
            out_dict.update({
                'orig_seg_label': seg_label,
                'orig_points_idx': idxs[out_dict['pc2img_idx']],
                'ori_keep_idx': data_dict['ori_keep_idx'],          # for non-determinstic
                'ori_img_points': data_dict['ori_img_points']       # for non-determinstic
            })

        if 'pseudo_label_2d' in data_dict.keys():
            out_dict['img_indices'] = out_dict['img_indices']
            ps_label_2d = ps_label_2d[keep_idx]
            out_dict.update({
                'pseudo_label_2d': ps_label_2d,
                'pseudo_label_3d': ps_label_3d
            })
        else:
            out_dict['img_indices'] = out_dict['img_indices']

        return out_dict


def test_SemanticKITTISCN():
    from latte.data.utils.visualize import draw_points_image_labels, draw_bird_eye_view
    np.random.seed(42)
    root_dir = 'latte/datasets/semantic_kitti'
    # ps_label_dir = '1226_ps_label'
    ps_label_dir = None
    split = ('train',)
    # split = ('val',)
    dataset = SemanticKITTISCN(split=split,
                               root_dir=root_dir,
                               ps_label_dir=ps_label_dir,
                               merge_classes=True,
                               output_orig=True,
                               bottom_crop=(480, 302)
                               )
    point_ls = []
    pose_ls = []
    for i in range(len(dataset)):
        data = dataset[i]
        point_ls.append(data['points'])
        pose_ls.append(data['pose'])
        
        # cat_ps_label = data['pseudo_label_3d']
        # pseudo_label_2d = data['pseudo_label_2d']

        # draw_points_image_labels(img, img_indices, ps_label, color_palette_type='SemanticKITTI', point_size=1)
        # draw_points_image_labels(img, img_indices, pseudo_label_2d, color_palette_type='SemanticKITTI', point_size=1)
        # assert len(pseudo_label_2d) == len(seg_label)
        # draw_bird_eye_view(coords)

        #! Debug line to check pose with frame
        if len(point_ls) == 3:
            import open3d as o3d
            pc_color = [np.zeros((point_ls[0].shape[0], 3))]
            pc_color[0][:, 0] = 255
            
            # point transform to the first scan & color assignment
            for i in range(1, len(point_ls)):
                curr_pc = np.concatenate((point_ls[i], np.ones((point_ls[i].shape[0], 1))), axis=1)
                curr_pc = ((np.linalg.inv(pose_ls[0]) @ pose_ls[i]) @ curr_pc.T).T
                point_ls[i] = curr_pc[:, :3]
                curr_color = np.zeros((point_ls[i].shape[0], 3))
                curr_color[:, i] = 255
                pc_color.append(curr_color)
            pc_color = np.concatenate(pc_color, axis=0)
            point_all = np.concatenate(point_ls, axis=0)
            
            device = o3d.core.Device("CPU:0")
            pc_o3d = o3d.t.geometry.PointCloud(device)
            pc_o3d.point['positions'] = o3d.core.Tensor(point_all[:, :3], o3d.core.float32, device)
            pc_o3d.point['colors'] = o3d.core.Tensor(pc_color, o3d.core.float32, device)
            o3d.t.io.write_point_cloud('latte/viz_samples/3scans.pcd', pc_o3d)
            
            point_ls = point_ls[1:]

            input("Press Enter to continue...")
    
    print("Completed looping semantic kitti, nothing wrong!")

def compute_class_weights():
    preprocess_dir = 'latte/datasets/semantic_kitti/preprocess/preprocess'
    split = ('train',)
    dataset = SemanticKITTIBase(split,
                                preprocess_dir,
                                merge_classes=True
                                )
    # compute points per class over whole dataset
    num_classes = len(dataset.class_names)
    points_per_class = np.zeros(num_classes, int)
    for i, data in enumerate(dataset.data):
        print('{}/{}'.format(i, len(dataset)))
        labels = dataset.label_mapping[data['seg_labels']]
        points_per_class += np.bincount(labels[labels != -100], minlength=num_classes)

    # compute log smoothed class weights
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ', class_weights / class_weights.min())


if __name__ == '__main__':
    test_SemanticKITTISCN()
    # compute_class_weights()