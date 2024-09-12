import os
import os.path as osp
import numpy as np
import pickle
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.lidarseg.utils import LidarsegClassMapper

from latte.data.nuscenes.projection import map_pointcloud_to_image
from latte.data.nuscenes import splits


def preprocess(nusc, split_names, root_dir, out_dir,
               keyword=None, keyword_action=None, subset_name=None,
               location=None):
    # cannot process day/night and location at the same time
    assert not (bool(keyword) and bool(location))
    if keyword:
        assert keyword_action in ['filter', 'exclude']

    # init dict to save
    pkl_dict = {}
    for split_name in split_names:
         pkl_dict[split_name] = {}      # use dict for each scene to track frame ID

    # fine to coarse label mapping (only 16 classes out of 32 are actually used in NuScenes lidarseg)
    class_mapper = LidarsegClassMapper(nusc)
    fine_2_carse_mapping_dict = class_mapper.get_fine_idx_2_coarse_idx()
    fine_2_coarse_mapping = np.array(
        [fine_2_carse_mapping_dict[fine_idx] for fine_idx in range(len(fine_2_carse_mapping_dict))]
    )
    
    # construct a dict of scenes, each scenes contains all sample frames
    ld_token_dict = {}
    
    for i, sample in enumerate(nusc.sample):
        curr_scene_name = nusc.get('scene', sample['scene_token'])['name']
        
        # store all raw frames of the current scene in keyframe_ID_dict
        if curr_scene_name not in ld_token_dict.keys():
            # loop from the first sample:
            token_list = []
            first_sample_token = nusc.get('scene', sample['scene_token'])['first_sample_token']
            first_sample = nusc.get('sample', first_sample_token)
            lidar_data = nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])
            token_list.append(lidar_data['token'])
            while lidar_data['next']:
                lidar_data = nusc.get('sample_data', lidar_data['next'])
                token_list.append(lidar_data['token'])
            ld_token_dict[curr_scene_name] = token_list

        # get if the current scene is in train, val or test
        curr_split = None
        for split_name in split_names:
            if curr_scene_name in getattr(splits, split_name):
                curr_split = split_name
                break
        if curr_split is None:
            continue

        if subset_name == 'night':
            if curr_split == 'train':
                if curr_scene_name in splits.val_night:
                    curr_split = 'val'
        if subset_name == 'singapore':
            if curr_split == 'train':
                if curr_scene_name in splits.val_singapore:
                    curr_split = 'val'
        if subset_name == 'all':
            if curr_split == 'train':
                if curr_scene_name in splits.val_all:
                    curr_split = 'val'

        # filter for day/night
        if keyword:
            scene_description = nusc.get("scene", sample["scene_token"])["description"]
            if keyword.lower() in scene_description.lower():
                if keyword_action == 'exclude':
                    # skip sample
                    continue
            else:
                if keyword_action == 'filter':
                    # skip sample
                    continue

        if location:
            scene = nusc.get("scene", sample["scene_token"])
            if location not in nusc.get("log", scene['log_token'])['location']:
                continue

        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        lidar_path, _, _ = nusc.get_sample_data(lidar_token)
        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_front_token)

        print('{}/{} {} {}, current split: {}'.format(i + 1, len(nusc.sample), curr_scene_name, lidar_path, curr_split))

        sd_rec_lidar = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        cs_record_lidar = nusc.get('calibrated_sensor',
                             sd_rec_lidar['calibrated_sensor_token'])
        pose_record_lidar = nusc.get('ego_pose', sd_rec_lidar['ego_pose_token'])
        sd_rec_cam = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        cs_record_cam = nusc.get('calibrated_sensor',
                             sd_rec_cam['calibrated_sensor_token'])
        pose_record_cam = nusc.get('ego_pose', sd_rec_cam['ego_pose_token'])

        calib_infos = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam['translation'],
            "ego2global_rotation_cam": pose_record_cam['rotation'],
            "cam2ego_translation": cs_record_cam['translation'],
            "cam2ego_rotation": cs_record_cam['rotation'],
            "cam_intrinsic": cam_intrinsic,
        }

        # load lidar points
        pts = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 5])[:, :3].T

        # map point cloud into front camera image
        pts_valid_flag, pts_cam_coord, pts_img, Tr_p2c = map_pointcloud_to_image(pts, (900, 1600, 3), calib_infos)
        # fliplr so that indexing is row, col and not col, row
        pts_img = np.ascontiguousarray(np.fliplr(pts_img))

        # load segmentation labels
        lidarseg_labels_filename = osp.join(nusc.dataroot, nusc.get('lidarseg', lidar_token)['filename'])
        seg_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)  # [num_points]
        seg_labels = fine_2_coarse_mapping[seg_labels]  # map from fine to coarse labels

        # convert to relative path
        lidar_path = lidar_path.replace(root_dir + '/', '')
        cam_path = cam_path.replace(root_dir + '/', '')

        # transpose to yield shape (num_points, 3)
        pts = pts.T

        # append data to train, val or test list in pkl_dict
        data_dict = {
            'seg_labels_pth': lidarseg_labels_filename,
            'points_img': pts_img,  # row, col format, shape: (num_points, 2)
            'lidar_path': lidar_path,
            'camera_path': cam_path,
            "sample_token": sample["token"],
            "scene_name": curr_scene_name,
            "calib": calib_infos,
            "pc2img_idx": pts_valid_flag,
            'timestamp': sample['timestamp'],
            'ld_token': sample['data']['LIDAR_TOP'],
            'proj_matrix': Tr_p2c
        }
        # store scene-wise scan & preserve their timestamp for sorting 
        if curr_scene_name not in pkl_dict[curr_split].keys():
            pkl_dict[curr_split][curr_scene_name] = [data_dict]
            # timestamp_dict[curr_split][curr_scene_name] = [sample['timestamp']]
        else:
            pkl_dict[curr_split][curr_scene_name].append(data_dict)
            # timestamp_dict[curr_split][curr_scene_name].append(sample['timestamp'])

    # sort scene-wise scans by timestamp
    for split in split_names:
        split_ls = []
        
        for scene in sorted(pkl_dict[split].keys()):
            sorted_scans = sorted(pkl_dict[split][scene], key=lambda x: x['timestamp'])
            ld_token_ls = ld_token_dict[scene]
            # timestamp to scan ID
            for i in range(len(sorted_scans)):
                sorted_scans[i]['scan_ID'] = ld_token_ls.index(sorted_scans[i]['ld_token'])
            split_ls.extend(sorted_scans)
        pkl_dict[split] = split_ls        
    
    # save to pickle file
    save_dir = osp.join(out_dir, 'preprocess')
    os.makedirs(save_dir, exist_ok=True)
    for split_name in split_names:
        save_path = osp.join(save_dir, '{}{}.pkl'.format(split_name, '_' + subset_name if subset_name else ''))
        with open(save_path, 'wb') as f:
            pickle.dump(pkl_dict[split_name], f)
            print('Wrote preprocessed data to ' + save_path)

def pose_generation(nusc, root_dir, pose_out_dir):
    from kiss_icp.datasets import dataset_factory
    from kiss_icp.pipeline import OdometryPipeline
    
    for seq_data in tqdm(nusc.scene):
        seq = seq_data["name"]
        seq_ID = int(seq.split('-')[-1])
        seq_out_dir = osp.join(pose_out_dir, seq)
        os.makedirs(seq_out_dir, exist_ok=True)
        OdometryPipeline(
            dataset=dataset_factory(
                dataloader="nuscenes",
                data_dir=root_dir,
                # Additional options
                sequence=seq_ID,
                topic=None,
                meta=None,
                nusc=nusc,
            ),
            config=None,
            deskew=False,
            max_range=None,
            visualize=False,
            n_scans=-1,
            jump=0,
            out_dir=seq_out_dir
        ).run()

def split_val(
    process_dir: str,
    train_pkl: str, 
    val_pkl: str, 
    val_size: int
    ):
    """
    Function to split val pickle from train pickle
    """
    from tqdm import tqdm
    import random
    random.seed(42)
    
    all_samples = []
    unique_scenes = []
    with open(osp.join(process_dir, train_pkl), 'rb') as f:
        all_samples.extend(pickle.load(f))
    
    print("Sort all availabel scene name:")
    for sample in tqdm(all_samples):
        if sample['scene_name'] not in unique_scenes:
            unique_scenes.append(sample['scene_name'])
    
    unique_scenes = sorted(unique_scenes)
    val_scenes = random.sample(unique_scenes, val_size)
    
    val_samples = []
    train_samples = []
    print("Re-organizing Val/Train split...")
    for samples in all_samples:
        if samples['scene_name'] in val_scenes:
            val_samples.append(samples)
        else:
            train_samples.append(samples)
    
    print("Saving re-organized Val/Train split...")
    with open(osp.join(process_dir, train_pkl), 'wb') as f:
        pickle.dump(train_samples, f)
    with open(osp.join(process_dir, val_pkl), 'wb') as f:
        pickle.dump(val_samples, f)
    


if __name__ == '__main__':
    root_dir = 'latte/datasets/nuscenes'
    out_dir = 'latte/datasets/nuscenes/preprocess_all'
    nusc = NuScenes(version='v1.0-trainval', dataroot=root_dir, verbose=True)
    # for faster debugging, the script can be run using the mini dataset
    # nusc = NuScenes(version='v1.0-mini', dataroot=root_dir, verbose=True)
    # We construct the splits by using the meta data of NuScenes:
    # USA/Singapore: We check if the location is Boston or Singapore.
    # Day/Night: We detect if "night" occurs in the scene description string.
    preprocess(nusc, ['train', 'test'], root_dir, out_dir, location='boston', subset_name='usa')
    preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, location='singapore', subset_name='singapore')
    preprocess(nusc, ['train', 'test'], root_dir, out_dir, keyword='night', keyword_action='exclude', subset_name='day')
    preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, keyword='night', keyword_action='filter', subset_name='night')

    # SemanticKITTI/nuScenes-lidarseg (to evaluate against LiDAR transfer baseline)
    # preprocess(nusc, ['train', 'val', 'test'], root_dir, out_dir, subset_name='all')
    # pose_out_dir = osp.join(out_dir, 'poses')
    # pose_generation(nusc, root_dir, pose_out_dir)
    split_val(out_dir + "/preprocess", "train_usa.pkl", "val_usa.pkl", 50)