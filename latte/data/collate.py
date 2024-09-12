import torch
import numpy as np
from functools import partial

from torch.nn.functional import normalize
import torch.nn.functional as F
import logging
from torchsparse.utils.collate import sparse_collate


def depth_computation(locs):
    """
    Custom computation function to transfer point cloud to depth points
    :param locs: point cloud coord of the shape (N, 3)
    :return: normalized depth points of the shape (N,1)
    """
    # rescale the coords
    # print(locs.shape)
    locs = locs / 20

    # compute the depth distance
    x, y, z = locs[:,0], locs[:,1], locs[:,2]
    depth = torch.sqrt(x*x + y*y + z*z)

    # utlize L1 normalization
    # print(depth)
    depth_max = torch.max(depth, 0)[0]
    depth_min = torch.min(depth, 0)[0]
    depth = (depth - depth_min) / (depth_max - depth_min)
    # print(depth)

    return depth

def inverse_to_all(seg_logit, data_batch):
    # Testing line, will be removed after debugging
    num_point = 0
    indices_list = data_batch['indices']
    for inds in indices_list:
        num_point += inds.shape[0]
    # print(seg_logit.shape[0], num_point)
    # assert num_point == seg_logit.shape[0]
    # assert len(data_batch['inverse_map']) == len(indices_list)

    inv_seg_logit = []
    start_flag = 0
    # print(seg_logit.shape)
    for i in range(len(data_batch['inverse_map'])):
        map = data_batch['inverse_map'][i]
        end_flag = indices_list[i].shape[0]
        pred_label_3d = seg_logit[start_flag:start_flag+end_flag][map]
        inv_seg_logit.append(pred_label_3d)
        start_flag += end_flag
        # pred_label_voxel_3d = pred_label_voxel_3d[mask:]
    seg_logit_3d = torch.cat(inv_seg_logit, dim=0)
    return seg_logit_3d


def range_to_point(seg_logit, data_batch, post_knn=None, post=False, output_prob=False):
    """
    Function to project range image back to 3D points
    Args:
        seg_logit: range-style logit, torch tensor of shape (N, H, W, K).
        data_batch: python dictionary contains properties of range img.
        post_knn: KNN module to perform NN search
        post: bool to indicate whether KNN is used for postprocessing.
        output_prob: whether to compute prob average in NN
    Return:
        all_pc_logit: all 3D seg logit mapped from range-style logit
        sub_pc_logit: 3D seg logit mapped to image field
        all_pc_pred: all 3D pred mapped from range-style logit
        sub_pc_pred: 3D pred mapped to image field
    """
    keep_idx = data_batch['keep_idx']
    # loop over batch & check len of proj_x/y with seg_logit
    all_output_ls = []
    sub_output_ls = []
    assert seg_logit.shape[0] == len(data_batch['proj_x'])
    if post:
        if not post_knn:
            # * currently does not support post_proc without KNN
            logging.warning("Lack KNN to perform post-processing")
            raise AssertionError
        for i in range(seg_logit.shape[0]):
            sub_seg_logit = seg_logit[i] if output_prob else seg_logit[i].argmax(2)
            pc_output = post_knn(data_batch['proj_range'][i],
                                data_batch['unproj_range'][i],
                                sub_seg_logit,
                                data_batch['proj_x'][i],
                                data_batch['proj_y'][i],
                                output_prob=output_prob)
            sub_output_ls.append(pc_output[keep_idx[i]])
            all_output_ls.append(pc_output)
        # compute prob if not using output_prob
        if output_prob:
            all_pc_logit = torch.cat(all_output_ls, dim=0)
            sub_pc_logit = torch.cat(sub_output_ls, dim=0)
            all_pc_pred = all_pc_logit.argmax(1)
            sub_pc_pred = sub_pc_logit.argmax(1)
        else:
            all_pc_logit = []
            sub_pc_logit = []
            for i in range(seg_logit.shape[0]):
                sub_seg_logit = seg_logit[i]
                proj_x = data_batch['proj_x'][i]
                proj_y = data_batch['proj_y'][i]
                sub_pc_logit = sub_seg_logit[proj_y.long(), \
                                             proj_x.long(), :]
                sub_pc_logit.append(sub_pc_logit[keep_idx[i]])
                all_pc_logit.append(sub_pc_logit)
            all_pc_logit = torch.cat(all_pc_logit, dim=0)
            sub_pc_logit = torch.cat(sub_pc_logit, dim=0)
            all_pc_pred = torch.cat(all_output_ls, dim=0)
            sub_pc_pred = torch.cat(sub_output_ls, dim=0)
    else:
        for i in range(seg_logit.shape[0]):
            sub_seg_logit = seg_logit[i]
            proj_x = data_batch['proj_x'][i]
            proj_y = data_batch['proj_y'][i]
            sub_pc_logit = sub_seg_logit[proj_y.long(), \
                                         proj_x.long(), :]
            sub_output_ls.append(sub_pc_logit[keep_idx[i]])
            all_output_ls.append(sub_pc_logit)
        all_pc_logit = torch.cat(all_output_ls, dim=0)
        sub_pc_logit = torch.cat(sub_output_ls, dim=0)
        all_pc_pred = all_pc_logit.argmax(1)
        sub_pc_pred = sub_pc_logit.argmax(1)

    return all_pc_logit, sub_pc_logit, all_pc_pred, sub_pc_pred


def range_crop(proj_in):
    """
    Function to crop range img to value-existing area
    Args:
        proj_in: torch tensor of range img, of shape (N, H, W, 5)
    Return:
        crop_proj_in: cropped range img based on proj_in[:,:,:,0] > 0
    """
    h_index = torch.count_nonzero(proj_in[:,:,:,0], dim=1)
    h_min_idx = torch.min(h_index)
    h_max_idx = torch.max(h_index)
    w_index = torch.count_nonzero(proj_in[:,:,:,0], dim=2)
    w_min_idx = torch.min(w_index)
    w_max_idx = torch.max(w_index)

    return proj_in[:, h_min_idx:h_max_idx+1, w_min_idx:w_max_idx+1, :]


def collate_range_base(input_dict_list, output_orig, output_image=True):
    """
    Collate function for range-image dataloader
    """
    labels = []
    all_labels = []
    proj_labels = []
    if output_image:
        imgs = []
        img_idxs = []
    if output_orig:
        orig_seg_label = []
        orig_points_idx = []
    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []
        pseudo_proj_label = []
    # Specifically for range image
    if 'proj_in' in input_dict_list[0].keys():
        # projected variables
        proj_in_ls = []
        max_w = 0      # largest w for pad
        proj_xyz_ls = []
        proj_remission_ls = []
        proj_idx_ls = []
        proj_mask_ls = []
        proj_range_ls = []
        proj_x_ls = []
        proj_y_ls = []
        # unprojected variables
        keep_idx_ls = []
        unproj_xyz_ls = []
        unproj_range_ls = []
        unproj_remissions_ls = []
        # check whether using pc MixMatch
        if 'cat_proj_in' in input_dict_list[0].keys():
            cat_proj_in_ls = []
            obj_mask_ls = []
            cat_pslabel_ls = []
            cat_unproj_xyz_ls = []
    if 'max_feat' in input_dict_list[0].keys():
        max_feat = []

    for idx, input_dict in enumerate(input_dict_list):
        if 'seg_label' in input_dict.keys():
            labels.append(torch.from_numpy(input_dict['seg_label']))
            all_labels.append(torch.from_numpy(input_dict['all_seg_label']))
            proj_labels.append(torch.from_numpy(input_dict['proj_label']))
        if output_image:
            imgs.append(torch.from_numpy(input_dict['img']))
            img_idxs.append(input_dict['img_indices'])
        if output_orig:
            if 'orig_seg_label' in input_dict.keys():
                orig_seg_label.append(input_dict['orig_seg_label'])
            orig_points_idx.append(input_dict['orig_points_idx'])
        if output_pselab:
            pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
            if input_dict['pseudo_label_3d'] is not None:
                pseudo_label_3d.append(torch.from_numpy(input_dict['pseudo_label_3d']))
                pseudo_proj_label.append(torch.from_numpy((input_dict['pseudo_proj_label'])))
        if 'max_feat' in input_dict.keys():
            max_feat.append(input_dict['max_feat'])
        # Specifically for range images
        if 'proj_in' in input_dict.keys():
            # projected variables
            proj_in = input_dict['proj_in']
            max_w = proj_in.shape[2] if proj_in.shape[2] >= max_w else max_w
            proj_in_ls.append(proj_in)
            proj_xyz_ls.append(input_dict['proj_xyz'])
            proj_remission_ls.append(input_dict['proj_remission'])
            proj_idx_ls.append(input_dict['proj_idx'])
            proj_mask_ls.append(input_dict['proj_mask'])
            proj_range_ls.append(input_dict['proj_range'])
            proj_x_ls.append(input_dict['proj_x'])
            proj_y_ls.append(input_dict['proj_y'])
            # unprojected variables
            keep_idx_ls.append(torch.from_numpy(input_dict['keep_idx']))
            unproj_xyz_ls.append(input_dict['unproj_xyz'])
            unproj_range_ls.append(input_dict['unproj_range'])
            unproj_remissions_ls.append(input_dict['unproj_remissions'])
            # pc MixMatch
            if 'cat_proj_in' in input_dict.keys():
                cat_proj_in_ls.append(input_dict['cat_proj_in'])
                obj_mask_ls.append(input_dict['obj_mask'])
                cat_pslabel_ls.append(input_dict['cat_pseudo_label'])
                cat_unproj_xyz_ls.append(input_dict['cat_unproj_xyz'])

    out_dict = {}
    if labels:
        labels = torch.cat(labels, 0)
        all_labels = torch.cat(all_labels, 0)
        proj_labels = torch.stack(proj_labels)
        out_dict['seg_label'] = labels
        out_dict['all_seg_label'] = all_labels
        out_dict['proj_label'] = proj_labels
    if output_image:
        out_dict['img'] = torch.stack(imgs)
        out_dict['img_indices'] = img_idxs
    if output_orig:
        if len(orig_seg_label) != 0:
            out_dict['orig_seg_label'] = orig_seg_label
        out_dict['orig_points_idx'] = orig_points_idx
    if output_pselab:
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
        out_dict['pseudo_proj_label'] = torch.stack(pseudo_proj_label)
    # Specifically for range image
    if 'proj_in' in input_dict_list[0].keys():
        out_dict['proj_in'] = torch.stack(proj_in_ls)
        out_dict['proj_xyz'] = proj_xyz_ls
        out_dict['proj_remission'] = proj_remission_ls
        out_dict['proj_idx'] = proj_idx_ls
        out_dict['proj_mask'] = proj_mask_ls
        out_dict['proj_range'] = proj_range_ls
        out_dict['proj_x'] = proj_x_ls
        out_dict['proj_y'] = proj_y_ls
        # unprojected variable
        out_dict['keep_idx'] = keep_idx_ls
        out_dict['unproj_xyz'] = unproj_xyz_ls
        out_dict['unproj_range'] = unproj_range_ls
        out_dict['unproj_remissions'] = unproj_remissions_ls
        # pc MixMatch
        if 'cat_proj_in' in input_dict_list[0].keys():
            out_dict['cat_proj_in'] = torch.stack(cat_proj_in_ls)
            out_dict['cat_proj_ps_label'] = torch.stack(cat_pslabel_ls)
            out_dict['obj_mask'] = obj_mask_ls
            out_dict['cat_unproj_xyz'] = cat_unproj_xyz_ls
    if 'max_feat' in input_dict_list[0].keys():
        out_dict['max_feat'] = np.asarray(max_feat)

    return out_dict


def collate_scn_base(
        input_dict_list, 
        output_depth, 
        output_orig, 
        output_image=True,
        output_cat=False,
        dual_scan=False, 
        ):
    """
    Custom collate function for SCN. The batch size is always 1,
    but the batch indices are appended to the locations.
    :param input_dict_list: a list of dicts from the dataloader
    :param output_orig: whether to output original point cloud/labels/indices
    :param output_depth: whether to output depth points
    :param output_image: whether to output images
    :return: Collated data batch as dict
    """
    locs=[]
    feats=[]
    labels=[]
    inverse_map = []
    indices_list = []
    lidar_pth_ls = []
    seq_ID_ls = []
    seq_name_ls = []
    # partial FOV mapping I/O
    pc2img_idx_ls = []
    scan_ID_ls = []
    scene_name_ls = []
    ori_point_ls = []
    pose_ls = []
    # Dense 3D to 2D
    proj_matrix_ls = []
    ori_img_size = []
    if output_image:
        imgs = []
        img_idxs = []
    if output_orig:
        orig_seg_label = []
        orig_points_idx = []
        ori_keep_idx = []
        ori_img_points = []
    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []

    for idx, input_dict in enumerate(input_dict_list):
        coords = torch.from_numpy(input_dict['coords'])
        batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
        locs.append(torch.cat([coords, batch_idxs], 1))
        feats.append(torch.from_numpy(input_dict['feats']))
        if 'lidar_path' in input_dict.keys():
            lidar_pth_ls.append(input_dict['lidar_path'])
        if 'seg_label' in input_dict.keys():
            labels.append(torch.from_numpy(input_dict['seg_label']))
        if output_image:
            imgs.append(torch.from_numpy(input_dict['img']))
            img_idxs.append(input_dict['img_indices'])
        if output_orig:
            if 'orig_seg_label' in input_dict.keys():
                orig_seg_label.append(input_dict['orig_seg_label'])
            orig_points_idx.append(input_dict['orig_points_idx'])
            if 'ori_keep_idx' in input_dict.keys():
                ori_keep_idx.append(input_dict['ori_keep_idx'])
                ori_img_points.append(input_dict['ori_img_points'])
        if output_pselab:
            pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
            if input_dict['pseudo_label_3d'] is not None:
                pseudo_label_3d.append(torch.from_numpy(input_dict['pseudo_label_3d']))
        # Specifically for SPVCNN
        if 'inverse_map' in input_dict.keys():
            inverse_map.append(input_dict['inverse_map'])
        if 'indices' in input_dict.keys():
            indices_list.append(input_dict['indices'])
        # Patrial I/O mapping
        if 'pc2img_idx' in input_dict.keys():
            pc2img_idx_ls.append(torch.from_numpy(input_dict['pc2img_idx'].astype(np.bool8)))
        if 'scene_name' in input_dict.keys():
            scene_name_ls.append(input_dict['scene_name'])
        if 'scan_ID' in input_dict.keys():
            scan_ID_ls.append(input_dict['scan_ID'])
            seq_name_ls.append(input_dict['scene_name'])
        if 'ori_points' in input_dict.keys():
            ori_point_ls.append(torch.from_numpy(input_dict['ori_points']))
        if 'pose' in input_dict.keys():
            pose_ls.append(torch.from_numpy(input_dict['pose']))
        if 'proj_matrix' in input_dict.keys():
            proj_matrix_ls.append(torch.from_numpy(input_dict['proj_matrix']))
        if 'ori_img_size' in input_dict.keys():
            ori_img_size.append(input_dict['ori_img_size'])

    locs = torch.cat(locs, 0)
    feats_cat = torch.cat(feats, 0)
    out_dict = {'x': [locs, feats_cat]}
    out_dict['lidar_path'] = lidar_pth_ls
    out_dict['scan_ID'] = scan_ID_ls
    out_dict['scene_name'] = seq_name_ls
    if labels:
        labels = torch.cat(labels, 0)
        out_dict['seg_label'] = labels
    if output_image:
        try:
            out_dict['img'] = torch.stack(imgs)
        except RuntimeError:
            max_H = max([img.shape[1] for img in imgs])
            max_W = max([img.shape[2] for img in imgs])
            out_dict['img'] = torch.zeros((len(imgs), 3, max_H, max_W))
            for j in range(len(imgs)):
                out_dict['img'][j, :, :imgs[j].shape[1], :imgs[j].shape[2]] = imgs[j]
        out_dict['img_indices'] = img_idxs
    if output_orig:
        if len(orig_seg_label) != 0:
            out_dict['orig_seg_label'] = orig_seg_label
        out_dict['orig_points_idx'] = orig_points_idx
        out_dict['ori_keep_idx'] = ori_keep_idx
        out_dict['ori_img_points'] = ori_img_points
    if output_pselab:
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
    if 'lidar' in input_dict_list[0].keys():
        out_dict['lidar'] = sparse_collate([input_dict['lidar'] for input_dict in input_dict_list])
        out_dict['feats'] = feats
    # Specifically for SPVCNN
    if len(inverse_map) > 0:
        out_dict['inverse_map'] = inverse_map
    if len(indices_list) > 0:
        out_dict['indices'] = indices_list
    # For partial FOV
    if len(pc2img_idx_ls) > 0:
        out_dict['pc2img_idx'] = pc2img_idx_ls
    if len(ori_point_ls) > 0:
        out_dict['ori_points'] = ori_point_ls
    if len(pose_ls) > 0:
        out_dict['pose'] = pose_ls
    if len(proj_matrix_ls) > 0:
        out_dict['proj_matrix'] = proj_matrix_ls
    if len(ori_img_size) > 0:
        out_dict['ori_img_size'] = ori_img_size

    return out_dict

def collate_seq_base_v2(input_dict_list, output_orig, output_image=True):
    # TODO: complete collate function for unified seq collation
    # combine all input_dict into a single list
    # and preserve index indicating the seq relationship
    seq_len = []
    input_dict_ls = []
    ori_idx_ls = []
    for input_dict in input_dict_list:
        input_dict_ls.extend(input_dict)
        seq_len.append(len(input_dict))
        ori_idx_ls.append(input_dict[0]['ori_idx'])
    input_dict_list = input_dict_ls

    # allocate list of inputs
    # shared collate options
    labels=[]
    lidar_pth_ls = []
    if output_image:
        imgs = []
        img_idxs = []
    if output_orig:
        orig_seg_label = []
        orig_points_idx = []
    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []
        
    # option 1: SPVCNN/SCN
    if 'proj_in' not in input_dict_list[0].keys():
        inverse_map = []
        indices_list = []
        keep_idx_ls = []
        points_r_ls = []
        locs = []
        feats = []
    # option 2: SalsaNext
    else:
        # projected variables
        proj_in_ls = []
        max_w = 0      # largest w for pad
        proj_xyz_ls = []
        proj_remission_ls = []
        proj_idx_ls = []
        proj_mask_ls = []
        proj_range_ls = []
        proj_x_ls = []
        proj_y_ls = []
        # unprojected variables
        keep_idx_ls = []
        unproj_xyz_ls = []
        unproj_range_ls = []
        unproj_remissions_ls = []

    for idx, input_dict in enumerate(input_dict_list):
        # shared variables
        if 'seg_label' in input_dict.keys() and input_dict['seg_label'] is not None:
            labels.append(torch.from_numpy(input_dict['seg_label']))
        keep_idx_ls.append(input_dict['keep_idx'])                          # pc_keep_idx
        points_r_ls.append(input_dict['points_r'])                          # registered pc
        lidar_pth_ls.append(input_dict['lidar_path'])
        if output_image:
            imgs.append(torch.from_numpy(input_dict['img']))                # images
            img_idxs.append(input_dict['img_indices'])                      # img_indices
        # original output for validation
        if output_orig and \
            'orig_seg_label' in input_dict.keys() and \
            input_dict['orig_seg_label'] is not None:
                orig_seg_label.append(input_dict['orig_seg_label'])
                orig_points_idx.append(input_dict['orig_points_idx'])
        # pseudo labels
        if output_pselab:
            pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
            if input_dict['pseudo_label_3d'] is not None:
                pseudo_label_3d.append(torch.from_numpy(input_dict['pseudo_label_3d']))
        # Network specific variable collation
        # option 1: SPVCNN/SCN
        if 'proj_in' not in input_dict:
            coords = torch.from_numpy(input_dict['coords'])
            batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
            locs.append(torch.cat([coords, batch_idxs], 1))
            feats.append(torch.from_numpy(input_dict['feats']))
            # Specifically for SPVCNN
            if 'inverse_map' in input_dict.keys():
                inverse_map.append(input_dict['inverse_map'])
            if 'indices' in input_dict.keys():
                indices_list.append(input_dict['indices'])
        # option 2: SalsaNext
        else:
            # projected variables
            proj_in = input_dict['proj_in']
            max_w = proj_in.shape[2] if proj_in.shape[2] >= max_w else max_w
            proj_in_ls.append(proj_in)
            proj_xyz_ls.append(input_dict['proj_xyz'])
            proj_remission_ls.append(input_dict['proj_remission'])
            proj_idx_ls.append(input_dict['proj_idx'])
            proj_mask_ls.append(input_dict['proj_mask'])
            proj_range_ls.append(input_dict['proj_range'])
            proj_x_ls.append(input_dict['proj_x'])
            proj_y_ls.append(input_dict['proj_y'])
            # unprojected variables
            keep_idx_ls.append(torch.from_numpy(input_dict['keep_idx']))
            unproj_xyz_ls.append(input_dict['unproj_xyz'])
            unproj_range_ls.append(input_dict['unproj_range'])
            unproj_remissions_ls.append(input_dict['unproj_remissions'])

    out_dict = {}
    # shared variables
    out_dict['keep_idx'] = keep_idx_ls
    out_dict['seq_length'] = seq_len
    out_dict['points'] = points_r_ls
    out_dict['ori_idx'] = ori_idx_ls
    out_dict['lidar_path'] = lidar_pth_ls
    if labels:
        out_dict['seg_label'] = torch.cat(labels, 0)
    else:
        out_dict['seg_label'] = None
    if output_image:
        out_dict['img'] = torch.stack(imgs)
        out_dict['img_indices'] = img_idxs
    if output_orig:
        if len(orig_seg_label) != 0:
            out_dict['orig_seg_label'] = orig_seg_label
        out_dict['orig_points_idx'] = orig_points_idx
    if output_pselab:
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
    # Network specific variables
    # option 1: SPVCNN/SCN
    if 'proj_in' not in input_dict_list[0].keys():
        locs = torch.cat(locs, 0)
        feats = torch.cat(feats, 0)
        out_dict['x'] = [locs, feats]
        # Specifically for SPVCNN
        if 'lidar' in input_dict_list[0].keys():
            from torchsparse.utils.collate import sparse_collate
            out_dict['lidar'] = sparse_collate([input_dict['lidar'] for input_dict in input_dict_list])
            out_dict['inverse_map'] = inverse_map
            out_dict['indices'] = indices_list
    # option 2: SalsaNext
    else:
        out_dict['proj_in'] = torch.stack(proj_in_ls)
        out_dict['proj_xyz'] = proj_xyz_ls
        out_dict['proj_remission'] = proj_remission_ls
        out_dict['proj_idx'] = proj_idx_ls
        out_dict['proj_mask'] = proj_mask_ls
        out_dict['proj_range'] = proj_range_ls
        out_dict['proj_x'] = proj_x_ls
        out_dict['proj_y'] = proj_y_ls
        # unprojected variable
        out_dict['keep_idx'] = keep_idx_ls
        out_dict['unproj_xyz'] = unproj_xyz_ls
        out_dict['unproj_range'] = unproj_range_ls
        out_dict['unproj_remissions'] = unproj_remissions_ls

    return out_dict

def collate_seq_base(input_dict_list, output_orig, output_image=True):
    # combine all input_dict into a single list
    # and preserve index indicating the seq relationship
    seq_len = []
    input_dict_ls = []
    for input_dict in input_dict_list:
        input_dict_ls.extend(input_dict)
        seq_len.append(len(input_dict))
    input_dict_list = input_dict_ls

    # allocate list of inputs
    labels=[]
    inverse_map = []
    indices_list = []
    keep_idx_ls = []
    lidar_pth_ls = []
    point_ls = []
    if output_image:
        imgs = []
        imgs_2 = []
        img_idxs = []
    if output_orig:
        orig_seg_label = []
        orig_points_idx = []
    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []

    for idx, input_dict in enumerate(input_dict_list):
        lidar_pth_ls.append(input_dict['lidar_path'])
        point_ls.append(input_dict['points'])
        if 'seg_label' in input_dict.keys():
            if input_dict['seg_label'] is not None:
                labels.append(torch.from_numpy(input_dict['seg_label']))
            else:
                labels.append(None)
        if output_image:
            imgs.append(torch.from_numpy(input_dict['img']))
            img_idxs.append(input_dict['img_indices'])
        if output_orig:
            if 'orig_seg_label' in input_dict.keys():
                orig_seg_label.append(input_dict['orig_seg_label'])
            orig_points_idx.append(input_dict['orig_points_idx'])
        if output_pselab:
            pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
            if input_dict['pseudo_label_3d'] is not None:
                pseudo_label_3d.append(torch.from_numpy(input_dict['pseudo_label_3d']))
        if 'img_2' in input_dict.keys():
            imgs_2.append(torch.from_numpy(input_dict['img_2']))
        # Specifically for SPVCNN
        if 'inverse_map' in input_dict.keys():
            inverse_map.append(input_dict['inverse_map'])
        if 'indices' in input_dict.keys():
            indices_list.append(input_dict['indices'])

    out_dict = {}
    out_dict['lidar_path'] = lidar_pth_ls
    out_dict['keep_idx'] = keep_idx_ls
    out_dict['seq_length'] = seq_len
    out_dict['points'] = point_ls
    if labels:
        out_dict['seg_label'] = labels
    else:
        out_dict['seg_label'] = [None]
    if output_image:
        out_dict['img'] = torch.stack(imgs)
        out_dict['img_indices'] = img_idxs
    if output_orig:
        if len(orig_seg_label) != 0:
            out_dict['orig_seg_label'] = orig_seg_label
        out_dict['orig_points_idx'] = orig_points_idx
    if output_pselab:
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
    if len(imgs_2) > 0:
        out_dict['img_2'] = torch.stack(imgs_2)
    if 'lidar' in input_dict_list[0].keys():
        from torchsparse.utils.collate import sparse_collate
        out_dict['lidar'] = sparse_collate([input_dict['lidar'] for input_dict in input_dict_list])
    # Specifically for SPVCNN
    if len(inverse_map) > 0:
        out_dict['inverse_map'] = inverse_map
    if len(indices_list) > 0:
        out_dict['indices'] = indices_list

    return out_dict

def get_collate_scn(output_orig, output_depth, output_cat=False, dual_scan=False):
    return partial(collate_scn_base,
                   output_orig=output_orig,
                   output_depth=output_depth,
                   output_cat=output_cat,
                   dual_scan=dual_scan)

def get_collate_range(output_orig):
    return partial(collate_range_base,
                   output_orig=output_orig)

def get_collate_seq(output_orig):
    return partial(collate_seq_base,
                   output_orig=output_orig)

def get_collate_seq_v2(output_orig):
    return partial(collate_seq_base_v2,
                   output_orig=output_orig)

def batch_mask_extractor(locs):
    batch_mask = []
    batch_tensor = locs[:, -1].int()
    # max_index = torch.max(batch_tensor).int()
    # min_index = torch.min(batch_tensor).int()
    # for idx in range(min_index, max_index):
    #     batch_mask.append(batch_tensor[batch_tensor == torch.LongTensor(idx)])
    batch_mask = torch.bincount(batch_tensor.int()).tolist()
    return batch_mask

