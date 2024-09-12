import torch
from functools import partial

from torch.nn.functional import normalize


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

def collate_scn_seq(input_dict_list, output_orig, output_depth, output_image=True):
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

    if output_image:
        imgs = []
        imgs_2 = []
        img_idxs = []

    if output_orig:
        orig_seg_label = []
        orig_points_idx = []
    
    if output_depth:
        depth_label = []

    output_pselab = 'pseudo_label_2d' in input_dict_list[0].keys()
    if output_pselab:
        pseudo_label_2d = []
        pseudo_label_3d = []
    
    idx = 0
    for _, input_dict in enumerate(input_dict_list):
        coords_ls = input_dict['coords']
        for i in range(len(coords_ls)):
            coords = torch.from_numpy(coords_ls[i])
            batch_idxs = torch.LongTensor(coords.shape[0], 1).fill_(idx)
            idx += 1
            locs.append(torch.cat([coords, batch_idxs], 1))
        feats.extend([torch.from_numpy(input_dict['feats'][i]) 
                      for i in range(len(input_dict['feats']))])
        if 'seg_label' in input_dict.keys():
            labels.extend([torch.from_numpy(input_dict['seg_label'][i])
                           for i in range(len(input_dict['seg_label']))])
        if output_image:
            imgs.extend([torch.from_numpy(input_dict['img'][i])
                         for i in range(len(input_dict['img']))])
            img_idxs.extend([torch.from_numpy(input_dict['img_indices'][i])
                             for i in range(len(input_dict['img_indices']))])
        if output_orig:
            if 'orig_seg_label' in input_dict.keys():
                orig_seg_label.extend([torch.from_numpy(input_dict['orig_seg_label'][i])
                                       for i in range(len(input_dict['orig_seg_label']))])
            orig_points_idx.extend([torch.from_numpy(input_dict['orig_points_idx'][i])
                                    for i in range(len(input_dict['orig_points_idx']))])
        if output_pselab:
            pseudo_label_2d.append(torch.from_numpy(input_dict['pseudo_label_2d']))
            if input_dict['pseudo_label_3d'] is not None:
                pseudo_label_3d.append(torch.from_numpy(input_dict['pseudo_label_3d']))
        if 'img_2' in input_dict.keys():
            imgs_2.append(torch.from_numpy(input_dict['img_2']))

    locs = torch.cat(locs, 0)
    feats = torch.cat(feats, 0)
    out_dict = {'x': [locs, feats]}
    if labels:
        labels = torch.cat(labels, 0)
        out_dict['seg_label'] = labels
    if output_image:
        out_dict['img'] = torch.stack(imgs)
        out_dict['img_indices'] = img_idxs
    if output_orig:
        if len(orig_seg_label) != 0:
            out_dict['orig_seg_label'] = orig_seg_label
        out_dict['orig_points_idx'] = orig_points_idx
    if output_depth:
        depth_label = torch.cat(depth_label,0)
        out_dict['depth_label'] = depth_label
    if output_pselab:
        out_dict['pseudo_label_2d'] = torch.cat(pseudo_label_2d, 0)
        out_dict['pseudo_label_3d'] = torch.cat(pseudo_label_3d, 0) if pseudo_label_3d else pseudo_label_3d
    if len(imgs_2) > 0:
        out_dict['img_2'] = torch.stack(imgs_2)
    if 'lidar' in input_dict_list[0].keys():
        from torchsparse.utils.collate import sparse_collate
        out_dict['lidar'] = sparse_collate([input_dict['lidar'] for input_dict in input_dict_list])
    return out_dict


def get_collate_seq(output_orig, output_depth):
    return partial(collate_scn_seq,
                   output_orig=output_orig,
                   output_depth=output_depth)

def batch_mask_extractor(locs):
    batch_mask = []
    batch_tensor = locs[:, -1].int()
    # max_index = torch.max(batch_tensor).int()
    # min_index = torch.min(batch_tensor).int()
    # for idx in range(min_index, max_index):
    #     batch_mask.append(batch_tensor[batch_tensor == torch.LongTensor(idx)])
    batch_mask = torch.bincount(batch_tensor.int()).tolist()
    return batch_mask

