import numpy as np
import random


def augment_and_scale_3d(points, 
                         scale, full_scale,
                         noisy_rot=0.0,
                         flip_x=0.0,
                         flip_y=0.0,
                         rot_z=0.0,
                         transl=False):
    """
    3D point cloud augmentation and scaling from points (in meters) to voxels
    :param points: 3D points in meters
    :param scale: voxel scale in 1 / m, e.g. 20 corresponds to 5cm voxels
    :param full_scale: size of the receptive field of SparseConvNet
    :param noisy_rot: scale of random noise added to all elements of a rotation matrix
    :param flip_x: probability of flipping the x-axis (left-right in nuScenes LiDAR coordinate system)
    :param flip_y: probability of flipping the y-axis (left-right in Kitti LiDAR coordinate system)
    :param rot_z: angle in rad around the z-axis (up-axis)
    :param transl: True or False, random translation inside the receptive field of the SCN, defined by full_scale
    :return coords: the coordinates that are given as input to SparseConvNet
    """
    rot_matrix = None
    if noisy_rot > 0 or flip_x > 0 or flip_y > 0 or rot_z > 0:
        rot_matrix = np.eye(3, dtype=np.float32)
        if noisy_rot > 0:
            # add noise to rotation matrix
            rot_matrix += np.random.randn(3, 3) * noisy_rot
        if flip_x > 0:
            # flip x axis: multiply element at (0, 0) with 1 or -1
            rot_matrix[0][0] *= np.random.randint(0, 2) * 2 - 1
        if flip_y > 0:
            # flip y axis: multiply element at (1, 1) with 1 or -1
            rot_matrix[1][1] *= np.random.randint(0, 2) * 2 - 1
        if rot_z > 0:
            # rotate around z-axis (up-axis)
            # theta = np.random.rand() * rot_z
            # * debug
            theta = rot_z
            z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]], dtype=np.float32)
            rot_matrix = rot_matrix.dot(z_rot_matrix)

        # scale with inverse voxel size (e.g. 20 corresponds to 5cm)
        # translate points to positive octant (receptive field of SCN in x, y, z coords is in interval [0, full_scale])
    if type(points) is not list:
        points = points.dot(rot_matrix) if rot_matrix is not None else points
        coords = np.round(points * scale)
        coords -= coords.min(0)
        if transl:
            # random translation inside receptive field of SCN
            offset = np.clip(full_scale - coords.max(0) - 0.001, a_min=0, a_max=None) * np.random.rand(3)
            coords += offset
        return coords, points
    else:
        transl_mtx = np.random.rand(3)
        coords = []
        arg_points = []
        for point in points:
            point = point.dot(rot_matrix) if rot_matrix is not None else point
            arg_points.append(point)
            coord = point * scale
            coord -= coord.min(0)
            if transl:
                # random translation inside receptive field of SCN
                offset = np.clip(full_scale - coord.max(0) - 0.001, a_min=0, a_max=None) * transl_mtx
                coord += offset
            coords.append(coord)
    return coords, arg_points

def range_augmentation(points, 
                       noisy_rot=0.0,
                       flip_x=0.0,
                       flip_y=0.0,
                       rot_z=0.0,
                       transl=False):
    """
    Range augmentation for salsanext
    """
    if transl:
        jitter_x = random.uniform(-5,5)
        jitter_y = random.uniform(-3, 3)
        jitter_z = random.uniform(-1, 0)
        points[:, 0] += jitter_x
        points[:, 1] += jitter_y
        points[:, 2] += jitter_z

    rot_matrix = None
    if noisy_rot > 0 or flip_x > 0 or flip_y > 0 or rot_z > 0:
        rot_matrix = np.eye(3, dtype=np.float32)
        if noisy_rot > 0:
            # add noise to rotation matrix
            rot_matrix += np.random.randn(3, 3) * noisy_rot
        if flip_x > 0:
            # flip x axis: multiply element at (0, 0) with 1 or -1
            rot_matrix[0][0] *= np.random.randint(0, 2) * 2 - 1
        if flip_y > 0:
            # flip y axis: multiply element at (1, 1) with 1 or -1
            rot_matrix[1][1] *= np.random.randint(0, 2) * 2 - 1
        if rot_z > 0:
            # rotate around z-axis (up-axis)
            theta = np.random.rand() * rot_z
            # * debug
            # theta = rot_z
            z_rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]], dtype=np.float32)
            rot_matrix = rot_matrix.dot(z_rot_matrix)

    # process points or points_ls
    if type(points) is list:
        points = [point.dot(rot_matrix) if rot_matrix is not None else point \
                    for point in points]
    else:
        points = points.dot(rot_matrix) if rot_matrix is not None else points
    return points

def range_projection(
        points: np.ndarray, 
        fov_up: float, 
        fov_down: float, 
        proj_W: int, 
        proj_H: int, 
        crop=False, 
        obj_mask: np.ndarray = None,
        ) -> dict:
    """
    Projection function to transform point cloud into range images.
    Args:
        points: 
            ndarray of shape (N, 3) point cloud
        fov_up: 
            field of view up in rad
        fov_down: 
            field of view down in rad
        proj_W: 
            width of projected range image
        proj_H: 
            height of projected range image
        crop: 
            whether crop range image to area with valid pixels
        obj_mask:
            mask indicating the location of newly added object pc 
        idx_only:
            whether to return not occluded index only (for SCN)
    Return:
        dictionay cotaning all range
    """
    remissions = points[:, 3]
    # compute the vertical fov of points
    fov = abs(fov_down) + abs(fov_up)
    # rescale size of range image based on h-fov
    # proj_W = int((proj_W * fov_h / (2 * np.pi)))

    # get depth of all points
    points = points[:, :3]
    depth = np.linalg.norm(points, 2, axis=1)

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    inter_pitch = scan_z / depth
    pitch = np.arcsin(inter_pitch)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W
    proj_y *= proj_H

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    # * specifically for part of point cloud
    if crop:
        proj_x = proj_x - np.min(proj_x)
        proj_W = np.max(proj_x) + 1

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # discard the pixel that occluded by added objects
    if obj_mask is not None:
        obj_proj_y = proj_y[obj_mask]
        obj_proj_x = proj_x[obj_mask]
        obj_proj_yx = np.concatenate(
            (obj_proj_y.reshape((-1, 1)), obj_proj_x.reshape((-1, 1))),
            axis=1
        )       # projected object points
        proj_yx = np.concatenate(
            (proj_y.reshape((-1, 1)),proj_x.reshape((-1, 1))), 
             axis=1
        )       # all projected points
        disc_idx = np.any(
            np.all(proj_yx[:, np.newaxis, :] == obj_proj_yx, axis=2),
            axis=1
        )
        pres_idx = np.logical_not(np.logical_xor(disc_idx, obj_mask))
        # filter all variable
        proj_y = proj_y[pres_idx]
        proj_x = proj_x[pres_idx]
        depth = depth[pres_idx]
        points = points[pres_idx, :]
        remissions = remissions[pres_idx]

    # copy original order of x, y ,depth
    ori_proj_y = np.copy(proj_y)
    ori_proj_x = np.copy(proj_x)
    ori_depth = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = points[order]
    remission = remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # generate range images and related outputs
    proj_range = np.full((proj_H, proj_W), -1, dtype=np.float32)
    proj_xyz = np.full((proj_H, proj_W, 3), -1,dtype=np.float32)
    proj_idx = np.full((proj_H, proj_W), -1, dtype=np.int32)
    proj_remission = np.full((proj_H, proj_W), -1, dtype=np.float32)
    
    proj_range[proj_y, proj_x] = depth
    proj_xyz[proj_y, proj_x] = points[:,:3]
    proj_remission[proj_y, proj_x] = remission
    proj_idx[proj_y, proj_x] = indices
    proj_mask = (proj_idx > 0).astype(np.int32)

    out_dict = {
        "proj_range": proj_range,
        "proj_xyz": proj_xyz,
        "proj_remission": proj_remission,
        "proj_idx": proj_idx,
        "proj_mask": proj_mask,
        "proj_x": ori_proj_x,
        "proj_y": ori_proj_y,
        "unproj_range": ori_depth,
        "unproj_remissions": remissions
        }
    
    if obj_mask is not None:
        out_dict.update({
            "pres_idx": pres_idx
        })
    
    return out_dict
