import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter.scatter as ts_scatter
import matplotlib.pyplot as plt

import numpy as np
from typing import Dict, List, Union, Tuple
from itertools import repeat
from latte.data.utils.refine_pseudo_labels import refine_ety

from latte.data.utils.visualize import colored_point_viz, NUSCENES_LIDARSEG_COLOR_PALETTE
from latte.models.losses import prob_2_entropy

def ravel_hash_cuda(x: torch.Tensor, min_offset: int = None) -> torch.Tensor:
    assert x.ndim == 2, x.shape
    x -= min_offset if min_offset is not None else torch.amin(x, dim=0)
    x = x.long()
    xmax = torch.amax(x, dim=0).long() + 1

    h = torch.zeros(x.shape[0], dtype=torch.int64).cuda()
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    
    return h


class HR_Block(nn.Module):
    """
    Module to output temporally correspondence
    This Module check the scene name and scan ID of anchor and hist,
    firstly find consecutive frames, and secondly find voxel-wise
    correspondence
    """
    def __init__(
        self, 
        temp_wd_size: int = 50, 
        max_range: int = 50,
        voxel_size: float = 0.2,
        conf_perc: float = 0.75,
        exclude_labels: Tuple[int] = None, 
        resize: Tuple[int, int] = None,     # (W, H)
        ):
        super(HR_Block, self).__init__()
        self.temp_wd_size = temp_wd_size
        self.temp_bond = math.floor(temp_wd_size / 2)
        self.max_range = max_range
        self.voxel_size = voxel_size
        # pre-defined offset for fixed origin
        self.min_offset = - math.ceil(max_range / voxel_size) - 1 
        self.conf_perc = conf_perc
        self.exclude_labels = torch.Tensor(exclude_labels) if exclude_labels else None
        self.resize = resize

    @staticmethod
    def ravel_hash(x: np.ndarray, min_offset: float) -> np.ndarray:
        assert x.ndim == 2, x.shape

        x -= min_offset
        x = x.astype(np.uint64, copy=False)
        xmax = np.max(x, axis=0).astype(np.uint64) + 1

        h = np.zeros(x.shape[0], dtype=np.uint64)
        for k in range(x.shape[1] - 1):
            h += x[:, k]
            h *= xmax[k + 1]
        h += x[:, -1]
        return h

    def sparse_quantize(self,
                        coords,
                        *,
                        return_index: bool = False,
                        return_inverse: bool = False) -> List[torch.Tensor]:
        """Modified version from torchsparse, use fix offset to ensure fix origin
        """
        if isinstance(self.voxel_size, (float, int)):
            voxel_size = tuple(repeat(self.voxel_size, 3))
        assert isinstance(voxel_size, tuple) and len(voxel_size) == 3

        voxel_size = np.array(voxel_size)
        coords = np.floor(coords / voxel_size).astype(np.int32)

        _, indices, inverse_indices = np.unique(self.ravel_hash(coords, self.min_offset),
                                                return_index=True,
                                                return_inverse=True)
        coords = torch.from_numpy(coords[indices])

        outputs = [coords]
        if return_index:
            outputs += [torch.from_numpy(indices)]
        if return_inverse:
            outputs += [torch.from_numpy(inverse_indices)]
        return outputs[0] if len(outputs) == 1 else outputs
    
    @staticmethod
    def batch_to_list(
        batch_logits: torch.Tensor, 
        list_ref: List[np.ndarray]
        ) -> List[torch.Tensor]:
        """Function to re-organize batch tensor to scan lists
        """
        left_idx = 0
        list_logits = []
        
        for scan_sample in list_ref:
            right_idx = left_idx + scan_sample.shape[0]
            list_logits.append(batch_logits[left_idx:right_idx, :])
            left_idx = right_idx
        
        return list_logits
    
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
    
    def forward(
        self, 
        anchor_batch: Dict, anchor_type: str, 
        hist_batch: Dict, hist_type: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward function to establish correspondence between anchor and histor
        Args:
            anchor_batch (Dict): anchor batch of data 
            hist_batch (Dict): historical batch of data
        """
        # Make sure needed infos are provided
        required_key = ['seg_logit', 'pose', 'scene_name', 'scan_ID', 'ori_points']
        assert set(required_key) <= set(anchor_batch.keys())
        assert set(required_key) <= set(hist_batch.keys())
        assert anchor_type.upper() in ["2D", "3D"]
        assert hist_type.upper() in ["2D", "3D"]
        
        # Re-organize batch logits
        ac_list_ref = 'img_indices' if anchor_type.upper() == "2D" else 'ori_points'
        hs_list_ref = 'img_indices' if hist_type.upper() == "2D" else 'ori_points'
        
        # Construct scene and scan ID tensor (scene_ID, scan_ID)
        # TODO: use int for scene ID when loading data
        if '-' in anchor_batch['scene_name'][0]:
            anchor_scenes = torch.Tensor(
                [int(sample.split('-')[-1]) for sample in anchor_batch['scene_name']]
                )
            hist_scenes = torch.Tensor(
                [int(sample.split('-')[-1]) for sample in hist_batch['scene_name']]
                )
        else:
            anchor_scenes = torch.Tensor(
                [int(sample) for sample in anchor_batch['scene_name']]
                )
            hist_scenes = torch.Tensor(
                [int(sample) for sample in hist_batch['scene_name']]
                )
        anchor_IDs = torch.Tensor([sample for sample in anchor_batch['scan_ID']])
        hist_IDs = torch.Tensor([sample for sample in hist_batch['scan_ID']])
        
        # Anchor init
        corr_ac_logits, corr_tr_logits, corr_ac_voxels, corr_tr_voxels = [[] for _ in range(4)]
        # Histrorical init
        hs_img_points, hs_pc2img_idx = [], []
        # Point-to-Voxel indices
        ac_p2v_indices, tr_p2v_indices, corr_voxel_ety = [], [], []
        hs_pc_ety = torch.zeros(hist_batch['seg_logit'].shape[0]).cuda()
        hs_pc_ety_count = torch.zeros(hist_batch['seg_logit'].shape[0]).int().cuda()
        
        # Entropy-based output
        ety_loss = []
        ac_left_idx = 0
        for i, curr_pose in enumerate(anchor_batch['pose']):
            # Specify corresponding samples
            curr_scene = anchor_scenes[i]
            curr_ID = anchor_IDs[i]
            corr_indices = torch.nonzero(torch.logical_and(
                hist_scenes == curr_scene,
                torch.abs(hist_IDs - curr_ID) <= self.temp_bond
            )).reshape(-1)
            corr_indices = corr_indices.tolist()
            ac_right_idx = ac_left_idx + anchor_batch[ac_list_ref][i].shape[0]
        
            # Transpose corresponding samples to current scan
            tr_points, tr_logits =  [], []
            
            # Compute the hs start index
            before_hs_idx = list(range(min(corr_indices)))
            init_hr_left = sum([hist_batch[hs_list_ref][j].shape[0] for j in before_hs_idx])
            hs_left_idx = init_hr_left
            
            # Sorting historical reference
            for idx in corr_indices:
                hs_right_idx = hs_left_idx + hist_batch[hs_list_ref][idx].shape[0]
                points = torch.cat(
                    [hist_batch['ori_points'][idx], 
                     torch.ones((hist_batch['ori_points'][idx].shape[0], 1))], 
                    dim=1
                )
                relative_Tr = torch.linalg.inv(curr_pose) @ hist_batch['pose'][idx]
                tr_points.append((relative_Tr @ points.double().T).T[:, :3])
                tr_logits.append(hist_batch['seg_logit'][hs_left_idx:hs_right_idx])
                
                # Update historical slide index
                hs_left_idx = hs_right_idx
            
            # Range-based filtering
            tr_points = torch.cat(tr_points, dim=0)
            tr_logits = torch.cat(tr_logits, dim=0)
            tr_mask = torch.linalg.norm(tr_points, ord=2, dim=1) <= self.max_range
            tr_points, tr_logits = tr_points[tr_mask], tr_logits[tr_mask]
            ac_mask = torch.linalg.norm(anchor_batch['ori_points'][i], ord=2, dim=1) <= self.max_range
            ac_points = anchor_batch['ori_points'][i][ac_mask]
            ac_logits = anchor_batch['seg_logit'][ac_left_idx:ac_right_idx][ac_mask]
            
            # Voxelization
            tr_voxels, tr_indices, tr_inverse  = self.sparse_quantize(tr_points.numpy(), return_inverse=True, return_index=True)
            ac_voxels, ac_indices, ac_inverse = self.sparse_quantize(ac_points.numpy(), return_inverse=True, return_index=True)
            
            tr_inverse, ac_inverse = tr_inverse.cuda(), ac_inverse.cuda()
            tr_voxel_logits = ts_scatter(F.softmax(tr_logits, dim=1), tr_inverse, dim=0, reduce="mean")
            ac_voxel_logits = ts_scatter(F.softmax(ac_logits, dim=1), ac_inverse, dim=0, reduce="mean")
            
            # shape check
            try:
                assert tr_voxel_logits.shape[0] == tr_voxels.shape[0]
            except AssertionError:
                print("Voxel shape: {} vs Voxel Logits shape: {}".format(tr_voxel_logits.shape[0], tr_voxels.shape[0]))
            try:
                assert ac_voxel_logits.shape[0] == ac_voxels.shape[0]
            except AssertionError:
                print("Voxel shape: {} vs Voxel Logits shape: {}".format(ac_voxel_logits.shape[0], ac_voxels.shape[0]))
            
            # voxel entropy / pixel-voxel entropy
            # Option 1: average entropy of all logits
            tr_pc_entropy = prob_2_entropy(F.softmax(tr_logits, dim=1)).sum(1)
            tr_voxel_entropy = ts_scatter(tr_pc_entropy, tr_inverse, dim=0, reduce="mean")
            
            # Option 2: entropy of avg logits
            # tr_voxel_entropy = prob_2_entropy(tr_voxel_logits).sum(1)
            hs_pc_ety[init_hr_left:hs_right_idx][tr_mask] = \
                (hs_pc_ety[init_hr_left:hs_right_idx][tr_mask] + tr_voxel_entropy[tr_inverse]).detach()
            hs_pc_ety_count[init_hr_left:hs_right_idx][tr_mask] = \
                (hs_pc_ety_count[init_hr_left:hs_right_idx][tr_mask] + 1).detach()
            
            # generating retrivial indices
            tr_p2v = tr_mask.nonzero().reshape(-1)
            tr_p2v = tr_p2v[tr_indices] + init_hr_left
            ac_p2v = ac_mask.nonzero().reshape(-1)
            ac_p2v = ac_p2v[ac_indices] + ac_left_idx
            
            # Find corresponding voxels & their logits
            with torch.no_grad():
                tr_voxels = tr_voxels.cuda()
                ac_voxels = ac_voxels.cuda()
                corr_mtx = torch.all(ac_voxels[:, None, :] == tr_voxels, dim=2)
                
                # Check the existence of corr
                try:
                    assert torch.any(corr_mtx)
                    assert torch.all(tr_voxels >= 0)
                    assert torch.all(ac_voxels >= 0)
                except AssertionError:
                    print("No correspondence found!")
                    raise AssertionError
                
                tr_corr = torch.any(corr_mtx, dim=0)
                ac_corr = torch.any(corr_mtx, dim=1)
            
                del corr_mtx
            
            # Save corr variables
            corr_ac_logits.append(ac_voxel_logits[ac_corr])
            corr_ac_voxels.append(ac_voxels[ac_corr].detach())
            corr_tr_logits.append(tr_voxel_logits[tr_corr].detach())
            corr_tr_voxels.append(tr_voxels[tr_corr].detach())
            # corr_conf_mask.append(ety_conf_mask[tr_corr].detach())
            corr_voxel_ety.append(tr_voxel_entropy)
            # save index details for v-ety retrivial
            ac_p2v_indices.append(ac_p2v[ac_corr.cpu().nonzero().reshape(-1)])
            tr_p2v_indices.append(tr_p2v[tr_corr.cpu().nonzero().reshape(-1)])
            
            del tr_inverse, ac_inverse, tr_corr, ac_corr
            torch.cuda.empty_cache()
            
            # Update anchor slide index
            ac_left_idx = ac_right_idx

        # Compute average ST-voxel entropy after iteration
        valid_mask = hs_pc_ety_count > 0 
        hs_pc_ety[valid_mask] = hs_pc_ety[valid_mask] / hs_pc_ety_count[valid_mask]
        
        # Compute class-wise quantile conf mask
        if self.conf_perc is not None and self.conf_perc > 0.0:
            ety_conf_mask = refine_ety(
                torch.cat(corr_voxel_ety, dim=0), 
                torch.cat(corr_tr_logits, dim=0).argmax(1), self.conf_perc
            )
        else:
            ety_conf_mask = torch.ones(
                torch.cat(corr_tr_voxels, dim=0).shape[0], dtype=torch.bool).cuda()
        
        return {
            'ac_logits': corr_ac_logits,
            'ac_voxels': corr_ac_voxels,
            'tr_logits': corr_tr_logits,
            'tr_voxels': corr_tr_voxels,
            'tr_conf_mask': ety_conf_mask,
            'tr_ety_loss': ety_loss,
            'tr_pc2img_mask': hs_pc2img_idx,
            'tr_img_points': hs_img_points,
            # STVE output
            'ac_p2v_indices': ac_p2v_indices,
            'tr_p2v_indices': tr_p2v_indices,
            'hs_pc_ety': hs_pc_ety,
        }
        

class CR_Block(nn.Module):
    """
    Module to compute correspondence loss
    """
    def __init__(self, reduction="mean", conf_mask=False, ety_weight=False):
        super(CR_Block, self).__init__()
        self.reduction = reduction
        self.conf_mask = conf_mask
        self.ety_weight = ety_weight
    
    def forward(self, logit_dict: Dict[str, torch.Tensor], self_corr: bool = False) -> torch.Tensor:
        all_ac_logits, all_ac_voxels = logit_dict['ac_logits'], logit_dict['ac_voxels']
        all_tr_logits, all_tr_voxels = logit_dict['tr_logits'], logit_dict['tr_voxels']
        if self.ety_weight and not self_corr:
            ac_pc_ety, ac_p2v_indices = logit_dict['ac_pc_ety'], logit_dict['ac_p2v_indices']
            tr_pc_ety, tr_p2v_indices = logit_dict['tr_pc_ety'], logit_dict['tr_p2v_indices']
        
        loss = []
        left_idx = 0
        for i, (ac_logits, ac_voxels, tr_logits, tr_voxels) in \
            enumerate(zip(all_ac_logits, all_ac_voxels, all_tr_logits, all_tr_voxels)):
            # Raval Hash for voxel tensors, and re-order to find logits correspondence
            with torch.no_grad():
                hash_ac_voxels = ravel_hash_cuda(ac_voxels)
                hash_tr_voxels = ravel_hash_cuda(tr_voxels)
                ac_v_indices = torch.argsort(hash_ac_voxels)
                tr_v_indices = torch.argsort(hash_tr_voxels)
                del hash_ac_voxels, hash_tr_voxels
                torch.cuda.empty_cache()
            
            right_idx = left_idx + ac_logits.shape[0]
            ac_logits = ac_logits[ac_v_indices]
            tr_logits = tr_logits[tr_v_indices]
            if self.conf_mask:
                mask = logit_dict['tr_conf_mask'][left_idx:right_idx][tr_v_indices]
                # update ac & tr logits based on mask
                ac_logits = ac_logits[mask]
                tr_logits = tr_logits[mask]
            # ST-Ety weighting
            if self.ety_weight and not self_corr:
                ac_voxel_ety = ac_pc_ety[ac_p2v_indices[i]][ac_v_indices]
                tr_voxel_ety = tr_pc_ety[tr_p2v_indices[i]][tr_v_indices]
                weight = torch.exp(-tr_voxel_ety) / \
                    (torch.exp(-tr_voxel_ety) + torch.exp(-ac_voxel_ety))
                weight = weight[mask] if self.conf_mask else weight
            else:
                weight = torch.ones(ac_logits.shape[0]).cuda()

            left_idx = right_idx
            #! Viz
            # ac_preds = ac_logits.argmax(1).cpu().numpy()
            # color_platte = np.array(NUSCENES_LIDARSEG_COLOR_PALETTE[1:]) / 255.
            # ac_pc_color = color_platte[ac_preds]
            # tr_preds = tr_logits.argmax(1).cpu().numpy()
            # color_platte = np.array(NUSCENES_LIDARSEG_COLOR_PALETTE[1:]) / 255.
            # tr_pc_color = color_platte[tr_preds]
            # colored_point_viz(
            #     ac_voxels[ac_v_indices].cpu().numpy() * 0.2, 'latte/viz_samples/cr_ac_voxels.pcd', ac_pc_color)
            # colored_point_viz(
            #     tr_voxels[tr_v_indices].cpu().numpy() * 0.2, 'latte/viz_samples/cr_tr_voxels.pcd', tr_pc_color)
            # input("Press Enter to continue...")
            
            #! Histogram Viz
            # plt.figure()
            # plt.hist(F.softmax(ac_logits.detach(), dim=1).amax(1).cpu().numpy(), bins=50, color='red', alpha=0.5)
            # plt.hist(F.softmax(tr_logits.detach(), dim=1).amax(1).cpu().numpy(), bins=50, color='blue', alpha=0.5)
            # plt.savefig('latte/viz_samples/hist_acVStr.png')
            # plt.close()
            
            # Learning part
            # TODO: Introduce voxel-based attention module
            tr_logits[tr_logits == 0.0] = 1e-8
            ac_logits[ac_logits == 0.0] = 1e-8
            loss.append(
                (weight * F.kl_div(
                    torch.log(ac_logits), tr_logits.detach(),reduction='none'
                ).sum(1)).mean()
            )
            if torch.any(torch.isnan(loss[-1])):
                input("HrDC find NaN loss")
            # loss.append(
            #     torch.linalg.norm(
            #         F.softmax(ac_logits, dim=1) - F.softmax(tr_logits, dim=1).detach(), dim=1
            #         ).mean()
            # )
        
        return sum(loss) / len(loss)


def compute_voxel_ety(
    pc_ety: torch.Tensor,
    v_indices: torch.Tensor,
    v_inverse: torch.Tensor,
    pc_mask: torch.Tensor
):
    """Function to cover pc ety to masked voxel ety
    """
    new_v_ety = torch.zeros_like(pc_mask)
    #TODO: Complete voxel ety convertion
    pass
    

class Dense_2D(nn.Module):
    """
    Module to compute correspondence loss
    """
    def __init__(
        self, 
        reduction="mean", 
        exclude_labels: Tuple = None,
        conf_perc: float = None
    ):
        super(Dense_2D, self).__init__()
        self.reduction = reduction
        self.exclude_labels = torch.Tensor(exclude_labels) if exclude_labels else None
        self.conf_perc = conf_perc
    
    def forward(
        self, 
        all_logits_2d: torch.Tensor,    # N x C x H x W
        tr_logits_3d: torch.Tensor,     
        tr_img_points: List[torch.Tensor],
        tr_pc2img_idx: List[torch.Tensor],
        viz_img: np.array = None
    ):  
        dense_loss = []
        W = all_logits_2d.shape[3]
        
        for i in range(all_logits_2d.shape[0]):
            curr_img_points = tr_img_points[i].long()
            curr_pc2img_idx = tr_pc2img_idx[i].reshape(-1)
            
            # pc2img idexing 2D corresponding 3D logits
            pc2img_logits = tr_logits_3d[curr_pc2img_idx]
            
            with torch.no_grad():
                # scatter mean using hash table
                hash_img_pc = (curr_img_points[:, 0] * W + curr_img_points[:, 1])
                hash_img_pc = hash_img_pc.long().cuda()
                
                ds_logits_3d = ts_scatter(F.softmax(pc2img_logits, dim=1), hash_img_pc, dim=0, reduce=self.reduction)
                ds_img_points = torch.unique(curr_img_points, dim=0)
                ds_logits_3d = ds_logits_3d[torch.nonzero(ds_logits_3d.sum(1)).reshape(-1)]
                del hash_img_pc
            
            # pixel corresponding extraction
            img_logits = F.softmax(
                all_logits_2d.permute(0, 2, 3, 1)[i][ds_img_points[:, 0], ds_img_points[:, 1]],
                dim=1)
            
            with torch.no_grad():
                # conf perc filerting
                ds_entropy_2d = prob_2_entropy(img_logits).sum(1).detach()
                ds_entropy_3d = prob_2_entropy(ds_logits_3d).sum(1).detach()
                if self.conf_perc or self.conf_perc > 0.0:
                    thred_3d = torch.quantile(ds_entropy_3d, self.conf_perc)
                    ety_mask_3d = ds_entropy_3d < thred_3d
                    thred_2d = torch.quantile(ds_entropy_2d, self.conf_perc)
                    ety_mask_2d = ds_entropy_2d < thred_2d
                else:
                    ety_mask_3d = torch.ones(ds_logits_3d.shape[0], dtype=torch.bool).cuda().detach()
                    ety_mask_2d = torch.ones(ds_entropy_2d.shape[0], dtype=torch.bool).cuda().detach()
                rv_ety_2d = 1 / ds_entropy_2d + 1e-30
                rv_ety_3d = 1 / ds_entropy_3d + 1e-30
                weight_2d = rv_ety_2d / (rv_ety_2d + rv_ety_3d)
                weight_3d = rv_ety_3d / (rv_ety_2d + rv_ety_3d)
            
                # exclude label if needed
                if self.exclude_labels:
                    invalid_mask_3d = torch.any(ds_logits_3d.argmax(1)[:, None] == self.exclude_labels, dim=1)
                    invalid_mask_2d = torch.any(img_logits.argmax(1)[:, None] == self.exclude_labels, dim=1)
                    invalid_mask = torch.logical_and(invalid_mask_2d, invalid_mask_3d)
                    ety_conf_mask = torch.logical_and(ety_conf_mask, invalid_mask)
            
            #! Viz
            # from latte.data.utils.visualize import draw_points_image_labels
            # curr_viz_img = np.moveaxis(viz_img[i], 0, 2)
            # draw_points_image_labels(
            #     curr_viz_img, ds_img_points[ety_conf_mask.cpu()], img_logits.argmax(1).cpu().numpy() + 1, 
            #     show=False, color_palette_type="NuScenesLidarSeg", save="latte/viz_samples/ds_2d_preds.png"
            # )       # Dense 2D prediction
            # draw_points_image_labels(
            #     curr_viz_img, ds_img_points[ety_conf_mask.cpu()], ds_logits_3d.argmax(1).cpu().numpy() + 1, 
            #     show=False, color_palette_type="NuScenesLidarSeg", save="latte/viz_samples/ds_3d_preds.png"
            # )       # Dense 3D prediction
            # input("Press Enter to continue...")
            
            # compute batch-wise loss
            if torch.any(ds_logits_3d == 0.0):
                ds_logits_3d[ds_logits_3d == 0.0] = ds_logits_3d[ds_logits_3d == 0.0] + 1e-8
            if torch.any(img_logits == 0.0):
                img_logits[img_logits == 0.0] = img_logits[img_logits == 0.0] + 1e-8
            
            kl_23_loss = F.kl_div(
                torch.log(img_logits), ds_logits_3d.detach(), reduction='none'
            ).sum(1)
            kl_23_loss = (weight_3d * kl_23_loss)[ety_mask_3d]
            kl_32_loss = F.kl_div(
                torch.log(ds_logits_3d), img_logits.detach(), reduction='none'
            ).sum(1)
            kl_32_loss = (weight_2d * kl_32_loss)[ety_mask_2d]
            dense_loss.append(torch.mean(kl_23_loss) + torch.mean(kl_32_loss))
            # dense_loss.append(kl_23_loss.mean())
            
        return sum(dense_loss) / len(dense_loss)

 
if __name__ == "__main__":
    from latte.data.build import build_dataloader
    from latte.models.build import build_model_3d, build_model_2d
    from latte.common.config import purge_cfg
    from latte.data.collate import inverse_to_all
    from latte.common.utils.checkpoint import CheckpointerV2
    from latte.config.xmuda import cfg
    import logging
    import time
    import os.path as osp
    
    config_file = "configs/a2d2_semantic_kitti/hrdc_st.yaml"
    
    cfg.merge_from_file(config_file)
    purge_cfg(cfg)
    cfg.freeze()
    
    # Build dataloader
    train_dataloader_src = build_dataloader(cfg, mode='test', domain='target', start_iteration=0, tta=True)
  
    # Build models
    model_2d, train_metric_2d = build_model_2d(cfg)
    logging.info('Build 2D model:\n{}'.format(str(cfg.MODEL_2D.TYPE)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    logging.info('Parameters: {:.2e}'.format(num_params))
    model_3d, train_metric_3d = build_model_3d(cfg)
    logging.info('Build 3D model:\n{}'.format(str(cfg.MODEL_3D.TYPE)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    logging.info('Parameters: {:.2e}'.format(num_params))
    
    # build checkpointer
    model_prefix = "latte/exp/models/0214_a2d2_semantic_kitti_baseline_0/"
    checkpointer_2d = CheckpointerV2(model_2d, save_dir='', logger=None)
    weight_path = osp.join(model_prefix, "model_2d_100000.pth")
    checkpointer_2d.load(weight_path, resume=False)
    checkpointer_3d = CheckpointerV2(model_3d, save_dir='', logger=None)
    weight_path = osp.join(model_prefix, "model_3d_100000.pth")
    checkpointer_3d.load(weight_path, resume=False)
    
    # resize = cfg.DATASET_TARGET.get(cfg.DATASET_TARGET.TYPE)['resize']
    hr_module = HR_Block(resize=None).cuda()
    cr_module = CR_Block(conf_mask=True).cuda()
    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()
    model_2d.eval()
    model_3d.eval()
    
    # Input preprocessing
    train_iter_src = enumerate(train_dataloader_src)
    for iteration in range(len(train_dataloader_src)):
        _, data_batch_src = train_iter_src.__next__()
        data_batch_src['lidar'] = data_batch_src['lidar'].cuda()
        pc2img_idx_src = [idx.cuda() for idx in data_batch_src['pc2img_idx']]
        data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
        data_batch_src['img'] = data_batch_src['img'].cuda()
        # data_batch_src['proj_matrix'] = [mtx.cuda() for mtx in data_batch_src['proj_matrix']]
    
        # Model forward
        preds_2d = model_2d.inference(data_batch_src, return_all=True)
        preds_3d = model_3d(data_batch_src)
        preds_3d['seg_logit'] = inverse_to_all(preds_3d['seg_logit'], data_batch_src)
        preds_3d['seg_logit2'] = inverse_to_all(preds_3d['seg_logit2'], data_batch_src)        
        
        def print_grad(grad):
            print(grad)
        
        extra_key = ['pose', 'scene_name', 'scan_ID', 
                     'ori_points', 'img_indices', 'proj_matrix', 'ori_img_size']
        # for key in extra_key:
        preds_2d.update({key: data_batch_src[key] for key in extra_key})
        preds_3d.update({key: data_batch_src[key] for key in extra_key})
        ori_points_2d = []
        for i in range(len(data_batch_src['ori_points'])):
            curr_ori_points = data_batch_src['ori_points'][i]
            curr_pc2img_idx = data_batch_src['pc2img_idx'][i]
            ori_points_2d.append(curr_ori_points[curr_pc2img_idx])
        preds_2d.update({'ori_points': ori_points_2d})
        
        # Main debug
        # start_time = time.time()
        # cm_3d2d_loss = cr_module(hr_module(
        #     preds_3d, "3D", preds_2d, "2D"
        # ))
        # print("3D to 2D corresponding time: {}, Loss: {}".format(
        #     time.time() - start_time, cm_3d2d_loss))
        
        start_time = time.time()
        img_h, img_w = (1600, 900)
        hr_out = hr_module.forward(
            preds_2d, "2D", preds_3d, "3D", ds_3d2d=False, 
            viz_img=data_batch_src['img'].cpu().numpy()
        )
        cm_2d3d_loss = cr_module(hr_out)
        # ds_3d2d_loss = ds_3d2d(
        #     preds_2d['all_logits_2d'],
        #     preds_3d['seg_logit'],
        #     hr_out['tr_img_points'],
        #     hr_out['tr_pc2img_mask'],
        # )
        print("Dense 2D to 3D corresponding time: {}, Loss: {}".format(
            time.time() - start_time, cm_2d3d_loss))
        
        cm_2d3d_loss.backward()
        
        # start_time = time.time()
        # mm_loss = cr_module(hr_module(
        #     preds_3d, "3D", preds_3d, "3D"
        # ))
        # print("Intra-modal corresponding time: {}, Loss: {}".format(
        #     time.time() - start_time, mm_loss))
        
        # mm_loss.backward()
        
        # start_time = time.time()
        # (cm_3d2d_loss + cm_2d3d_loss + mm_loss).backward()
        # print("Backpropagate time: {}".format(
        #     time.time() - start_time))
        