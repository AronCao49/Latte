import os
import numpy as np
import logging
import time
from copy import deepcopy

import torch
import torch.nn.functional as F
import open3d as o3d
from latte.data.utils.all_to_partial import pc2img_logit

from latte.data.utils.evaluate import Evaluator
from latte.data.utils.visualize import draw_points_image_labels
from latte.models.losses import entropy_loss, prob_2_entropy
from latte.data.collate import inverse_to_all

def cross_modal_lifting(preds_2d, img_indices):
    img_feats = []
    for i in range(preds_2d.shape[0]):
        img_feats.append(preds_2d[i][img_indices[i][:, 0], img_indices[i][:, 1]])
    img_feats = torch.cat(img_feats, 0)

    return img_feats

def covariance(features):
    assert len(features.shape) == 2
    n = features.shape[0]
    tmp = np.ones((1, n)) @ features
    cov = (features.T @ features - (tmp.T @ tmp) / n) / n
    return cov

def tt_validate(cfg,
             data_batch,
             preds_2d,
             preds_3d,
             evaluator_dict,
             val_metric_logger,
             entropy_fuse=True,
             ps_label_xm=None,
             xm_label_mask: torch.Tensor = None,
             save_pth=None,
             save_dict=None,
             ):

    with torch.no_grad():
        # mapping back to full label for SPVCNN
        all_pred_logit_voxel_3d = preds_3d['all_seg_logit']
        pred_logit_voxel_3d = preds_3d['seg_logit']
        pred_label_voxel_3d = pred_logit_voxel_3d.argmax(1).cpu().numpy()
        pred_label_3d_all = all_pred_logit_voxel_3d.argmax(1).cpu().numpy()
        pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy()
        
        # For partial FOV
        # data_batch['pc2img_idx'] = [idx.cuda() for idx in data_batch['pc2img_idx']]
        
        # softmax average (ensembling)
        probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
        probs_3d = F.softmax(pred_logit_voxel_3d, dim=1)
        pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu().numpy()
        # ProbEn ensembling
        if cfg.VAL.prob_en:
            sum_logits = torch.log(probs_2d) + torch.log(probs_3d)
            exp_logits = torch.exp(sum_logits)
            pred_prob_en = exp_logits / torch.sum(exp_logits, dim=1).unsqueeze(1)
            pred_label_voxel_en = pred_prob_en.argmax(1).cpu().numpy()
        if ps_label_xm is not None:
            pred_label_sd = ps_label_xm.cpu().numpy()
            if xm_label_mask is not None:
                xm_label_mask = xm_label_mask.cpu().numpy()
                pred_label_sd[xm_label_mask] = pred_label_voxel_ensemble[xm_label_mask]

        val_2d_ety = prob_2_entropy(probs_2d)
        val_3d_ety = prob_2_entropy(probs_3d)
        val_metric_logger.update(val_2d_ety=torch.mean(val_2d_ety))
        val_metric_logger.update(val_3d_ety=torch.mean(val_3d_ety))
        if entropy_fuse:
            # rv_ety_2d = 1 / (prob_2_entropy(probs_2d) + 1e-30)
            # rv_ety_3d = 1 / (prob_2_entropy(probs_3d) + 1e-30)
            rv_ety_2d = torch.exp(-prob_2_entropy(probs_2d).sum(1))
            rv_ety_3d = torch.exp(-prob_2_entropy(probs_3d).sum(1))
            weight_2d = rv_ety_2d / (rv_ety_2d + rv_ety_3d)
            weight_3d = rv_ety_3d / (rv_ety_2d + rv_ety_3d)
            pred_label_ety_ensemble = (weight_2d.unsqueeze(1) * probs_2d + weight_3d.unsqueeze(1) * probs_3d).argmax(1).cpu().numpy()
            # print(pred_label_ensemble.shape, pred_label_ety_ensemble.shape)

        # get original point cloud from before voxelization
        # print(data_batch.keys())
        seg_label = data_batch['orig_seg_label']
        points_idx = data_batch['orig_points_idx']
        # loop over batch
        left_idx = 0
        for batch_ind in range(len(seg_label)):
            curr_points_idx = points_idx[batch_ind]
            # check if all points have predictions (= all voxels inside receptive field)
            assert np.all(curr_points_idx)

            curr_seg_label = seg_label[batch_ind]
            curr_pc2img_idx = data_batch['pc2img_idx'][batch_ind].cpu().numpy()
            curr_seg_label = curr_seg_label[curr_pc2img_idx]
            right_idx = left_idx + curr_points_idx.sum()
            pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
            pred_label_3d = pred_label_voxel_3d[left_idx:right_idx]
            pred_label_ensemble = pred_label_voxel_ensemble[left_idx:right_idx]
            pred_label_e_ensemble = pred_label_ety_ensemble[left_idx:right_idx]

            # print(pred_label_ensemble.shape, pred_label_ety_ensemble.shape)

            # evaluate
            evaluator_dict["2D"].update(pred_label_2d, curr_seg_label)
            evaluator_dict["3D"].update(pred_label_3d, curr_seg_label)
            if ps_label_xm is not None:
                evaluator_dict["2D+3D"].update(
                    pred_label_sd[left_idx:right_idx], curr_seg_label)
            else:
                evaluator_dict["2D+3D"].update(
                    pred_label_ensemble, curr_seg_label)
            
            if save_pth is not None:
                if not os.path.exists(save_pth):
                    os.makedirs(save_pth, exist_ok=True)
                curr_image = data_batch['img'][batch_ind].detach().cpu().numpy()
                curr_image = np.moveaxis(curr_image, 0, -1)
                dataset_cfg = cfg.get('DATASET_TARGET')
                color_palatte_type = 'Synthia'
                curr_seq, curr_id = data_batch['scene_name'][batch_ind], data_batch['scan_ID'][batch_ind]
                draw_points_image_labels(
                    curr_image, data_batch['img_indices'][batch_ind], pred_label_2d,
                    color_palette_type=color_palatte_type, 
                    save=os.path.join(save_pth, "{}_{}_{}.png".format(curr_seq, curr_id, "2D")))
                draw_points_image_labels(
                    curr_image, data_batch['img_indices'][batch_ind], pred_label_3d,
                    color_palette_type=color_palatte_type, 
                    save=os.path.join(save_pth, "{}_{}_{}.png".format(curr_seq, curr_id, "3D")))
                if ps_label_xm is not None:
                    pred_label_sd_curr = pred_label_sd[left_idx:right_idx]
                    draw_points_image_labels(
                        curr_image, data_batch['img_indices'][batch_ind], pred_label_sd_curr,
                        color_palette_type=color_palatte_type, 
                        save=os.path.join(save_pth, "{}_{}_{}.png".format(curr_seq, curr_id, "xM")))
                else:   
                    draw_points_image_labels(
                        curr_image, data_batch['img_indices'][batch_ind], pred_label_ensemble,
                        color_palette_type=color_palatte_type, 
                        save=os.path.join(save_pth, "{}_{}_{}.png".format(curr_seq, curr_id, "xM")))
                draw_points_image_labels(
                    curr_image, data_batch['img_indices'][batch_ind], curr_seg_label,
                    color_palette_type=color_palatte_type, 
                    save=os.path.join(save_pth, "{}_{}_{}.png".format(curr_seq, curr_id, "GT")))   
            
            
            if "ety fuse" in evaluator_dict.keys():
                evaluator_dict["ety fuse"].update(pred_label_e_ensemble, curr_seg_label)
            left_idx = right_idx
            
        if "3D all" in evaluator_dict.keys():
            all_seg_label = deepcopy(data_batch['seg_label'].numpy())
            evaluator_dict["3D all"].update(pred_label_3d_all, all_seg_label)

        seg_loss_2d = F.cross_entropy(
            preds_2d['seg_logit'].cpu(), 
            pc2img_logit(data_batch['seg_label'], data_batch['pc2img_idx'])
        )
        seg_loss_3d = F.cross_entropy(all_pred_logit_voxel_3d.cpu(), data_batch['seg_label'])
        val_metric_logger.update(seg_loss_2d=seg_loss_2d)
        if seg_loss_3d is not None:
            val_metric_logger.update(seg_loss_3d=seg_loss_3d)
            
        val_metric_logger.update(iou_2d=evaluator_dict["2D"].overall_iou)
        val_metric_logger.update(iou_3d=evaluator_dict["3D"].overall_iou)
        # val_metric_logger.update(iou_3d_all=evaluator_dict["3D all"].overall_iou)      
        
    return evaluator_dict