#!/usr/bin/env python
import os

from latte.data.utils.evaluate import Evaluator
from latte.data.utils.refine_pseudo_labels import refine_pseudo_labels
# Limite cpu usage if needed
os.environ['OMP_NUM_THREADS'] = "4"

import os.path as osp
import argparse
import logging
import time
import socket
import warnings
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

from latte.common.utils.metric_logger import MetricLogger, iou_to_excel
from latte.common.utils.ema_util import create_ema_model, update_ema_all, update_ema_norm
from latte.data.collate import inverse_to_all
from latte.models.build import build_model_ds2d, build_model_hrcr
from latte.data.utils.all_to_partial import pc2img_logit
from latte.models.losses import prob_2_entropy
from latte.data.utils.tt_validate import tt_validate

def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/a2d2_semantic_kitti/hrdc_st.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--task',
        help='Specific task name',
        default='SS2MM',
        type=str,
        )
    parser.add_argument(
        '--save_pth',
        help='Path to save visualization',
        default=None,
        type=str,
        )
    args = parser.parse_args()
    return args


def latte(
    cfg, 
    logger,
    # model related args 
    model_2d: torch.nn.Module,
    model_3d: torch.nn.Module,
    optimizer_2d: torch.optim.Optimizer,
    optimizer_3d: torch.optim.Optimizer,
    scheduler_2d: torch.optim.lr_scheduler,
    scheduler_3d: torch.optim.lr_scheduler,
    train_dataloader_trg: DataLoader,
    checkpoint_data_2d, checkpoint_data_3d,
    checkpointer_2d, checkpointer_3d,
    summary_writer: SummaryWriter,
    # save results
    save_pth=None
) -> Dict[str, Evaluator]:
    
    # Retrieve some training period specs
    start_iteration = 0
    val_period = cfg.VAL.PERIOD
    max_iteration = len(train_dataloader_trg)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD
    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d': None,
        '3d': None
    }
    best_metric_iter = {'2d': -1, '3d': -1}
    # add metrics
    train_metric_logger = MetricLogger(delimiter='  ')
    val_metric_logger = MetricLogger(delimiter='  ')
    
    # build HR & CR module
    if (cfg.TTA.LATTE.lambda_hr_23_trg > 0 or cfg.TTA.LATTE.lambda_hr_32_trg > 0):
        hr_module, cr_module = build_model_hrcr(cfg)
        hr_module, cr_module = hr_module.cuda(), cr_module.cuda()
    
    class_names = train_dataloader_trg.dataset.class_names
    evaluator_2d = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if model_3d else None
    evaluator_dict = {
        "2D": evaluator_2d,
        "3D": evaluator_3d,
        "2D+3D": evaluator_ensemble,
        }

    # Teacher models configuration    
    ema_model_2d, log_modules_2d = create_ema_model(model_2d)
    ema_model_3d, log_modules_3d = create_ema_model(model_3d)
    logger.info("[Teacher Model 2D] - Config: Reset Dropout layer: {}".format(log_modules_2d[0]))
    logger.info("[Teacher Model 3D] - Config: Reset Dropout layer: {}".format(log_modules_3d[0]))

    def setup_train():
        train_metric_logger.reset()

    def setup_validate():
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    logger.info('Start training from iteration {}'.format(start_iteration))
    setup_train()
    end = time.time()
    for iteration, data_batch_trg in enumerate(train_dataloader_trg):
        # fetch data_batches for source & target
        # copy data from cpu to gpu
        if 'SCN' in cfg.MODEL_3D.TYPE:
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
        elif 'SPVCNN' in cfg.MODEL_3D.TYPE:
            data_batch_trg['lidar'] = data_batch_trg['lidar'].cuda()
        elif 'SalsaNext' in cfg.MODEL_3D.TYPE:
            data_batch_trg['proj_in'] = data_batch_trg['proj_in'].float().cuda()
        else:
            raise NotImplementedError('The requested network {} is not supported for now.'.format(cfg.MODEL_3D.TYPE))
        # For partial FOV
        if 'pc2img_idx' in data_batch_trg.keys():
            pc2img_idx_trg = [idx.cuda() for idx in data_batch_trg['pc2img_idx']]
        # copy seg_label & image
        data_batch_trg['img'] = data_batch_trg['img'].cuda()

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()

        # ---------------------------------------------------------------------------- #
        # One-pass inference
        # ---------------------------------------------------------------------------- #
        with torch.no_grad():
            setup_validate()
            ema_preds_2d = ema_model_2d.inference(data_batch_trg)
            all_ema_preds_3d = ema_model_3d(data_batch_trg)
            ema_logit_2d = ema_preds_2d['seg_logit']
            all_ema_preds_3d['seg_logit'] = inverse_to_all(all_ema_preds_3d['seg_logit'],
                                        data_batch_trg) if cfg.MODEL_3D.TYPE.upper() == "SPVCNN" \
                        else all_ema_preds_3d['seg_logit']
            pc2img_mask, ema_logit_3d = pc2img_logit(all_ema_preds_3d['seg_logit'], pc2img_idx_trg, return_mask=True)

            if torch.any(torch.isnan(F.softmax(ema_logit_3d, dim=1))):
                input("Find Nan, set pointer to continues")
        
        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #
        loss_2d = []
        loss_3d = []
        preds_2d = model_2d.inference(
            data_batch_trg, requires_grad=True, return_all=False)
        preds_3d = model_3d(data_batch_trg)
        preds_3d['seg_logit'] = inverse_to_all(preds_3d['seg_logit'],
            data_batch_trg) if cfg.MODEL_3D.TYPE.upper() == "SPVCNN" else preds_3d['seg_logit']     # 14625 MB
        
        if (cfg.TTA.LATTE.lambda_hr_23_trg > 0 or cfg.TTA.LATTE.lambda_hr_32_trg > 0 or cfg.TTA.LATTE.lambda_voxel_ety > 0) \
            and iteration >= cfg.TTA.LATTE.start_iter:
            extra_key = ['pose', 'scene_name', 'scan_ID', 'ori_points', 
                         'img_indices', 'ori_img_size', 'proj_matrix', 'pc2img_idx']
            # anchor 2D/3D logits
            preds_2d.update({key: data_batch_trg[key] for key in extra_key})
            preds_3d.update({key: data_batch_trg[key] for key in extra_key})
            # hist 2D/3D logits
            ema_preds_2d.update({key: data_batch_trg[key] for key in extra_key})
            all_ema_preds_3d.update({key: data_batch_trg[key] for key in extra_key})
            ori_points_2d = []
            for i in range(len(data_batch_trg['ori_points'])):
                curr_ori_points = data_batch_trg['ori_points'][i]
                curr_pc2img_idx = data_batch_trg['pc2img_idx'][i]
                ori_points_2d.append(curr_ori_points[curr_pc2img_idx])
            preds_2d.update({'ori_points': ori_points_2d})
            ema_preds_2d.update({'ori_points': ori_points_2d})
            
            # Spatial Temporal Voxelization + Slide windows
            hr_23out_dict = hr_module(
                    preds_2d, "2D", all_ema_preds_3d, "3D")
            hr_32out_dict = hr_module(
                    preds_3d, "3D", ema_preds_2d, "2D")
            
            # ---------------------------------------------------------------------------- #
            # Pseudo-label generation (soft version)
            # ---------------------------------------------------------------------------- #
            probs_2d = F.softmax(ema_logit_2d, dim=1)
            all_probs_3d = F.softmax(all_ema_preds_3d['seg_logit'], dim=1)
            probs_3d = F.softmax(ema_logit_3d, dim=1)
            st_ety_3d, st_ety_2d = hr_23out_dict['hs_pc_ety'], hr_32out_dict['hs_pc_ety']
            valid_mask_3d, valid_mask_2d = st_ety_3d != 0, st_ety_2d != 0
            conf_perc = cfg.HR_BLOCK.conf_perc
            if conf_perc is not None and conf_perc >= 0:
                valid_mask_3d = torch.logical_and(
                    st_ety_3d <= torch.quantile(st_ety_3d, conf_perc), valid_mask_3d
                )
                valid_mask_2d = torch.logical_and(
                    st_ety_2d <= torch.quantile(st_ety_2d, conf_perc), valid_mask_2d
                )
            
            # update pc st-ety
            pc_2d_ety = prob_2_entropy(probs_2d).sum(1)
            pc_3d_ety = prob_2_entropy(all_probs_3d).sum(1)
            pc_2d_ety[valid_mask_2d] = st_ety_2d[valid_mask_2d]
            pc_3d_ety[valid_mask_3d] = st_ety_3d[valid_mask_3d]
            
            rv_ety_2d = torch.exp(-pc_2d_ety)
            rv_ety_3d = torch.exp(-pc_3d_ety)
            rv_ety_2d[valid_mask_2d] = torch.exp(-st_ety_2d[valid_mask_2d])
            rv_ety_3d[valid_mask_3d] = torch.exp(-st_ety_3d[valid_mask_3d])
            # replace point ety with ST-Voxel ety
            weight_2d = rv_ety_2d / (rv_ety_2d + rv_ety_3d[pc2img_mask])
            weight_3d = rv_ety_3d[pc2img_mask] / (rv_ety_2d + rv_ety_3d[pc2img_mask])
            
            ps_label_prob = all_probs_3d
            ps_label_prob[pc2img_mask, :] = weight_2d.unsqueeze(1) * probs_2d + weight_3d.unsqueeze(1) * probs_3d
            ps_label_xm = ps_label_prob.argmax(1)
            ps_label_xm = refine_pseudo_labels(ps_label_prob.amax(1), ps_label_xm)
            
            # ---------------------------------------------------------------------------- #
            # Eval on target
            # ---------------------------------------------------------------------------- #
            # record original prediction results
            val_pred_2d = {'seg_logit': ema_logit_2d}
            val_pred_3d = {'seg_logit': ema_logit_3d, 'all_seg_logit': all_ema_preds_3d['seg_logit']}
            start_time_val = time.time()
            save_dict = None
            evaluator_dict = tt_validate(cfg,
                                        data_batch_trg,
                                        val_pred_2d,
                                        val_pred_3d,
                                        evaluator_dict,
                                        val_metric_logger,
                                        ps_label_xm=ps_label_prob[pc2img_mask, :].argmax(1),
                                        save_pth=save_pth if iteration >= 3000 else None,
                                        save_dict=save_dict)
            epoch_time_val = time.time() - start_time_val
            
            # ---------------------------------------------------------------------------- #
            # log evaluation performance
            # ---------------------------------------------------------------------------- #
            if val_period > 0 and iteration % val_period == 0:
                logger.info('Iteration[{}]-Test {}  total_time: {:.2f}s'.format(
                    iteration, val_metric_logger.summary_str, epoch_time_val))

                # summary
                if summary_writer is not None:
                    keywords = ('loss', 'acc', 'iou')
                    for name, meter in val_metric_logger.meters.items():
                        if all(k not in name for k in keywords):
                            continue
                        summary_writer.add_scalar('val/' + name, meter.avg, global_step=iteration)
            
            # Loss computation
            # 2D to 3D
            if cfg.TTA.LATTE.lambda_hr_23_trg > 0:
                # update entropy
                hr_23out_dict.update({"ac_pc_ety": pc_2d_ety, "tr_pc_ety": pc_3d_ety })
                # forward
                hr_23_loss = cr_module(hr_23out_dict)
                loss_2d.append(cfg.TTA.LATTE.lambda_hr_23_trg * hr_23_loss)
                train_metric_logger.update(trg_hr_23_loss=hr_23_loss)
            
            # 3D to 2D
            if cfg.TTA.LATTE.lambda_hr_32_trg > 0:
                # update entropy
                hr_32out_dict.update({"ac_pc_ety": pc_3d_ety, "tr_pc_ety": pc_2d_ety })
                # forward
                hr_32_loss = cr_module(hr_32out_dict)
                loss_3d.append(cfg.TTA.LATTE.lambda_hr_32_trg * hr_32_loss)
                train_metric_logger.update(trg_hr_32_loss=hr_32_loss)

        if cfg.TTA.LATTE.lambda_ety_ps > 0:
            # entropy-based ps-label
            pl_loss_trg_2d = F.cross_entropy(preds_2d['seg_logit'], ps_label_xm[pc2img_mask].long().detach(), weight=class_weights)
            pl_loss_trg_3d = F.cross_entropy(preds_3d['seg_logit'], ps_label_xm.long().detach(), weight=class_weights)
        else:
            ps_label_prob = all_probs_3d
            ps_label_prob[pc2img_mask] = (probs_2d + probs_3d) / 2
            ps_label_xm = ps_label_prob.argmax(1)
            ps_label_xm = refine_pseudo_labels(ps_label_prob.amax(1), ps_label_xm)
            pl_loss_trg_2d = F.cross_entropy(preds_2d['seg_logit'], ps_label_xm[pc2img_mask].long().detach(), weight=class_weights)
            pl_loss_trg_3d = F.cross_entropy(preds_3d['seg_logit'], ps_label_xm.long().detach(), weight=class_weights)
        
        loss_2d.append(pl_loss_trg_2d)
        loss_3d.append(pl_loss_trg_3d)
        train_metric_logger.update(
            pl_loss_trg_2d=pl_loss_trg_2d,
            pl_loss_trg_3d=pl_loss_trg_3d
            )
        
        # backward
        (sum(loss_3d) + sum(loss_2d)).backward()
        
        del hr_32out_dict, hr_23out_dict, loss_3d, loss_2d
        
        optimizer_2d.step()
        optimizer_3d.step()

        # update ema_model using student network
        if cfg.TRAIN.UPDATE_PARAMS.lower() == "norm":
            ema_model_2d = update_ema_norm(ema_model_2d, model_2d, 0.99)
            ema_model_3d = update_ema_norm(ema_model_3d, model_3d, 0.99)
        elif cfg.TRAIN.UPDATE_PARAMS.lower() == "all":
            ema_model_2d = update_ema_all(ema_model_2d, model_2d, 0.99)
            ema_model_3d = update_ema_all(ema_model_3d, model_3d, 0.99)
        else:
            raise IndexError("Required update params has not been defined yet: {}".format(cfg.TRAIN.UPDATE_PARAMS))

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time)
        torch.cuda.empty_cache()

        # log
        if iteration == 0 or (cfg.TRAIN.LOG_PERIOD > 0 and iteration % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                    ]
                ).format(
                    iter=iteration,
                    meters=str(train_metric_logger),
                    lr=optimizer_2d.param_groups[0]['lr'],
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and iteration % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou', 'ety')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=iteration)

        # checkpoint
        if (ckpt_period > 0 and iteration % ckpt_period == 0) or (iteration + 1 >= max_iteration):
            checkpoint_data_2d['iteration'] = iteration
            checkpoint_data_2d[best_metric_name] = best_metric['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(iteration), **checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = iteration
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(iteration), **checkpoint_data_3d)

        scheduler_2d.step()
        scheduler_3d.step()
        end = time.time()

        if iteration > 0 and iteration % cfg.TRAIN.TTA_EVAL_PERIOD == 0:
            logger.info('-'*65)
            logger.info('Evaluation Results')
            logger.info('-'*65)
            for modality, evaluator in evaluator_dict.items():
                logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
                logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
                logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))
    
    return evaluator_dict
