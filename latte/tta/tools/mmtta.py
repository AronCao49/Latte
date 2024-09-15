#!/usr/bin/env python
import os

from latte.data.utils.evaluate import Evaluator
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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from latte.common.solver.build import build_optimizer, build_scheduler
from latte.common.utils.checkpoint import CheckpointerV2
from latte.common.utils.logger import get_logger, setup_logger
from latte.common.utils.metric_logger import MetricLogger, iou_to_excel
from latte.common.utils.torch_util import set_random_seed
from latte.common.utils.loss import BerhuLoss
from latte.common.utils.ema_util import configure_model_norm, create_ema_model, update_ema_norm, configure_model_all, update_ema_all
from latte.data.collate import inverse_to_all
from latte.models.build import build_model_2d, build_model_3d, build_model_hrcr
from latte.data.build import build_dataloader
from latte.data.utils.validate import validate
from latte.data.utils.all_to_partial import pc2img_logit
from latte.models.losses import entropy_loss
from latte.data.utils.tt_validate import tt_validate

def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/a2d2_semantic_kitti/mmtta.yaml',
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


def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger


def mmtta(
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
    
    # Teacher models configuration
    ema_model_2d, log_modules_2d = create_ema_model(model_2d)
    ema_model_3d, log_modules_3d = create_ema_model(model_3d)
    logger.info("[Teacher Model 2D] - Config: Reset Dropout layer: {}".format(log_modules_2d[0]))
    logger.info("[Teacher Model 3D] - Config: Reset Dropout layer: {}".format(log_modules_3d[0]))

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = MetricLogger(delimiter='  ')
    val_metric_logger = MetricLogger(delimiter='  ')
    # create evaluators
    class_names = train_dataloader_trg.dataset.class_names
    evaluator_2d = Evaluator(class_names)
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if model_3d else None
    evaluator_dict = {
        "2D": evaluator_2d,
        "3D": evaluator_3d,
        "2D+3D": evaluator_ensemble,
        }

    def setup_train():
        train_metric_logger.reset()

    def setup_validate():
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    setup_train()
    end = time.time()
    for iteration, data_batch_trg in enumerate(train_dataloader_trg):
        # fetch data_batches for source & target
        # copy data from cpu to gpu
        if 'SPVCNN' in cfg.MODEL_3D.TYPE:
            data_batch_trg['lidar'] = data_batch_trg['lidar'].cuda()
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
            all_ema_logit_3d = inverse_to_all(all_ema_preds_3d['seg_logit'],
                                        data_batch_trg) if cfg.MODEL_3D.TYPE.upper() == "SPVCNN" \
                        else all_ema_preds_3d['seg_logit']
            ema_logit_3d = pc2img_logit(all_ema_logit_3d, pc2img_idx_trg)
        
            # inference
            # record original prediction results
            val_pred_2d = {'seg_logit': ema_logit_2d}
            val_pred_3d = {'seg_logit': ema_logit_3d, 'all_seg_logit': all_ema_logit_3d}
            start_time_val = time.time()
            save_dict = None
            evaluator_dict = tt_validate(cfg,
                                        data_batch_trg,
                                        val_pred_2d,
                                        val_pred_3d,
                                        evaluator_dict,
                                        val_metric_logger,
                                        save_pth=save_pth if iteration >= 500 else None,
                                        # save_pth=save_pth,
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

            # best validation
            for modality in ['2d', '3d']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = iteration
        
        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #
        loss_2d = []
        loss_3d = []
        preds_2d = model_2d.inference(data_batch_trg)
        preds_3d = model_3d(data_batch_trg)
        ori_logit_2d = preds_2d['seg_logit']
        all_ori_logit_3d = inverse_to_all(preds_3d['seg_logit'],
            data_batch_trg) if cfg.MODEL_3D.TYPE.upper() == "SPVCNN" else preds_3d['seg_logit']
        ori_logit_3d = pc2img_logit(all_ori_logit_3d, pc2img_idx_trg)
        # ---------------------------------------------------------------------------- #
        # Pseudo-label generation (soft version)
        # ---------------------------------------------------------------------------- #
        # Intra-PG
        seg_logit_2d = (ori_logit_2d + ema_logit_2d) / 2
        seg_logit_3d = (ori_logit_3d + ema_logit_3d) / 2
        # Inter-PR
        weight_2d = (1/(F.kl_div(F.log_softmax(ori_logit_2d, dim=1), F.softmax(ema_logit_2d, dim=1), reduction='none').sum(1)) + \
                        1/(F.kl_div(F.log_softmax(ema_logit_2d, dim=1), F.softmax(ori_logit_2d, dim=1), reduction='none').sum(1))) / 2
        weight_3d = (1/(F.kl_div(F.log_softmax(ori_logit_3d, dim=1), F.softmax(ema_logit_3d, dim=1), reduction='none').sum(1)) + \
                        1/(F.kl_div(F.log_softmax(ema_logit_3d, dim=1), F.softmax(ori_logit_3d, dim=1), reduction='none').sum(1))) / 2
        # ps-label filtering
        mask_ics = torch.cat((weight_2d.unsqueeze(1), weight_3d.unsqueeze(1)), dim=1).max(dim=1)[0] < 0.3
        weight_2d = weight_2d / (weight_2d + weight_3d)
        weight_3d = weight_3d / (weight_2d + weight_3d)
        ps_label_xm = (weight_2d.unsqueeze(1) * seg_logit_2d + weight_3d.unsqueeze(1) * seg_logit_3d).argmax(1)
        ps_label_xm[mask_ics] = -100

        # loss function part
        pl_loss_trg_2d = F.cross_entropy(ori_logit_2d, ps_label_xm.long().detach(), weight=class_weights)
        pl_loss_trg_3d = F.cross_entropy(ori_logit_3d, ps_label_xm.long().detach(), weight=class_weights)
        train_metric_logger.update(
            pl_loss_trg_2d=pl_loss_trg_2d,
            pl_loss_trg_3d=pl_loss_trg_3d
            )
        train_metric_logger.update(
            pl_loss_trg_2d=pl_loss_trg_2d,
            pl_loss_trg_3d=pl_loss_trg_3d
            )
        loss_2d.append(pl_loss_trg_2d)
        loss_3d.append(pl_loss_trg_3d)

        # back
        (sum(loss_3d) + sum(loss_2d)).backward()
        optimizer_2d.step()
        optimizer_3d.step()
        scheduler_2d.step()
        scheduler_3d.step()

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
    

