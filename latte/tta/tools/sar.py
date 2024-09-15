#!/usr/bin/env python
from copy import deepcopy
import os
from latte.common.utils.sar_utils import SAM

from latte.data.utils.evaluate import Evaluator
from latte.data.utils.refine_pseudo_labels import refine_pseudo_labels
# Limite cpu usage if needed
os.environ['OMP_NUM_THREADS'] = "4"

import math
import argparse
import time
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
from latte.common.utils.loss import BerhuLoss, softmax_entropy
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
        default='configs/a2d2_semantic_kitti/sar_rs.yaml',
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
    args = parser.parse_args()
    return args


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'block4' in nm:
            continue
        if 'up4.1.0.net' in nm:
            continue
        if nm in ['norm']:
            continue

        if 'norm' in type(m).__name__.lower():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def configure_model(model):
    """Configure model for use with SAR."""
    # train mode, because SAR optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what SAR updates
    model.requires_grad_(False)
    # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if 'norm' in type(m).__name__.lower():
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


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


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


def sar(
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
    
    # Configure models, create ema models
    model_2d = configure_model(model_2d)
    model_3d = configure_model(model_3d)
    params_2d, _ = collect_params(model_2d)
    params_3d, _ = collect_params(model_3d)
    optimizer_2d = SAM(params_2d, torch.optim.SGD, lr=cfg.OPTIMIZER.MODEL_2D.BASE_LR, momentum=0.9)
    optimizer_3d = SAM(params_3d, torch.optim.SGD, lr=cfg.OPTIMIZER.MODEL_3D.BASE_LR, momentum=0.9)

    # Save state for reset
    init_model_2d, init_optim_2d = copy_model_and_optimizer(model_2d, optimizer_2d)
    init_model_3d, init_optim_3d = copy_model_and_optimizer(model_3d, optimizer_3d)


    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    start_iteration = 0
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
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # reset metric
        val_metric_logger.reset()

    setup_train()
    end = time.time()
    ema_2d, ema_3d = None, None
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
        loss_2d = []
        loss_3d = []
        preds_2d = model_2d.inference(data_batch_trg)
        preds_3d = model_3d(data_batch_trg)
        preds_3d['seg_logit'] = inverse_to_all(preds_3d['seg_logit'],
            data_batch_trg) if cfg.MODEL_3D.TYPE.upper() == "SPVCNN" else preds_3d['seg_logit']
        pc2img_mask, ori_logit_3d = pc2img_logit(preds_3d['seg_logit'], pc2img_idx_trg, return_mask=True)
        
        with torch.no_grad():
            setup_validate()
        
            # inference
            # record original prediction results
            val_pred_2d = {'seg_logit': preds_2d['seg_logit']}
            val_pred_3d = {'seg_logit': ori_logit_3d, 'all_seg_logit': preds_3d['seg_logit']}
            start_time_val = time.time()
            save_pth = None
            save_dict = None
            evaluator_dict = tt_validate(cfg,
                                        data_batch_trg,
                                        val_pred_2d,
                                        val_pred_3d,
                                        evaluator_dict,
                                        val_metric_logger,
                                        save_pth=save_pth,
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
        # First entropy backward
        # ---------------------------------------------------------------------------- #
        ety_2d = softmax_entropy(preds_2d['seg_logit'])
        ety_3d = softmax_entropy(preds_3d['seg_logit'])
        filter_ids_1_2d = torch.where(ety_2d < math.log(len(class_names)) * 0.4)[0]
        filter_ids_1_3d = torch.where(ety_3d < math.log(len(class_names)) * 0.4)[0]
        ety_2d, ety_3d = ety_2d[filter_ids_1_2d], ety_3d[filter_ids_1_3d]
        loss_1_2d, loss_1_3d = ety_2d.mean(0), ety_3d.mean(0)
        train_metric_logger.update(loss_1_2d=loss_1_2d, loss_1_3d=loss_1_3d)
        
        loss_1_2d.backward()
        loss_1_3d.backward()
        optimizer_2d.first_step(zero_grad=True)
        optimizer_3d.first_step(zero_grad=True)
        
        # ---------------------------------------------------------------------------- #
        # Second entropy backward
        # ---------------------------------------------------------------------------- #
        loss_2d, loss_3d = [], []
        preds_2d = model_2d.inference(data_batch_trg)
        preds_3d = model_3d(data_batch_trg)
        preds_3d['seg_logit'] = inverse_to_all(preds_3d['seg_logit'],
            data_batch_trg) if cfg.MODEL_3D.TYPE.upper() == "SPVCNN" else preds_3d['seg_logit']
        
        ety_2d_2 = softmax_entropy(preds_2d['seg_logit'])
        ety_3d_2 = softmax_entropy(preds_3d['seg_logit'])
        ety_2d_2, ety_3d_2 = ety_2d_2[filter_ids_1_2d], ety_3d_2[filter_ids_1_3d]
        loss_2_2d_v, loss_2_3d_v = ety_2d_2.clone().detach().mean(0), ety_3d_2.clone().detach().mean(0)
        filter_ids_2_2d = torch.where(ety_2d_2 < math.log(len(class_names)) * 0.4)[0]
        filter_ids_2_3d = torch.where(ety_3d_2 < math.log(len(class_names)) * 0.4)[0]
        loss_2_2d, loss_2_3d = ety_2d_2[filter_ids_2_2d].mean(0), ety_3d_2[filter_ids_2_3d].mean(0)
        loss_2d.append(loss_2_2d)
        loss_3d.append(loss_2_3d)

        # update ema
        if not np.isnan(loss_2_2d.item()):
            ema_2d = update_ema(ema_2d, loss_2_2d.item())
        if not np.isnan(loss_2_3d.item()):
            ema_3d = update_ema(ema_3d, loss_2_3d.item())

        
        # cross modal learning
        if cfg.TTA.SAR.lambda_xm_trg > 0:
            # cross-modal loss: KL divergence
            xm_loss_trg_2d = F.kl_div(F.log_softmax(F.relu(preds_2d['seg_logit']), dim=1),
                                      F.softmax(F.relu(preds_3d['seg_logit'][pc2img_mask]).detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_trg_3d = F.kl_div(F.log_softmax(F.relu(preds_3d['seg_logit'][pc2img_mask]), dim=1),
                                      F.softmax(F.relu(preds_2d['seg_logit']).detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                       xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d.append(cfg.TTA.SAR.lambda_xm_trg * xm_loss_trg_2d)
            loss_3d.append(cfg.TTA.SAR.lambda_xm_trg * xm_loss_trg_3d)

        train_metric_logger.update(loss_2_2d=loss_2_2d, loss_2_3d=loss_2_3d)
        
        (sum(loss_2d) + sum(loss_3d)).backward()
        optimizer_2d.second_step(zero_grad=True)
        optimizer_3d.second_step(zero_grad=True)

        # reset
        # ema_2d, ema_3d = 0.1, 0.1
        if ema_2d is not None and ema_2d < 0.2 and cfg.TTA.SAR.reset:
            # logger.info("2D ema < 0.2, reset")
            load_model_and_optimizer(model_2d, optimizer_2d, init_model_2d, init_optim_2d)
        if ema_3d is not None and ema_3d < 0.2 and cfg.TTA.SAR.reset:
            # logger.info("3D ema < 0.2, reset")
            load_model_and_optimizer(model_3d, optimizer_3d, init_model_3d, init_optim_3d)
        
        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time)
        del preds_2d, preds_3d, ety_2d, ety_2d_2, ety_3d, ety_3d_2, \
            filter_ids_1_2d, filter_ids_1_3d, filter_ids_2_2d, filter_ids_2_3d
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