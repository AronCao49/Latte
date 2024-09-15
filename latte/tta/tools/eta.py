#!/usr/bin/env python
import os

from latte.data.utils.evaluate import Evaluator
from latte.data.utils.refine_pseudo_labels import refine_pseudo_labels
# Limite cpu usage if needed
os.environ['OMP_NUM_THREADS'] = "4"

import argparse
import time
from typing import Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from latte.common.utils.metric_logger import MetricLogger, iou_to_excel
from latte.common.utils.loss import softmax_entropy
from latte.data.collate import inverse_to_all
from latte.data.utils.all_to_partial import pc2img_logit
from latte.data.utils.tt_validate import tt_validate

def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/a2d2_semantic_kitti/sar.yaml',
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


def update_model_probs(current_model_probs, new_probs):
    """
    Function to update probs of ETTA, from https://github.com/mr-eggplant/EATA/blob/main/eata.py
    """
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)


def eta(
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

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    setup_train()
    end = time.time()
    cur_probs_2d = None
    cur_probs_3d = None
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
            save_pth = save_pth
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
        # ETA forward
        # ---------------------------------------------------------------------------- #
        ety_2d = softmax_entropy(preds_2d['seg_logit'])
        ety_3d = softmax_entropy(preds_3d['seg_logit'][pc2img_mask])
        filter_1_2d = torch.where(ety_2d < cfg.TTA.ETA.e_margin)[0]
        filter_1_3d = torch.where(ety_3d < cfg.TTA.ETA.e_margin)[0]
        ids1_2d = filter_1_2d
        ids1_3d = filter_1_3d
        ids2_2d = torch.where(ids1_2d[0] > -0.1)[0]
        ids2_3d = torch.where(ids1_3d[0] > -0.1)[0]
        ety_2d, ety_3d = ety_2d[filter_1_2d], ety_3d[filter_1_3d]
        
        # ETA filtering redundant samples
        # 2D filtering
        if cur_probs_2d is not None:
            cos_sim_2d = F.cosine_similarity(cur_probs_2d.unsqueeze(dim=0), 
                                                preds_2d['seg_logit'][filter_1_2d].softmax(1), dim=1)
            filter_2_2d = torch.where(torch.abs(cos_sim_2d) < cfg.TTA.ETA.d_margin)[0]
            ety_2d = ety_2d[filter_2_2d]
            ids2_2d = filter_2_2d
            upd_probs_2d = update_model_probs(cur_probs_2d,
                                                preds_2d['seg_logit'][filter_1_2d][filter_2_2d].softmax(1))
        else:
            upd_probs_2d = update_model_probs(cur_probs_2d,
                                                preds_2d['seg_logit'][filter_1_2d].softmax(1))
        # 3D filtering
        if cur_probs_3d is not None:    
            cos_sim_3d = F.cosine_similarity(cur_probs_3d.unsqueeze(dim=0), 
                                                preds_3d['seg_logit'][pc2img_mask][filter_1_3d].softmax(1), dim=1)
            filter_2_3d = torch.where(torch.abs(cos_sim_3d) < cfg.TTA.ETA.d_margin)[0]
            ety_3d = ety_3d[filter_2_3d]
            ids2_3d = filter_2_3d
            upd_probs_3d = update_model_probs(cur_probs_3d,
                                                preds_3d['seg_logit'][pc2img_mask][filter_1_3d][filter_2_3d].softmax(1))
        else:
            upd_probs_3d = update_model_probs(cur_probs_3d,
                                                preds_3d['seg_logit'][pc2img_mask][filter_1_3d].softmax(1))
        
        # if filter_2_2d
        coeff_2d = 1 / (torch.exp(ety_2d.clone().detach() - math.log(len(class_names)) * 0.4))
        etta_loss_2d =  ety_2d.mul(coeff_2d).mean(0)
        coeff_3d = 1 / (torch.exp(ety_3d.clone().detach() - math.log(len(class_names)) * 0.4))
        etta_loss_3d =  ety_3d.mul(coeff_3d).mean(0)
        # updated cur_probs for 2D and 3D
        cur_probs_2d = upd_probs_2d
        cur_probs_3d = upd_probs_3d

        # cross modal learning
        if cfg.TTA.ETA.lambda_xm_trg > 0:
            # cross-modal loss: KL divergence
            xm_loss_trg_2d = F.kl_div(F.log_softmax(F.relu(preds_2d['seg_logit']), dim=1),
                                      F.softmax(F.relu(preds_3d['seg_logit'][pc2img_mask]).detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_trg_3d = F.kl_div(F.log_softmax(F.relu(preds_3d['seg_logit'][pc2img_mask]), dim=1),
                                      F.softmax(F.relu(preds_2d['seg_logit']).detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                       xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d.append(cfg.TTA.ETA.lambda_xm_trg * xm_loss_trg_2d)
            loss_3d.append(cfg.TTA.ETA.lambda_xm_trg * xm_loss_trg_3d)
        
        # check whether exist reliable samples
        # 2D
        if preds_2d['seg_logit'][ids1_2d][ids2_2d].shape[0] != 0:
            # train logger
            train_metric_logger.update(etta_loss_2d=etta_loss_2d)
            loss_2d.append(cfg.TTA.ETA.lambda_eta * etta_loss_2d)
            sum(loss_2d).backward()
            optimizer_2d.step()
            scheduler_2d.step()
        # 3D
        if preds_3d['seg_logit'][pc2img_mask][ids1_3d][ids2_3d].shape[0] != 0:
            # train logger
            train_metric_logger.update(etta_loss_3d=etta_loss_3d)
            loss_3d.append(cfg.TTA.ETA.lambda_eta * etta_loss_3d)
            sum(loss_3d).backward()
            optimizer_3d.step()
            scheduler_3d.step()

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time)
        del preds_2d, preds_3d, ety_2d, ety_3d
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
