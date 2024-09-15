#!/usr/bin/env python
from copy import deepcopy
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

import torch
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

from latte.common.solver.build import build_optimizer, build_scheduler
from latte.common.utils.checkpoint import CheckpointerV2
from latte.common.utils.logger import get_logger
from latte.common.utils.metric_logger import MetricLogger, iou_to_excel
from latte.common.utils.torch_util import set_random_seed
from latte.common.utils.ema_util import configure_model_all, configure_model_norm
from latte.models.build import build_model_2d, build_model_3d
from latte.data.build import build_dataloader

# TTA methods
from latte.tta.tools import eta, mmtta, pslabel, sar, tent, latte, bn_gt, xmuda

def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/nuscenes/usa_singapore/eta.yaml',
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
        default='TTA',
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


def train(cfg, logger, output_dir='', run_name='', save_pth=None, tta_type: str = None):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    # logger = logging.getLogger('xmuda.train')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d, _ = build_model_2d(cfg)
    logger.info('Build 2D model:\n{}'.format(str(cfg.MODEL_2D.TYPE)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    logger.info('Parameters: {:.2e}'.format(num_params))

    # build 3d model
    model_3d, _ = build_model_3d(cfg)
    logger.info('Build 3D model:\n{}'.format(str(cfg.MODEL_3D.TYPE)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    logger.info('Parameters: {:.2e}'.format(num_params))

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()

    # build optimizer
    optim_2d_cfg = cfg.get('OPTIMIZER')['MODEL_2D']
    optim_3d_cfg = cfg.get('OPTIMIZER')['MODEL_3D']
    optimizer_2d = build_optimizer(optim_2d_cfg, model_2d)
    optimizer_3d = build_optimizer(optim_3d_cfg, model_3d)

    # build lr scheduler
    scheduler_2d_cfg = cfg.get('SCHEDULER')['MODEL_2D']
    scheduler_3d_cfg = cfg.get('SCHEDULER')['MODEL_3D']
    scheduler_2d = build_scheduler(scheduler_2d_cfg, optimizer_2d)
    scheduler_3d = build_scheduler(scheduler_3d_cfg, optimizer_3d)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_2d = CheckpointerV2(model_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d = checkpointer_2d.load(cfg.MODEL_2D.CKPT_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_3d = CheckpointerV2(model_3d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d = checkpointer_3d.load(cfg.MODEL_3D.CKPT_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    
    # Configure models
    if cfg.TRAIN.UPDATE_PARAMS.lower() == "norm":
        model_2d, log_modules_2d = configure_model_norm(model_2d, cfg.TRAIN.RESET_DROPOUT)
        model_3d, log_modules_3d = configure_model_norm(model_3d, cfg.TRAIN.RESET_DROPOUT)
        logger.info("[Student Model 2D] - Config: Only update normalization layer: {}".format(log_modules_2d.pop(0)))
        logger.info("[Student Model 3D] - Config: Only update normalization layer: {}".format(log_modules_3d.pop(0)))
    elif cfg.TRAIN.UPDATE_PARAMS.lower() == "all":
        model_2d, log_modules_2d = configure_model_all(model_2d, cfg.TRAIN.RESET_DROPOUT)
        model_3d, log_modules_3d = configure_model_all(model_3d, cfg.TRAIN.RESET_DROPOUT)
    else:
        raise IndexError("Required update params has not been defined yet: {}".format(cfg.TRAIN.UPDATE_PARAMS))
    # Log reset dropout params
    if cfg.TRAIN.RESET_DROPOUT:
        logger.info("[Student Model 2D] - Config: Reset Dropout layer: {}".format(log_modules_2d[0]))
        logger.info("[Student Model 3D] - Config: Reset Dropout layer: {}".format(log_modules_3d[0]))

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    start_iteration = 0

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader_trg = build_dataloader(cfg, mode='test', domain='target', start_iteration=start_iteration, tta=True)
    logger.info('Start training from iteration {}'.format(start_iteration))

    tta_input = {
        "model_2d": model_2d, "model_3d": model_3d,                 # models
        "optimizer_2d": optimizer_2d, "optimizer_3d": optimizer_3d, # optimizers
        "scheduler_2d": scheduler_2d, "scheduler_3d": scheduler_3d, # schedulers
        "train_dataloader_trg": train_dataloader_trg,               # dataloader
        "summary_writer": summary_writer,                           # Tensorboard writer
        "checkpoint_data_2d": checkpoint_data_2d,
        "checkpoint_data_3d": checkpoint_data_3d,
        "checkpointer_2d": checkpointer_2d,
        "checkpointer_3d": checkpointer_3d,
        "save_pth": save_pth,
    }
    
    # Various TTA method options
    if tta_type.upper() == "ETA":
        evaluator_dict = eta(cfg, logger, **tta_input)
    elif tta_type.upper() == "MMTTA":
        evaluator_dict = mmtta(cfg, logger, **tta_input)
    elif tta_type.upper() == "PSLABEL":
        evaluator_dict = pslabel(cfg, logger, **tta_input)
    elif tta_type.upper() == "SAR":
        evaluator_dict = sar(cfg, logger, **tta_input)
    elif tta_type.upper() == "TENT":
        evaluator_dict = tent(cfg, logger, **tta_input)
    elif tta_type.upper() == "LATTE":
        evaluator_dict = latte(cfg, logger, **tta_input)
    elif tta_type.upper() == "BN_GT":
        evaluator_dict = bn_gt(cfg, logger, **tta_input)
    elif tta_type.upper() == "XMUDA":
        evaluator_dict = xmuda(cfg, logger, **tta_input)
    else:
        raise NotImplementedError(
            "The require TTA method is not supported: {}".format(tta_type.upper()))
    
    # Report final results
    logger.info('-'*65)
    logger.info('Evaluation Results')
    logger.info('-'*65)
    for modality, evaluator in evaluator_dict.items():
        logger.info('{} overall accuracy={:.2f}%'.format(modality, 100.0 * evaluator.overall_acc))
        logger.info('{} overall IOU={:.2f}'.format(modality, 100.0 * evaluator.overall_iou))
        logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.format(modality, evaluator.print_table()))

    # save final results
    eval_keys = evaluator_dict.keys()
    iou_to_excel(evaluator_dict, osp.join(output_dir, 'class_iou.xlsx'), eval_keys)
    logger.info("Class-wise IoU saved to {}".format(osp.join(output_dir, 'class_iou.xlsx')))
    

def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from latte.common.config import purge_cfg
    from latte.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    hostname = socket.gethostname()
    # replace '@' with config path
    if output_dir:
        models_output_dir = osp.join(cfg.OUTPUT_DIR)
        month_day = time.strftime('%m%d')
        spec_dir = osp.splitext(args.config_file)[0].replace('/', '_')
        spec_dir = month_day + spec_dir[7:] + '_' + os.environ['CUDA_VISIBLE_DEVICES']
        models_output_dir = osp.join(models_output_dir, spec_dir)
        flag = 1
        # check whether there exists a same dir. If so, generate a new one by adding number at the end
        while osp.isdir(models_output_dir):
            pth_idx = models_output_dir.rfind('-')
            models_output_dir = models_output_dir[:pth_idx] + '-' + str(flag) \
                if pth_idx != -1 else models_output_dir + '-' + str(flag)
            flag += 1
        os.makedirs(models_output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m%d')
    run_name = '{:s}_{:s}'.format(hostname, timestamp)
    tta_type = str(cfg.TTA.TYPE)

    logs_output_dir = models_output_dir
    log_file_pth = osp.join(
        logs_output_dir,
        "{}_{}_train_{:s}_{}.log".format(args.task, tta_type, run_name, os.environ['CUDA_VISIBLE_DEVICES'])
    )
    # check existence of log file
    flag = 1
    while osp.exists(log_file_pth):
        pth_idx = log_file_pth.rfind('-')
        log_file_pth = log_file_pth[:pth_idx] + '-' + str(flag) + '.log' \
            if pth_idx != -1 else log_file_pth[:-4] + '-' + str(flag) + '.log'
        flag += 1
    logger = get_logger(
        output=log_file_pth,
        abbrev_name=tta_type
        )
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    # assert cfg.TRAIN.XMUDA.lambda_xm_src > 0 or cfg.TRAIN.XMUDA.lambda_xm_trg > 0 or cfg.TRAIN.XMUDA.lambda_pl > 0 or \
    #        cfg.TRAIN.XMUDA.lambda_minent > 0
    train(cfg, logger, models_output_dir, run_name, args.save_pth, tta_type)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    
    # Limit cpu usage if needed
    main()
