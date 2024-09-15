#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from latte.common.solver.build import build_optimizer, build_scheduler
from latte.common.utils.checkpoint import CheckpointerV2
from latte.common.utils.logger import get_logger, setup_logger
from latte.common.utils.metric_logger import MetricLogger
from latte.common.utils.torch_util import set_random_seed
from latte.data.collate import inverse_to_all
from latte.data.utils.all_to_partial import pc2img_logit
from latte.models.build import build_model_2d, build_model_3d
from latte.data.build import build_dataloader
from latte.data.utils.validate import validate

os.environ['OMP_NUM_THREADS'] = "4"
torch.set_num_threads(4)

def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='configs/synthia_semantic_kitti/baseline.yaml',
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


def train(cfg, logger, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #
    # logger = logging.getLogger('xmuda.train')

    set_random_seed(cfg.RNG_SEED)

    # build 2d model
    model_2d, train_metric_2d = build_model_2d(cfg)
    logger.info('Build 2D model:\n{}'.format(str(cfg.MODEL_2D.TYPE)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    # build 3d model
    model_3d, train_metric_3d = build_model_3d(cfg)
    logger.info('Build 3D model:\n{}'.format(str(cfg.MODEL_3D.TYPE)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

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
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d = checkpointer_2d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_3d = CheckpointerV2(model_3d,
                                     optimizer=optimizer_3d,
                                     scheduler=scheduler_3d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d = checkpointer_3d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_2d.get('iteration', 0)

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='source') if val_period > 0 else None

    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None)
    }
    best_metric_iter = {'2d': -1, '3d': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d])
    val_metric_logger = MetricLogger(delimiter='  ')

    def setup_train():
        # set training mode
        model_2d.train()
        model_3d.train()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_2d.eval()
        model_3d.eval()
        # reset metric
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    setup_train()
    end = time.time()
    train_iter_src = enumerate(train_dataloader_src)
    for iteration in range(start_iteration, max_iteration):
        # fetch data_batches for source & target
        _, data_batch_src = train_iter_src.__next__()
        # copy data from cpu to gpu
        if 'SCN' in cfg.MODEL_3D.TYPE:
            data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
        elif 'SPVCNN' in cfg.MODEL_3D.TYPE:
            data_batch_src['lidar'] = data_batch_src['lidar'].cuda()
        elif 'SalsaNext' in cfg.MODEL_3D.TYPE:
                data_batch_src['proj_in'] = data_batch_src['proj_in'].float().cuda()
                data_batch_src['all_seg_label'] = data_batch_src['all_seg_label'].detach().numpy()
        else:
            raise NotImplementedError('The requested network {} is not supported for now.'.format(cfg.MODEL_3D.TYPE))
        # For partial FOV
        if 'pc2img_idx' in data_batch_src.keys():
            pc2img_idx_src = [idx.cuda() for idx in data_batch_src['pc2img_idx']]
        # copy seg_label & image
        data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
        data_batch_src['img'] = data_batch_src['img'].cuda()

        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()

        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #
        preds_2d = model_2d(data_batch_src)
        preds_3d = model_3d(data_batch_src)
        seg_label_2d = data_batch_src['seg_label']

        # network-based postprocess
        if "SPVCNN" in cfg.MODEL_3D.TYPE:
            preds_3d['seg_logit'] = inverse_to_all(preds_3d['seg_logit'], data_batch_src)
            preds_3d['seg_logit2'] = inverse_to_all(preds_3d['seg_logit2'], data_batch_src)
        if "pc2img_idx" in data_batch_src.keys():
            seg_logit_3d = pc2img_logit(preds_3d['seg_logit2'], pc2img_idx_src)
            seg_logit_3d_cls = pc2img_logit(preds_3d['seg_logit'], pc2img_idx_src)
            seg_label_2d = pc2img_logit(data_batch_src['seg_label'], pc2img_idx_src)

        # segmentation loss: cross entropy
        loss_src_2d = F.cross_entropy(preds_2d['seg_logit'], seg_label_2d, weight=class_weights)
        loss_src_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        train_metric_logger.update(loss_src_2d=loss_src_2d, loss_src_3d=loss_src_3d)
        loss_2d = loss_src_2d
        loss_3d = loss_src_3d

        # update metric (e.g. IoU)
        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d, {'seg_label': seg_label_2d})
            train_metric_3d.update_dict(preds_3d, data_batch_src)

        # backward
        loss_2d.backward()
        loss_3d.backward()

        optimizer_2d.step()
        optimizer_3d.step()

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time)
        torch.cuda.empty_cache()
        
        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr=optimizer_2d.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()

            validate(cfg,
                     model_2d,
                     model_3d,
                     val_dataloader,
                     val_metric_logger,
                     logger)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in ['2d', '3d']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter

            # restore training
            setup_train()

        scheduler_2d.step()
        scheduler_3d.step()
        end = time.time()

    for modality in ['2d', '3d']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(modality.upper(),
                                                                     cfg.VAL.METRIC,
                                                                     best_metric[modality] * 100,
                                                                     best_metric_iter[modality]))


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
        models_output_dir = osp.join(cfg.OUTPUT_DIR, 'models')
        month_day = time.strftime('%m%d')
        spec_dir = osp.splitext(args.config_file)[0].replace('/', '_')
        spec_dir = month_day + spec_dir[7:] + '_' + os.environ['CUDA_VISIBLE_DEVICES']
        models_output_dir = osp.join(models_output_dir, spec_dir)
        flag = 1
        # check whether there exists a same dir. If so, generate a new one by adding number at the end
        while osp.isdir(models_output_dir):
            models_output_dir = models_output_dir + '-' + str(flag)
            flag += 1
            continue
        os.makedirs(models_output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m%d')
    run_name = '{:s}-{:s}'.format(hostname, timestamp)

    log_file_pth = osp.join(
        models_output_dir,
        "{}_train_{:s}_{}.log".format(args.task, run_name, os.environ['CUDA_VISIBLE_DEVICES'])
    )
    logger = get_logger(
        output=log_file_pth,
        abbrev_name='SS2MM'
        )
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    # assert cfg.TRAIN.XMUDA.lambda_xm_trg == 0 and cfg.TRAIN.XMUDA.lambda_pl == 0
    train(cfg, logger, models_output_dir, run_name)


if __name__ == '__main__':
    main()
