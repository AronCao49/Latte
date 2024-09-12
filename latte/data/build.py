# from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader, default_collate
from yacs.config import CfgNode as CN

from latte.common.utils.torch_util import worker_init_fn
from latte.data.collate import get_collate_range, get_collate_scn, \
                               get_collate_seq, get_collate_seq_v2
from latte.common.utils.sampler import IterationBasedBatchSampler
from latte.data.nuscenes.nuscenes_dataloader import NuScenesSCN
from latte.data.a2d2.a2d2_dataloader import A2D2SCN
from latte.data.semantic_kitti.semantic_kitti_dataloader import SemanticKITTISCN
from latte.data.synthia.synthia_dataloader import SynthiaSCN


def build_dataloader(cfg, mode='train', domain='source', start_iteration=0, halve_batch_size=False, mix_match=False, seq=False, tta=False):
    assert mode in ['train', 'val', 'val_corr', 'test', 'train_labeled', 'train_unlabeled', 'visual']
    dataset_cfg = cfg.get('DATASET_' + domain.upper())
    split = dataset_cfg[mode.upper()]
    is_train = 'train' in mode
    batch_size = cfg['TRAIN'].BATCH_SIZE if is_train else cfg['VAL'].BATCH_SIZE
    if halve_batch_size:
        batch_size = batch_size // 2

    # build dataset
    # Make a copy of dataset_kwargs so that we can pop augmentation afterwards without destroying the cfg.
    # Note that the build_dataloader fn is called twice for train and val.
    dataset_kwargs = CN(dataset_cfg.get(dataset_cfg.TYPE, dict()))
    if 'SCN' in cfg.MODEL_3D.keys():
        assert dataset_kwargs.full_scale == cfg.MODEL_3D.SCN.full_scale
    augmentation = dataset_kwargs.pop('augmentation')
    augmentation = augmentation if is_train or not tta else dict()
    # disable point mix-match during val & test
    # if domain == 'target':
    #     use_pc_mm = dataset_kwargs.pop('use_pc_mm')
    #     dual_scan = dataset_kwargs.pop('dual_scan')
    #     use_pc_mm = use_pc_mm and mode == 'train'
    #     dual_scan = dual_scan and mode == 'train'
    # else:
    #     dual_scan = False
    #     use_pc_mm = False
    # use pselab_paths only when training on target
    if domain == 'target' and not is_train:
        try:
            dataset_kwargs.pop('pselab_paths')
        except KeyError:
            dataset_kwargs.pop('ps_label_dir')

    if dataset_cfg.TYPE == 'NuScenesSCN':
        dataset = NuScenesSCN(split=split,
                              output_orig=not is_train,
                              backbone=cfg.MODEL_3D.TYPE,
                              **dataset_kwargs,
                              **augmentation)
    elif dataset_cfg.TYPE == 'A2D2SCN':
        dataset = A2D2SCN(split=split,
                          backbone=cfg.MODEL_3D.TYPE,
                          **dataset_kwargs,
                          **augmentation)
    elif dataset_cfg.TYPE == 'SemanticKITTISCN':
        dataset = SemanticKITTISCN(split=split,
                                   output_orig=not is_train,
                                   backbone=cfg.MODEL_3D.TYPE,
                                   **dataset_kwargs,
                                   **augmentation)
    elif dataset_cfg.TYPE == 'SynthiaSCN':
        dataset = SynthiaSCN(output_orig=not is_train,
                             backbone=cfg.MODEL_3D.TYPE,
                             **dataset_kwargs,
                             **augmentation)
    else:
        raise ValueError('Unsupported type of dataset: {}.'.format(dataset_cfg.TYPE))

    if 'SCN' in dataset_cfg.TYPE:
        collate_fn = get_collate_scn(output_orig=not is_train,
                                     output_depth=cfg.TRAIN.DEPTH_PRED)
    else:
        collate_fn = default_collate

    if is_train:
        # sampler = SequentialSampler(dataset)
        sampler = RandomSampler(dataset) if not tta else SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=cfg.DATALOADER.DROP_LAST)
        batch_sampler = IterationBasedBatchSampler(batch_sampler, cfg.SCHEDULER.MAX_ITERATION, start_iteration) \
            if not tta else batch_sampler
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn
        )

    return dataloader
