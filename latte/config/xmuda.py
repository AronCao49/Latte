"""xMUDA experiments configuration"""
import os.path as osp

from latte.common.config.base import CN, _C
import math

# public alias
cfg = _C
_C.VAL.METRIC = 'seg_iou'
# KNN search
_C.VAL.use_knn = False
_C.VAL.knn_prob = False
_C.VAL.prob_en = False

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN.CLASS_WEIGHTS = []
_C.TRAIN.UPDATE_PARAMS = "all"
_C.TRAIN.RESET_DROPOUT = False
_C.TRAIN.TTA_EVAL_PERIOD = 500

# ---------------------------------------------------------------------------- #
# xMUDA options
# ---------------------------------------------------------------------------- #
_C.TTA.XMUDA = CN()
_C.TTA.XMUDA.lambda_xm_trg = 0.1
_C.TTA.XMUDA.lambda_pl = 0.0

# ---------------------------------------------------------------------------- #
# Historical Revisit options
# ---------------------------------------------------------------------------- #
_C.TTA.LATTE = CN()
_C.TTA.LATTE.start_iter = 0
_C.TTA.LATTE.restore_rate = 0.0
_C.TTA.LATTE.restore_thre = 0.0
_C.TTA.LATTE.lambda_ety_ps = 0.0
_C.TTA.LATTE.lambda_hr_23_trg = 0.0
_C.TTA.LATTE.lambda_hr_32_trg = 0.0

# ---------------------------------------------------------------------------- #
# MMTTA options
# ---------------------------------------------------------------------------- #
_C.TTA.MMTTA = CN()

# ---------------------------------------------------------------------------- #
# TENT options
# ---------------------------------------------------------------------------- #
_C.TTA.TENT = CN()
_C.TTA.TENT.lambda_tent = 1.0
_C.TTA.TENT.lambda_pl = 0.0
_C.TTA.TENT.lambda_xm_trg = 0.0

# ---------------------------------------------------------------------------- #
# ETA options
# ---------------------------------------------------------------------------- #
_C.TTA.ETA = CN()
_C.TTA.ETA.e_margin = math.log(1000)*0.40
_C.TTA.ETA.d_margin = 0.05
_C.TTA.ETA.lambda_eta = 1.0
_C.TTA.ETA.lambda_xm_trg = 0.0

# ---------------------------------------------------------------------------- #
# SAR options
# ---------------------------------------------------------------------------- #
_C.TTA.SAR = CN()
_C.TTA.SAR.margin = math.log(1000)*0.40
_C.TTA.SAR.reset = False
_C.TTA.SAR.lambda_xm_trg = 0.0

# ---------------------------------------------------------------------------- #
# PsLabel options
# ---------------------------------------------------------------------------- #
_C.TTA.PSLABEL = CN()
_C.TTA.PSLABEL.start_iter = 0
_C.TTA.PSLABEL.lambda_ps = 0.0

# ---------------------------------------------------------------------------- #
# DA options
# ---------------------------------------------------------------------------- #
_C.TRAIN.DA_METHOD = None

# ---------------------------------------------------------------------------- #
# Depth Prediction options
# ---------------------------------------------------------------------------- #
_C.TRAIN.DEPTH_PRED_COE = CN()
_C.TRAIN.DEPTH_PRED_COE.lambda_dp_src = 0.0
_C.TRAIN.DEPTH_PRED_COE.lambda_dp_trg = 0.0

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASET_SOURCE = CN()
_C.DATASET_SOURCE.TYPE = ''
_C.DATASET_SOURCE.TRAIN = tuple()
_C.DATASET_SOURCE.VAL = tuple()

_C.DATASET_TARGET = CN()
_C.DATASET_TARGET.TYPE = ''
_C.DATASET_TARGET.TRAIN = tuple()
_C.DATASET_TARGET.VAL = tuple()
_C.DATASET_TARGET.VAL_CORR = tuple()
_C.DATASET_TARGET.TEST = tuple()
_C.DATASET_TARGET.VISUAL = tuple()

# NuScenesSCN
_C.DATASET_SOURCE.NuScenesSCN = CN()
_C.DATASET_SOURCE.NuScenesSCN.preprocess_dir = ''
_C.DATASET_SOURCE.NuScenesSCN.nuscenes_dir = ''
_C.DATASET_SOURCE.NuScenesSCN.merge_classes = True
_C.DATASET_SOURCE.NuScenesSCN.use_feats = True
# 3D
_C.DATASET_SOURCE.NuScenesSCN.scale = 20
_C.DATASET_SOURCE.NuScenesSCN.full_scale = 4096
_C.DATASET_SOURCE.NuScenesSCN.front_only = False
# 2D
_C.DATASET_SOURCE.NuScenesSCN.resize = (400, 225)

_C.DATASET_SOURCE.NuScenesSCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.NuScenesSCN.augmentation = CN()
_C.DATASET_SOURCE.NuScenesSCN.augmentation.bottom_crop = ()
_C.DATASET_SOURCE.NuScenesSCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.NuScenesSCN.augmentation.flip_x = 0.5
_C.DATASET_SOURCE.NuScenesSCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.NuScenesSCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.NuScenesSCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.NuScenesSCN.augmentation.color_jitter = (0.4, 0.4, 0.4)
# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.NuScenesSCN = CN(_C.DATASET_SOURCE.NuScenesSCN)
_C.DATASET_TARGET.NuScenesSCN.pselab_paths = tuple()

# A2D2SCN
_C.DATASET_SOURCE.A2D2SCN = CN()
_C.DATASET_SOURCE.A2D2SCN.preprocess_dir = ''
_C.DATASET_SOURCE.A2D2SCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.A2D2SCN.scale = 20
_C.DATASET_SOURCE.A2D2SCN.full_scale = 4096
# 2D
_C.DATASET_SOURCE.A2D2SCN.use_image = True
_C.DATASET_SOURCE.A2D2SCN.resize = (480, 302)
_C.DATASET_SOURCE.A2D2SCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.A2D2SCN.augmentation = CN()
_C.DATASET_SOURCE.A2D2SCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.A2D2SCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.A2D2SCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.A2D2SCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.A2D2SCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.A2D2SCN.augmentation.color_jitter = (0.4, 0.4, 0.4)

# SemanticKITTISCN
_C.DATASET_SOURCE.SemanticKITTISCN = CN()
_C.DATASET_SOURCE.SemanticKITTISCN.root_dir = ''
_C.DATASET_SOURCE.SemanticKITTISCN.merge_classes = True
_C.DATASET_SOURCE.SemanticKITTISCN.cat_type = 's'
# 3D
_C.DATASET_SOURCE.SemanticKITTISCN.scale = 20
_C.DATASET_SOURCE.SemanticKITTISCN.full_scale = 4096
_C.DATASET_SOURCE.SemanticKITTISCN.front_only = True
# 2D
_C.DATASET_SOURCE.SemanticKITTISCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation = CN()
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.bottom_crop = (480, 302)
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.SemanticKITTISCN.augmentation.color_jitter = (0.4, 0.4, 0.4)
# copy over the same arguments to target dataset settings
_C.DATASET_TARGET.SemanticKITTISCN = CN(_C.DATASET_SOURCE.SemanticKITTISCN)
_C.DATASET_TARGET.SemanticKITTISCN.ps_label_dir = None

# SynthiaSCN
_C.DATASET_SOURCE.SynthiaSCN = CN()
_C.DATASET_SOURCE.SynthiaSCN.synthia_dir = ''
_C.DATASET_SOURCE.SynthiaSCN.merge_classes = True
# 3D
_C.DATASET_SOURCE.SynthiaSCN.scale = 20
_C.DATASET_SOURCE.SynthiaSCN.full_scale = 4096
# 2D
_C.DATASET_SOURCE.SynthiaSCN.resize = (640, 380)
_C.DATASET_SOURCE.SynthiaSCN.image_normalizer = ()
# 3D augmentation
_C.DATASET_SOURCE.SynthiaSCN.augmentation = CN()
_C.DATASET_SOURCE.SynthiaSCN.augmentation.noisy_rot = 0.1
_C.DATASET_SOURCE.SynthiaSCN.augmentation.flip_y = 0.5
_C.DATASET_SOURCE.SynthiaSCN.augmentation.rot_z = 6.2831  # 2 * pi
_C.DATASET_SOURCE.SynthiaSCN.augmentation.transl = True
# 2D augmentation
_C.DATASET_SOURCE.SynthiaSCN.augmentation.fliplr = 0.5
_C.DATASET_SOURCE.SynthiaSCN.augmentation.bottom_crop = (350, 350)
_C.DATASET_SOURCE.SynthiaSCN.augmentation.color_jitter = (0.4, 0.4, 0.4)


# ---------------------------------------------------------------------------- #
# Model 2D
# ---------------------------------------------------------------------------- #
_C.MODEL_2D = CN()
_C.MODEL_2D.TYPE = ''
_C.MODEL_2D.CKPT_PATH = ''
_C.MODEL_2D.NUM_CLASSES = 5
_C.MODEL_2D.DUAL_HEAD = False
_C.MODEL_2D.LOSS = "Default"
# ---------------------------------------------------------------------------- #
# UNetResNet34 options
# ---------------------------------------------------------------------------- #
_C.MODEL_2D.UNetResNet34 = CN()
_C.MODEL_2D.UNetResNet34.pretrained = True
_C.MODEL_2D.UNetResNet34.test_cfg = CN()
_C.MODEL_2D.UNetResNet34.test_cfg.mode = 'whole'
_C.MODEL_2D.UNetResNet34.test_cfg.stride = ()
_C.MODEL_2D.UNetResNet34.test_cfg.crop_size = ()
# ---------------------------------------------------------------------------- #
# DeepLabV3_ResNet50 options
# ---------------------------------------------------------------------------- #
_C.MODEL_2D.DeepLabV3 = CN()
_C.MODEL_2D.DeepLabV3.pretrained = True
_C.MODEL_2D.DeepLabV3.test_cfg = CN()
_C.MODEL_2D.DeepLabV3.test_cfg.mode = 'whole'
_C.MODEL_2D.DeepLabV3.test_cfg.stride = ()
_C.MODEL_2D.DeepLabV3.test_cfg.crop_size = ()
# ---------------------------------------------------------------------------- #
# DeepLabV3_ResNet50 options
# ---------------------------------------------------------------------------- #
_C.MODEL_2D.DeepLabV3_MobileNet = CN()
_C.MODEL_2D.DeepLabV3_MobileNet.pretrained = True
_C.MODEL_2D.DeepLabV3_MobileNet.test_cfg = CN()
_C.MODEL_2D.DeepLabV3_MobileNet.test_cfg.mode = 'whole'
_C.MODEL_2D.DeepLabV3_MobileNet.test_cfg.stride = ()
_C.MODEL_2D.DeepLabV3_MobileNet.test_cfg.crop_size = ()
# ---------------------------------------------------------------------------- #
# SegFormer options (b2 be default)
# ---------------------------------------------------------------------------- #
_C.MODEL_2D.SegFormer = CN()
_C.MODEL_2D.SegFormer.pretrained_path = ""
_C.MODEL_2D.SegFormer.backbone = "mit_b2"
_C.MODEL_2D.SegFormer.decoder_cfg = CN()
_C.MODEL_2D.SegFormer.decoder_cfg.feature_strides = [4, 8, 16, 32]
_C.MODEL_2D.SegFormer.decoder_cfg.in_channels = [64, 128, 320, 512]
_C.MODEL_2D.SegFormer.decoder_cfg.embedding_dim = 768
_C.MODEL_2D.SegFormer.decoder_cfg.dropout_rate = 0.1
_C.MODEL_2D.SegFormer.test_cfg = CN()
_C.MODEL_2D.SegFormer.test_cfg.mode = 'whole'
_C.MODEL_2D.SegFormer.test_cfg.stride = ()
_C.MODEL_2D.SegFormer.test_cfg.crop_size = ()
# ---------------------------------------------------------------------------- #
# Model 3D
# ---------------------------------------------------------------------------- #
_C.MODEL_3D = CN()
_C.MODEL_3D.TYPE = ''
_C.MODEL_3D.CKPT_PATH = ''
_C.MODEL_3D.NUM_CLASSES = 5
_C.MODEL_3D.DUAL_HEAD = False
_C.MODEL_3D.LOSS = "Default"
# ----------------------------------------------------------------------------- #
# SCN options
# ----------------------------------------------------------------------------- #
_C.MODEL_3D.SCN = CN()
_C.MODEL_3D.SCN.in_channels = 1
_C.MODEL_3D.SCN.m = 16  # number of unet features (multiplied in each layer)
_C.MODEL_3D.SCN.block_reps = 1  # block repetitions
_C.MODEL_3D.SCN.residual_blocks = False  # ResNet style basic blocks
_C.MODEL_3D.SCN.full_scale = 4096
_C.MODEL_3D.SCN.num_planes = 7
_C.MODEL_3D.SCN.pretrained = False
# ----------------------------------------------------------------------------- #
# SPVCNN options
# ----------------------------------------------------------------------------- #
_C.MODEL_3D.SPVCNN = CN()
_C.MODEL_3D.SPVCNN.pretrained = False
_C.MODEL_3D.SPVCNN.cr = 1.0
_C.MODEL_3D.SPVCNN_Base = CN()
_C.MODEL_3D.SPVCNN_Base.pretrained = True
# ----------------------------------------------------------------------------- #
# SalsaNext options
# ----------------------------------------------------------------------------- #
_C.MODEL_3D.SalsaNext = CN()
_C.MODEL_3D.SalsaNext.pretrained = False
_C.MODEL_3D.SalsaNext_Base = CN()
_C.MODEL_3D.SalsaNext_Base.pretrained = True
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Historical Revisit Module
# ---------------------------------------------------------------------------- #
_C.HR_BLOCK = CN()
_C.HR_BLOCK.temp_wd_size = 50
_C.HR_BLOCK.max_range = 50
_C.HR_BLOCK.voxel_size = 0.05
_C.HR_BLOCK.conf_perc = None
_C.HR_BLOCK.exclude_labels = None
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Corresponding Relation Module
# ---------------------------------------------------------------------------- #
_C.CR_BLOCK = CN()
_C.CR_BLOCK.reduction = "mean"
_C.CR_BLOCK.conf_mask = False
_C.CR_BLOCK.ety_weight = False
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Dense 2D Module
# ---------------------------------------------------------------------------- #
_C.Dense_2D = CN()
_C.Dense_2D.reduction = "mean"
_C.Dense_2D.exclude_labels = None
_C.Dense_2D.conf_perc = None
# ---------------------------------------------------------------------------- #


# Misc options
# ---------------------------------------------------------------------------- #
# @ will be replaced by config path
_C.OUTPUT_DIR = osp.expanduser('~/workspace/outputs/xmuda/@')
