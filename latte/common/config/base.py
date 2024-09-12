"""Basic experiments configuration
For different tasks, a specific configuration might be created by importing this basic config.
"""

from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Config definition
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Resume
# ---------------------------------------------------------------------------- #
# Resume dir to continue training
_C.RESUME_DIR = ''
# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True
# Whether to resume the optimizer and the scheduler
_C.RESUME_STATES = True
# Path of weights to resume
_C.RESUME_PATH = ''

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.TYPE = ''

# ---------------------------------------------------------------------------- #
# DataLoader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 12
# Whether to drop last
_C.DATALOADER.DROP_LAST = True

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.MODEL_2D = CN()
_C.OPTIMIZER.MODEL_2D.TYPE = ''
_C.OPTIMIZER.MODEL_3D = CN()
_C.OPTIMIZER.MODEL_3D.TYPE = ''

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.OPTIMIZER.MODEL_2D.BASE_LR = 0.001
_C.OPTIMIZER.MODEL_2D.WEIGHT_DECAY = 0.0
_C.OPTIMIZER.MODEL_3D.BASE_LR = 0.001
_C.OPTIMIZER.MODEL_3D.WEIGHT_DECAY = 0.0

# Specific parameters of optimizers
_C.OPTIMIZER.MODEL_2D.SGD = CN()
_C.OPTIMIZER.MODEL_2D.SGD.momentum = 0.9
_C.OPTIMIZER.MODEL_2D.SGD.dampening = 0.0

_C.OPTIMIZER.MODEL_3D.SGD = CN()
_C.OPTIMIZER.MODEL_3D.SGD.momentum = 0.9
_C.OPTIMIZER.MODEL_3D.SGD.dampening = 0.0

_C.OPTIMIZER.MODEL_2D.Adam = CN()
_C.OPTIMIZER.MODEL_2D.Adam.betas = (0.9, 0.999)

_C.OPTIMIZER.MODEL_3D.Adam = CN()
_C.OPTIMIZER.MODEL_3D.Adam.betas = (0.9, 0.999)

_C.OPTIMIZER.MODEL_2D.AdamW = CN()
_C.OPTIMIZER.MODEL_2D.AdamW.betas = (0.9, 0.999)
_C.OPTIMIZER.MODEL_2D.AdamW.param_keys = ()
_C.OPTIMIZER.MODEL_2D.AdamW.mult_types = ()
_C.OPTIMIZER.MODEL_2D.AdamW.mults = ()

_C.OPTIMIZER.MODEL_3D.AdamW = CN()
_C.OPTIMIZER.MODEL_3D.AdamW.betas = (0.9, 0.999)
_C.OPTIMIZER.MODEL_3D.AdamW.param_keys = ()
_C.OPTIMIZER.MODEL_3D.AdamW.mult_types = ()
_C.OPTIMIZER.MODEL_3D.AdamW.mults = ()

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.SCHEDULER = CN()
_C.SCHEDULER.MODEL_2D = CN()
_C.SCHEDULER.MODEL_2D.TYPE = ''
_C.SCHEDULER.MODEL_3D = CN()
_C.SCHEDULER.MODEL_3D.TYPE = ''

_C.SCHEDULER.MAX_ITERATION = 1

# 2D SCHEDULER LR
# Minimum learning rate. 0.0 for disable.
_C.SCHEDULER.MODEL_2D.CLIP_LR = 0.0

# Specific parameters of schedulers
_C.SCHEDULER.MODEL_2D.StepLR = CN()
_C.SCHEDULER.MODEL_2D.StepLR.step_size = 0
_C.SCHEDULER.MODEL_2D.StepLR.gamma = 0.1

_C.SCHEDULER.MODEL_2D.MultiStepLR = CN()
_C.SCHEDULER.MODEL_2D.MultiStepLR.milestones = ()
_C.SCHEDULER.MODEL_2D.MultiStepLR.gamma = 0.1

_C.SCHEDULER.MODEL_2D.PolynomialLR = CN()
_C.SCHEDULER.MODEL_2D.PolynomialLR.verbose = False
_C.SCHEDULER.MODEL_2D.PolynomialLR.power = 1.0
_C.SCHEDULER.MODEL_2D.PolynomialLR.total_iters = 100000

# 3D SCHEDULER LR
# Minimum learning rate. 0.0 for disable.
_C.SCHEDULER.MODEL_3D.CLIP_LR = 0.0

# Specific parameters of schedulers
_C.SCHEDULER.MODEL_3D.StepLR = CN()
_C.SCHEDULER.MODEL_3D.StepLR.step_size = 0
_C.SCHEDULER.MODEL_3D.StepLR.gamma = 0.1

_C.SCHEDULER.MODEL_3D.MultiStepLR = CN()
_C.SCHEDULER.MODEL_3D.MultiStepLR.milestones = ()
_C.SCHEDULER.MODEL_3D.MultiStepLR.gamma = 0.1

_C.SCHEDULER.MODEL_3D.PolynomialLR = CN()
_C.SCHEDULER.MODEL_3D.PolynomialLR.verbose = False
_C.SCHEDULER.MODEL_3D.PolynomialLR.power = 1.0
_C.SCHEDULER.MODEL_3D.PolynomialLR.total_iters = 100000

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Batch size
_C.TRAIN.BATCH_SIZE = 1
# Period to save checkpoints. 0 for disable
_C.TRAIN.CHECKPOINT_PERIOD = 0
# Period to log training status. 0 for disable
_C.TRAIN.LOG_PERIOD = 200
# Period to summary training status. 0 for disable
_C.TRAIN.SUMMARY_PERIOD = 0
# Max number of checkpoints to keep
_C.TRAIN.MAX_TO_KEEP = 5

# Regex patterns of modules and/or parameters to freeze
_C.TRAIN.FROZEN_PATTERNS = ()

# Auxullary task: depth prediction
_C.TRAIN.DEPTH_PRED = False
# ---------------------------------------------------------------------------- #
# Specific validation options
# ---------------------------------------------------------------------------- #
_C.VAL = CN()

# Batch size
_C.VAL.BATCH_SIZE = 1
# Period to validate. 0 for disable
_C.VAL.PERIOD = 0
# Period to log validation status. 0 for disable
_C.VAL.LOG_PERIOD = 20
# The metric for best validation performance
_C.VAL.METRIC = ''

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = '@'

# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
# -1 means use time seed.
_C.RNG_SEED = 1

# ---------------------------------------------------------------------------- #
# TTA methods and options
# ---------------------------------------------------------------------------- #
_C.TTA = CN()
_C.TTA.TYPE = ''
