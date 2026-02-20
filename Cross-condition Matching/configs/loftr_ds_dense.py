from yacs.config import CfgNode as CN

# ==========================================================
# Root config node
# ==========================================================
_CN = CN()

##############  ↓  LoFTR Pipeline  ↓  ##############
_CN.LOFTR = CN()
_CN.LOFTR.BACKBONE_TYPE = 'ResNetFPN'
_CN.LOFTR.RESOLUTION = (16, 4)  # options: [(8, 2), (16, 4)]
_CN.LOFTR.FINE_WINDOW_SIZE = 5
_CN.LOFTR.FINE_CONCAT_COARSE_FEAT = True

# ----------------------------------------------------------
# 1. LoFTR backbone (ResNet-FPN)
# ----------------------------------------------------------
_CN.LOFTR.RESNETFPN = CN()
_CN.LOFTR.RESNETFPN.INITIAL_DIM = 128
_CN.LOFTR.RESNETFPN.BLOCK_DIMS = [128, 196, 256, 512, 1024, 1024]

# ----------------------------------------------------------
# 2. LoFTR coarse transformer
# ----------------------------------------------------------
_CN.LOFTR.COARSE = CN()
_CN.LOFTR.COARSE.D_MODEL = 768
_CN.LOFTR.COARSE.D_FFN = 768
_CN.LOFTR.COARSE.NHEAD = 8
_CN.LOFTR.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.LOFTR.COARSE.ATTENTION = 'linear'
_CN.LOFTR.COARSE.TEMP_BUG_FIX = True

# ----------------------------------------------------------
# 3. Coarse matching (⚠ 已合并你的 override)
# ----------------------------------------------------------
_CN.LOFTR.MATCH_COARSE = CN()
_CN.LOFTR.MATCH_COARSE.THR = 0.2
_CN.LOFTR.MATCH_COARSE.BORDER_RM = 1

# === merged from your cfg override ===
_CN.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
_CN.LOFTR.MATCH_COARSE.SPARSE_SPVS = False
# =====================================

_CN.LOFTR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.LOFTR.MATCH_COARSE.SKH_ITERS = 3
_CN.LOFTR.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.LOFTR.MATCH_COARSE.SKH_PREFILTER = False
_CN.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2
_CN.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200

# ----------------------------------------------------------
# 4. LoFTR fine transformer
# ----------------------------------------------------------
_CN.LOFTR.FINE = CN()
_CN.LOFTR.FINE.D_MODEL = 128
_CN.LOFTR.FINE.D_FFN = 128
_CN.LOFTR.FINE.NHEAD = 8
_CN.LOFTR.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.LOFTR.FINE.ATTENTION = 'linear'

# ----------------------------------------------------------
# 5. Loss
# ----------------------------------------------------------
_CN.LOFTR.LOSS = CN()
_CN.LOFTR.LOSS.COARSE_TYPE = 'focal'
_CN.LOFTR.LOSS.COARSE_WEIGHT = 1.0
_CN.LOFTR.LOSS.FOCAL_ALPHA = 0.25
_CN.LOFTR.LOSS.FOCAL_GAMMA = 2.0
_CN.LOFTR.LOSS.POS_WEIGHT = 1.0
_CN.LOFTR.LOSS.NEG_WEIGHT = 1.0

_CN.LOFTR.LOSS.FINE_TYPE = 'l2_with_std'
_CN.LOFTR.LOSS.FINE_WEIGHT = 1.0
_CN.LOFTR.LOSS.FINE_CORRECT_THR = 1.0

##############  Dataset  ##############
_CN.DATASET = CN()

_CN.DATASET.TRAINVAL_DATA_SOURCE = None
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None

_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None
_CN.DATASET.VAL_INTRINSIC_PATH = None

_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None
_CN.DATASET.TEST_INTRINSIC_PATH = None

_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None

# MegaDepth
_CN.DATASET.MGDPT_IMG_RESIZE = 640
_CN.DATASET.MGDPT_IMG_PAD = True
_CN.DATASET.MGDPT_DEPTH_PAD = True
_CN.DATASET.MGDPT_DF = 8

##############  Trainer  ##############
_CN.TRAINER = CN()

_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 96
_CN.TRAINER.CANONICAL_LR = 3e-3
_CN.TRAINER.SCALING = None
_CN.TRAINER.FIND_LR = False

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"
_CN.TRAINER.TRUE_LR = None
_CN.TRAINER.ADAM_DECAY = 0.0
_CN.TRAINER.ADAMW_DECAY = 0.1

# warmup
_CN.TRAINER.WARMUP_TYPE = 'linear'
_CN.TRAINER.WARMUP_RATIO = 0.0
_CN.TRAINER.WARMUP_STEP = 4800

# scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'

# === merged from your cfg override ===
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12, 17, 20, 23, 26, 29]
# =====================================

_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30
_CN.TRAINER.ELR_GAMMA = 0.999992

# plotting
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32
_CN.TRAINER.PLOT_MODE = 'evaluation'
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometry & pose
_CN.TRAINER.EPI_ERR_THR = 5e-4
_CN.TRAINER.POSE_GEO_MODEL = 'E'
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# sampler
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True
_CN.TRAINER.SB_SUBSET_SHUFFLE = True
_CN.TRAINER.SB_REPEAT = 1
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# misc
_CN.TRAINER.GRADIENT_CLIPPING = 0.5
_CN.TRAINER.SEED = 66


# ==========================================================
# YACS-required exports
# ==========================================================
def get_cfg_defaults():
    return _CN.clone()

# ⚠ REQUIRED by config.merge_from_file(py)
cfg = _CN.clone()
