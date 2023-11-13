# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os

import yaml
from yacs.config import CfgNode as CN

_C = CN() # 创建一个CN容器来装载参数

# Base config files
_C.BASE = ['']
# mode: train or test
_C.MODE = ''
_C.DEVICE = '1'
_C.POPAR_FORM = True
# Path to output folder, overwritten by command line argument
_C.OUTPUT = '/sda1/zhouziyu/ssl/downstream_checkpoints/RSNAPneumonia/'
_C.PRETRAIN_MODE = 'popar_pec'
_C.LINEAR_PROB = False
_C.BACKBONE = 'swin_base'
_C.PATIENCE = 20

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Input image size
_C.DATA.IMG_SIZE = 224
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = '/sda1/zhouziyu/ssl/dataset/RSNA/stage_2_train_images_png/'
# dataset name
_C.DATA.DATASET = 'RSNA'
# dataset lists and labels
_C.DATA.TRAIN_LIST = '/home/zhouziyu/warmup/sslpretrain/BenchmarkTransformers/dataset/RSNAPneumonia_train.txt'
_C.DATA.VAL_LIST = '/home/zhouziyu/warmup/sslpretrain/BenchmarkTransformers/dataset/RSNAPneumonia_val.txt'
_C.DATA.TEST_LIST = '/home/zhouziyu/warmup/sslpretrain/BenchmarkTransformers/dataset/RSNAPneumonia_test.txt'


# Dataset split fold
_C.DATA.FOLD = '0'
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_base_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 3 # 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300 #150
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False # True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'#'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''

# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'): # hasattr判断args中是否有name属性
            return True
        return False

    # merge from specific arguments
    if _check_args('backbone'):
        config.BACKBONE = args.backbone
    if _check_args('device'):
        config.DEVICE = args.device
    if _check_args('mode'):
        config.MODE = args.mode
    if _check_args('popar_form'):
        config.POPAR_FORM = args.popar_form
    if _check_args('pretrain_mode'):
        config.PRETRAIN_MODE = args.pretrain_mode
    if _check_args('pretrain_weight'):
        config.MODEL.PRETRAINED = args.pretrain_weight
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('dataset'):
        config.DATA.DATASET = args.dataset
    if _check_args('fold'):
        config.DATA.FOLD = args.fold
    if _check_args('img_size'):
        config.DATA.IMG_SIZE = args.img_size
    if _check_args('num_classes'):
        config.MODEL.NUM_CLASSES = args.num_classes
    if _check_args('epoch'):
        config.TRAIN.EPOCHS = args.epoch
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('train_list'):
        config.DATA.TRAIN_LIST = args.train_list
    if _check_args('val_list'):
        config.DATA.VAL_LIST = args.val_list
    if _check_args('test_list'):
        config.DATA.TEST_LIST = args.test_list
    # if _check_args('image_size'):
    # config.DATA.IMAGE_SIZE = args.image_size
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    if _check_args('local_rank'):
        config.LOCAL_RANK = args.local_rank
    if _check_args('linear_prob'):
        config.LINEAR_PROB = args.linear_prob
    if _check_args('patience'):
        config.PATIENCE = args.patience

    # output folder
    if config.DATA.IMG_SIZE == 448:
        if config.LINEAR_PROB:
            config.OUTPUT = os.path.join(config.OUTPUT, config.PRETRAIN_MODE+'_linearprob_448_'+config.DATA.FOLD)
        else:
            config.OUTPUT = os.path.join(config.OUTPUT, config.PRETRAIN_MODE+'_448_'+config.DATA.FOLD)
    else:
        config.OUTPUT = os.path.join(config.OUTPUT, config.PRETRAIN_MODE+config.DATA.FOLD)

    if config.PRETRAIN_MODE == 'popar_pec':
        config.MODEL.PRETRAINED = '/mnt/sda/zhouziyu/liang/NIHChestXray/checkpoints/ssl_pretrained_weight/pec_popar/pec+popar/output/last.pth'
    elif config.PRETRAIN_MODE == 'only_pec':
        config.MODEL.PRETRAINED = '/mnt/sda/zhouziyu/liang/NIHChestXray/checkpoints/ssl_pretrained_weight/pec_popar/onlypec_single/output/last.pth'
    elif config.PRETRAIN_MODE == 'popar': # my POPAR pretrained model
        config.MODEL.PRETRAINED = '/mnt/sda/zhouziyu/liang/NIHChestXray/checkpoints/ssl_pretrained_weight/POPAR_swin_depth2,2,18,2_head4,8,16,32_nih14_in_channel3/last.pth'
    elif config.PRETRAIN_MODE == 'imagenet': # imagenet pretrained model
        config.MODEL.PRETRAINED = '/mnt/sda/zhouziyu/liang/NIHChestXray/checkpoints/swin_base_patch4_window7_224_22k.pth'
    elif config.PRETRAIN_MODE == 'NIHchest': # NIHchest pretrained model
        config.MODEL.PRETRAINED = "/mnt/sda/zhouziyu/liang/NIHChestXray/checkpoints/scratch1/swin_base_patch4_window7_224/default/best.pth"
    elif config.PRETRAIN_MODE == 'from_scratch':
        config.MODEL.PRETRAINED = ''

    if config.DATA.IMG_SIZE == 224:
        config.DATA.CROP_SIZE = 256
    elif config.DATA.IMG_SIZE == 448:
        config.DATA.CROP_SIZE = 512

    config.freeze()

def get_config_RSNA(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config