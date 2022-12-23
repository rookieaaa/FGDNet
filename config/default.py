from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as op
import yaml
from yacs.config import CfgNode as CN

from libs.comm import comm


_C = CN()

_C.BASE = ['']
_C.NAME = ''
_C.DATA_DIR = ''
_C.DIST_BACKEND = 'nccl'
_C.GPUS = (0,)
_C.MULTIPROCESSING_DISTRIBUTED = True
_C.OUTPUT_DIR = ''
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 20
_C.RANK = 0
_C.VERBOSE = True
_C.WORKERS = 4
_C.MODEL_SUMMARY = False

_C.AMP = CN()
_C.AMP.ENABLED = False
_C.AMP.MEMORY_FORMAT = 'nchw'

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = ''
_C.MODEL.NUM_CLASSES = 5
_C.MODEL.SET = CN(new_allowed=True)

# DATASET
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = ''
_C.DATASET.DATA_FORMAT = 'jpg'

# data augmentation
_C.AUG = CN()
_C.AUG.PATH = []
_C.AUG.PRETRAIN = []

# train
_C.TRAIN = CN()

_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.LR_SCHEDULER = CN(new_allowed=True)
_C.TRAIN.SCALE_LR = True
_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.OPTIMIZER_ARGS = CN(new_allowed=True)
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.WITHOUT_WD_LIST = []
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 100

_C.TRAIN.IMAGE_SIZE = [224, 224]
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

_C.TRAIN.EVAL_BEGIN_EPOCH = 0

_C.TRAIN.DETECT_ANOMALY = False

_C.TRAIN.CLIP_GRAD_NORM = 0.0
_C.TRAIN.SAVE_ALL_MODELS = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, op.join(op.dirname(cfg_file), cfg)
            )
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    config.merge_from_list(args.opts)
    config.TRAIN.LR *= comm.world_size
    file_name, _ = op.splitext(op.basename(args.cfg))
    config.NAME = file_name + config.NAME
    config.RANK = comm.rank
    config.TRAIN.LR_SCHEDULER.ARGS.epochs = config.TRAIN.END_EPOCH
    config.freeze()


def save_config(cfg, path):
    with open(path, 'w') as f:
        f.write(cfg.dump())


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

