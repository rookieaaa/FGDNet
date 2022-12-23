from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.optim as optim

def set_wd(cfg, model):
    without_decay_list = cfg.TRAIN.WITHOUT_WD_LIST
    without_decay_depthwise = []
    without_decay_norm = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)
        elif isinstance(m, nn.LayerNorm):
            without_decay_norm.append(m.weight)
            without_decay_norm.append(m.bias)
    with_decay = []
    without_decay = []

    skip = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()

    for n, p in model.named_parameters():
        ever_set = False

        if p.requires_grad is False:
            continue

        skip_flag = False
        if n in skip:
            without_decay.append(p)
            skip_flag = True
        else:
            for i in skip:
                if i in n:
                    without_decay.append(p)
                    skip_flag = True

        if skip_flag:
            continue

        for pp in without_decay_depthwise:
            if p is pp:
                without_decay.append(p)
                ever_set = True
                break

        for pp in without_decay_norm:
            if p is pp:
                without_decay.append(p)
                ever_set = True
                break

        if (
            (not ever_set)
            and 'bias' in without_decay_list
            and n.endswith('.bias')
        ):
            without_decay.append(p)
        elif not ever_set:
            with_decay.append(p)

    params = [
        {'params': with_decay},
        {'params': without_decay, 'weight_decay': 0.}
    ]
    return params


def build_optimizer(cfg, model):
    params = set_wd(cfg, model)
    return optim.AdamW(
        params,
        lr=cfg.TRAIN.LR,
        weight_decay=cfg.TRAIN.WD,
    )
    return optimizer

