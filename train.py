from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast

from timm.scheduler import create_scheduler

from einops import rearrange

from config import config
from config import update_config
from config import save_config

from libs.comm import comm
from libs.utils import create_logger, getDataLoader
from libs.utils import setup_cudnn
from libs.utils import set_wd
from libs.utils import accuracy

from model import get_backbone


def main():
    args = parse_args()
    print(args)
    setup_cudnn(config)

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'train')
    tb_log_dir = final_output_dir
    if comm.is_main_process():
        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        save_config(config, output_config_path)

    model = get_backbone(config)
    if args.pretrain_model != "":
        model_CKPT = torch.load(args.pretrained_model)
        model.load_state_dict(model_CKPT)
    model.to(torch.device('cuda'))
    model.to(memory_format=torch.channels_last)

    best_perf = 0.0
    best_model = True
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = optim.AdamW(
        set_wd(config, model),
        lr=config.TRAIN.LR,
        weight_decay=config.TRAIN.WD,
    )

    train_loader = getDataLoader(config.DATASET.ROOT + "/train", config, True)
    valid_loader = getDataLoader(config.DATASET.ROOT + "/val", config, False)
    print("data load finished")

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.L1Loss()
    criterion.cuda()
    criterion_eval = nn.CrossEntropyLoss()
    # criterion_eval = nn.L1Loss()
    criterion_eval.cuda()

    lr_param = config.TRAIN.LR_SCHEDULER.ARGS
    lr_scheduler, _ = create_scheduler(lr_param, optimizer)
    lr_scheduler.step(begin_epoch)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("train start")
    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):

        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            train_one_epoch(config, train_loader, model, criterion, optimizer,
                            epoch, scaler=scaler)

        if epoch >= config.TRAIN.EVAL_BEGIN_EPOCH:
            perf = test(valid_loader, model, criterion_eval)

            best_model = (perf > best_perf)
            best_perf = perf if best_model else best_perf

        lr_scheduler.step(epoch=epoch + 1)

        torch.save(model, os.path.join(config.OUTPUT_DIR, 'checkpoint.pth'))

        if best_model:
            torch.save(model, os.path.join(config.OUTPUT_DIR, 'best.pth'))

    torch.save(model, os.path.join(config.OUTPUT_DIR, 'final.pth'))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch, scaler=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.train()

    for i, (x, y) in enumerate(train_loader):
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        print(x.shape)
        print(y.shape)
        with autocast(enabled=config.AMP.ENABLED):
            # x = x.contiguous(memory_format=torch.channels_last)
            # y = y.contiguous(memory_format=torch.channels_last)

            outputs = model(x)
            loss = criterion(outputs, y)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        scaler.scale(loss).backward(create_graph=is_second_order)
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), x.size(0))

        prec_res, res_detail = accuracy(outputs, y, (1, 3))

        prec1 = prec_res[0]
        prec3 = prec_res[1]

        top1.update(prec1, x.size(0))
        top3.update(prec3, x.size(0))

        msg = '=> Epoch[{0}][{1}/{2}]: ' \
              'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
              'Accuracy@3 {top3.val:.3f} ({top3.avg:.3f})\t'.format(
            epoch, i, len(train_loader), top1=top1, top3=top3)
        print(msg)


@torch.no_grad()
def test(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.eval()

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i, (x, y) in enumerate(val_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        outputs = model(x)

        loss = criterion(outputs, y)
        losses.update(loss.item(), x.size(0))

        prec_res, res_detail = accuracy(outputs, y, (1, 3))
        TN += res_detail[0]
        TP += res_detail[1]
        FN += res_detail[2]
        FP += res_detail[3]

        prec1 = prec_res[0]
        prec3 = prec_res[1]

        top1.update(prec1, x.size(0))
        top3.update(prec3, x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    top1_acc, top3_acc, loss_avg = map(lambda x: x.avg, [top1, top3, losses])

    msg = '=> TEST:\t' \
          'Loss {loss_avg:.4f}\t' \
          'Error@1 {error1:.3f}%\t' \
          'Error@3 {error3:.3f}%\t' \
          'Accuracy@1 {top1:.3f}%\t' \
          'Accuracy@3 {top3:.3f}%\t' \
          'TP {TP}%\t' \
          'TN {TN}%\t' \
          'FP {FP}%\t' \
          'FN {FN}%\t'.format(
        loss_avg=loss_avg, top1=top1_acc,
        top3=top3_acc, error1=100 - top1_acc,
        error3=100 - top3_acc, TP=TP, FP=FP, TN=TN, FN=FN
    )
    print(msg)
    model.train()
    return top1_acc

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument("--pretrain_model", type=str, default="")

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
