from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import timedelta
from pathlib import Path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import os
import logging
import shutil
import time

import numpy as np
from PIL import Image
from torchvision import transforms,models
from model import augmentNet

from .comm import comm
import os


def setup_cudnn(config):
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

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

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    dataset = cfg.DATASET.DATASET
    cfg_name = cfg.NAME

    final_output_dir = root_output_dir / dataset / cfg_name

    root_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(final_output_dir, cfg.RANK, phase)

    return str(final_output_dir)

def setup_logger(final_output_dir, rank, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_rank{}.txt'.format(phase, time_str, rank)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s:[P:%(process)d]:' + comm.head + ' %(message)s'
    logging.basicConfig(
        filename=str(final_log_file), format=head
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(head)
    )
    logging.getLogger('').addHandler(console)

def load_image(image, img_size=None):
    image = Image.fromarray(np.uint8(image))

    if img_size is not None:
        image = image.resize((img_size, img_size))  # change image size to (3, img_size, img_size)

    transform = transforms.Compose([
        # convert the (H x W x C) PIL image in the range(0, 255) into (C x H x W) tensor in the range(0.0, 1.0)
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # this is from ImageNet dataset
    ])

    # change image's size to (b, 3, h, w)
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image

def get_image_from_path(path):
    dirs = os.listdir(path)
    cls = 0
    img_list = []
    cls_list = []
    for dir in dirs:
        files = os.listdir(path + "/" + dir)
        for file in files:
            img = Image.open(path + "/" + dir + "/" + file)
            img_arr = np.asarray(img)
            img_list.append(img_arr)
            cls_list.append(cls)
        cls += 1
    return img_list, cls_list

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '2': 'conv1_2',
                  '5': 'conv2_1',
                  '7': 'conv2_2',
                  '10': 'conv3_1',
                  '16': 'conv3_2',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '25': 'conv4_3',
                  '28': 'conv5_1',
                  '30': 'conv5_2',
                  '34': 'conv5_3',
                  }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
           features[layers[name]] = x
        features[name] = x
    return features


def get_grim_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    gram_matrix = torch.mm(tensor, tensor.t())
    return gram_matrix

def img_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)

    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    image = image * 255
    return image

def style_transfer(image, style, style_net, content_net, content_weight, style_weight, content_layer_weight, style_layer_weight, device="cuda"):
    content_image = load_image(image)
    content_image = content_image.to(device)
    content_features = get_features(content_image, content_net)

    target = content_image.clone().requires_grad_(True).to(device)

    optimizer = optim.Adam(style_net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    steps = 3

    content_loss_epoch = []
    style_loss_epoch = []
    total_loss_epoch = []

    for epoch in range(0, steps+1):

        target, sum_var = style_net(content_image)
        target.to(device)
        target.requires_grad_(True)

        target_features = get_features(target, content_net)  # extract output image's all feature maps

        content_loss = 0
        for layer in content_layer_weight:
            if content_layer_weight[layer] == 0:
                continue
            layer_style_loss = content_layer_weight[layer] * torch.mean(
                (target_features[layer] - content_features[layer]) ** 2)
            content_loss += layer_style_loss

        style_loss = 0
        for layer in style_layer_weight:
            target_feature = target_features[layer]  # output image's feature map after layer
            target_gram_matrix = get_grim_matrix(target_feature)
            style_gram_matrix = style[layer]

            layer_style_loss = style_layer_weight[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
            b, c, h, w = target_feature.shape
            style_loss += layer_style_loss / (c * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss_epoch.append(total_loss)

        style_loss_epoch.append(style_weight * style_loss)
        content_loss_epoch.append(content_weight * content_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        output_image = target

    return img_convert(output_image)

def data_augment(content_path, style_paths, pretrain_paths=[], device="cuda", content_layer_weight=None, style_layer_weight=None):

    if style_layer_weight == None:
        style_layer_weight = {'conv1_1': 0.1,
                             'conv2_1': 0.2,
                             'conv3_1': 0.4,
                             'conv4_1': 0.8,
                             'conv5_1': 1.6,
                             'conv5_3': 0}

    if content_layer_weight == None:
        content_layer_weight = {'conv1_2': 0.0,
                           'conv2_2': 0.0,
                           'conv3_2': 0.0,
                           'conv4_2': 0.8,
                           'conv4_3': 0.0,
                           'conv5_2': 0.2,
                           'conv5_3': 0.0}

    content_image_list, cls_list = get_image_from_path(content_path)
    ori_len = len(content_image_list)

    VGG = models.vgg19(pretrained=True).features
    VGG.to(device)

    for parameter in VGG.parameters():
        parameter.requires_grad_(False)

    if len(pretrain_paths) == 0:
        pretrain_paths = [None]*len(style_paths)

    for style_path, pretrain_path in zip(style_paths, pretrain_paths):
        style_image = Image.open(style_path)
        style_image = np.asarray(style_image)
        style_image = load_image(style_image, img_size=256)  # temporary/style.png
        style_image = style_image.to(device)
        style_features = get_features(style_image, VGG)
        style_gram_matrixs = {layer: get_grim_matrix(style_features[layer]) for layer in style_features}

        style_net = augmentNet.AuNet()
        if pretrain_path != None:
            model_CKPT = torch.load(pretrain_path)
            style_net.load_state_dict(model_CKPT["state_dict"])
        style_net.to(device)

        for i in range(ori_len):
            img = style_transfer(content_image_list[i], style_gram_matrixs, style_net, VGG, 50, 1, content_layer_weight, style_layer_weight)
            content_image_list.append(img)
            cls_list.append(cls_list[i])

    return content_image_list, cls_list

def getDataLoader(path, config, isAug=False, istrain=True):
    img_list = None
    cls_list = None
    if isAug is False:
        img_list, cls_list = get_image_from_path(path)
    else:
        img_list, cls_list = data_augment(path, config.AUG.PATH, config.AUG.PRETRAIN if config.AUG.PRETRAIN!="" else None)

    imgs = torch.FloatTensor(img_list)
    clss = torch.LongTensor(cls_list)
    dataset = TensorDataset(imgs, clss)

    loader = DataLoader(dataset,
                              batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
                              num_workers=config.WORKERS,
                              pin_memory=config.PIN_MEMORY,
                              drop_last=True if istrain else False,
                              shuffle=True)
    return loader

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    if isinstance(output, list):
        output = output[-1]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for p, t in zip(pred[:, :1].reshape(-1), target):
        if t == 0:
            if p == t:
                TN += 1
            else:
                FN += 1
        else:
            if p != 0:
                TP += 1
            else:
                FP += 1
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res,[TP,TN,FP,FN]

def save_checkpoint_on_master(model,
                              *,
                              optimizer,
                              output_dir,
                              in_epoch,
                              epoch_or_step,
                              best_perf):
    states = model.state_dict()

    save_dict = {
        'epoch' if in_epoch else 'step': epoch_or_step + 1,
        'state_dict': states,
        'perf': best_perf,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(save_dict, os.path.join(output_dir, 'checkpoint.pth'))



def save_model_on_master(model, distributed, out_dir, fname):
    fname_full = os.path.join(out_dir, fname)
    torch.save(
        model.module.state_dict() if distributed else model.state_dict(),
        fname_full
    )


