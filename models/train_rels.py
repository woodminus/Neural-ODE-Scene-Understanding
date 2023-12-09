from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
from torch import optim
import torch
import torch.nn as nn
import pandas as pd
import time
import os

from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau

conf = ModelConfig()
from lib.NODIS import NODIS

train, val, test = VG.splits(num_val_im=5000, filter_duplicate_rels=True,
                             use_proposals=conf.use_proposals,
                            filter_non_overlap=conf.mode == 'sgdet')

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector = NODIS(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                 num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                 use_resnet=conf.use_resnet, order=conf.order,
                 use_proposals=conf.use_proposals)

# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False

print(print_para(detector), flush=True)

def get_optim(lr):

    fc_params = [p for n,p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n,p in detector.named_parameters() if not n.startswith('roi_fmap') and not n.startswith('detector.features')
                     and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]

    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.l2, lr=1e-4, eps=1e-3) #1e-4
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler

ckpt = torch.load(conf.ckpt)
if conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
        # optimistic_restore(detector.detector, torch.load('checkpoints/vgdet/vg-28.tar')['state_dict'])
else:
    start_epoch = -1
    optimistic_restore(detector.detector, ckpt['state_dict'])

    detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

detector.cuda()

def train_epoch(epoch_num):

    detector.train()
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):

        tr.append(train_batch(batch, verbose=b % (conf.print_interval*10) == 0)) #b == 0))
        #F.interpolate(F.interpolate(batch.imgs, scale_factor=0.25), scale_factor=4)
        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)

            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batc