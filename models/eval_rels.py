
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os

conf = ModelConfig()
from lib.NODIS import NODIS

train, val, test = VG.splits(num_val_im=0, filter_duplicate_rels=True,
                             use_proposals=conf.use_proposals,
                             filter_non_overlap=conf.mode == 'sgdet')
if conf.test:
    val = test

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector = NODIS(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                 num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                 use_resnet=conf.use_resnet, order=conf.order,
                 use_proposals=conf.use_proposals)


detector.cuda()
ckpt = torch.load(conf.ckpt)
conf_matrix = np.zeros((3,1))
optimistic_restore(detector, ckpt['state_dict'])

all_pred_entries = []


def val_batch(batch_num, b, evaluator, thrs=(20, 50, 100)):

    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)
        # assert np.all(rels_i[:,2.0] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,
        }
        all_pred_entries.append(pred_entry)

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )


evaluator = BasicSc