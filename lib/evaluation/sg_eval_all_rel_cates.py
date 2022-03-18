"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import numpy as np
from functools import reduce
from lib.pytorch_misc import intersect_2d, argsort_desc
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from config import MODES
import sys
np.set_printoptions(precision=3)

class BasicSceneGraphEvaluator:
    def __init__(self, mode, multiple_preds=False):
        self.result_dict = {}
        self.mode = mode
        rel_cats = {
            0: 'all_rel_cates',
            1: "above",
            2: "across",
            3: "against",
            4: "along",
            5: "and",
            6: "at",
            7: "attached to",
            8: "behind",
            9: "belonging to",
            10: "between",
            11: "carrying",
            12: "covered in",
            13: "covering",
            14: "eating",
            15: "flying in",
            16: "for",
            17: "from",
            18: "growing on",
            19: "hanging from",
            20: "has",
            21: "holding",
            22: "in",
            23: "in front of",
            24: "laying on",
            25: "looking at",
            26: "lying on",
            27: "made of",
            28: "mounted on",
            29: "near",
            30: "of",
            31: "on",
            32: "on back of",
            33: "over",
            34: "painted on",
            35: "parked on",
            36: "part of",
            37: "playing",
            38: "riding",
            39: "says",
            40: "sitting on",
            41: "standing on",
            42: "to",
            43: "under",
            44: "using",
            45: "walking in",
            46: "walking on",
            47: "watching",
            48: "wearing",
            49: "wears",
            50: "with"
        }
        self.rel_cats = rel_cats
        self.result_dict[self.mode + '_recall'] = {20: {}, 50: {}, 100: []}
        for key, value in self.result_dict[self.mode + '_recall'].items():
            self.result_dict[self.mode + '_recall'][key] = {}
            for rel_cat_id, rel_cat_name in rel_cats.items():
                self.result_dict[self.mode + '_recall'][key][rel_cat_name] = []
        self.multiple_preds = multiple_preds

    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in MODES}
        return evaluators

    @classmethod
    def vrd_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, multiple_preds=True, **kwargs) for m in ('preddet', 'phrdet')}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_scores, viz_dict=None, iou_thresh=0.5):
        res = evaluate_from_dict(gt_entry, pred_scores, self.mode, self.result_dict,
                                  viz_dict=viz_dict, iou_thresh=iou_thresh, multiple_preds=self.multiple_preds, rel_cats=self.rel_cats)
        # self.print_stats()
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            for rel_cat_id, rel_cat_name in self.rel_cats.items():
                print('R@%i: %f' % (k, np.mean(v[rel_cat_name])), rel_cat_name)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, multiple_preds=False,
                       viz_dict=None, rel_cats=None, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param mode: 'det' or 'cls'
    :param result_dict:
    :param viz_dict:
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    gt_rels_nums = [0 for x in range(len(rel_cats))]
    for rel in gt_rels:
        gt_rels_nums[rel[2]] += 1
        gt_rels_nums[0] += 1


    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']

    if mode == 'predcls':
        pred_boxes = gt_boxes
        pred_classes = gt_classes
        obj_scores = np.ones(gt_classes.shape[0])
    elif mode == 'sgcls':
        pred_boxes = gt_boxes
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
    elif mode == 'sgdet' or mode == 'phrdet':
        pred_boxes = pred_entry['pred_boxes'].astype(float)
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k]