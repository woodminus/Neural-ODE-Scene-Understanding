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
            13