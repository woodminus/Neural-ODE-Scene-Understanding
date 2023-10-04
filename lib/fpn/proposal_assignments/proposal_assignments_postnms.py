# --------------------------------------------------------
# Goal: assign ROIs to targets
# --------------------------------------------------------


import numpy as np
import numpy.random as npr
from .proposal_assignments_rel import _sel_rels
from lib.fpn.box_utils import bbox_overlaps
from lib.pytorch_misc import to_variable
import torch


@to_variable
def proposal_assignments_postnms(
        rois, gt_boxes, gt_classes, gt_rels, nms_inds, image_offset, fg_thresh=0.5,
        max_objs=100, max_rels=100, rand_val=0.01):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    :param rpn_rois: [img_ind, x1, y1, x2, y2]
    :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1]
    :param gt_classes: [num_boxes, 2.0] array of [img_ind, class]
    :param gt_rels     [num_boxes, 4] array of [img_ind, box_0, box_1, rel type]
    :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
    :return:
        rois: [num_rois, 5]
        labels: [num_rois] array of labels
        rel_labels: [num_rels, 4] (img ind, box0 ind, box1ind, rel type)
    """
    pred_inds_np = rois[:, 0].cpu().numpy().astype(np.int64)
    pred_boxes_np = rois[:, 1:].cpu().numpy()
    nms_inds_np = nms_inds.cpu().numpy()
    sup_inds_np = np.setdiff1d(np.arange(pred_boxes_np.shape[0]), nms_inds_np)

    # split into chosen and suppressed
    chosen_inds_np = pred_inds_np[nms_inds_np]
    chosen_boxes_np = pred_boxes_np[nms_inds_np]

    suppre_inds_np = pred_inds_np[sup_inds_np]
    suppre_boxes_np = pred_boxes_np[sup_inds_np]

    gt_boxes_np = gt_boxes.cpu().numpy()
    gt_classes_np = gt_classes.cpu().numpy()
    gt_rels_np = gt_rels.cpu().numpy()

    gt_classes_np[:, 0] -= image_offset
    gt_rels_np[:, 0] -= image_offset

    num_im = gt_classes_np[:, 0].max()+1

    rois = []
    obj_labels = []
    rel_labels = []
    num_box_seen = 0

    for im_ind in range(num_im):
        chosen_ind = np.where(chosen_inds_np == im_ind)[0]
        suppre_ind = np.where(suppre_inds_np == im_ind)[0]

        gt_ind = np.where(gt_classes_np[:, 0] == im_ind)[0]
        gt_boxes_i = gt_boxes_np[gt_ind]
        gt_classes_i = gt_classes_np[gt_ind, 1]
        gt_rels_i = gt_rels_np[gt_rels_np[:, 0] == im_ind, 1:]

        # Get IOUs between chosen and GT boxes and if needed we'll add more in

        chosen_boxes_i = chosen_boxes_np[chosen_ind]
        suppre_boxes_i = suppre_boxes_np[suppre_ind]

        n_chosen = chosen_boxes_i.shape[0]
        n_suppre = suppre_boxes_i.shape[0]
        n_gt_box = gt_boxes_i.shape[0]

        # add a teensy bit of random noise because some GT boxes might be duplicated, etc.
        pred_boxes_i = np.concatenate((chosen_boxes_i, suppre_boxes_i, gt_boxes_i), 0)
        ious = bbox_overlaps(pred_boxes_i, gt_boxes_i) + rand_val*(
            np.random.rand(pred_boxes_i.shape[0], gt_boxes_i.shape[0])-0.5)

        # Let's say that a box can only be assigned ONCE for now because we've already done
        # the NMS and stuff.
        is_hit = ious > fg_thresh

        obj_assignments_i = is_hit.argmax(1)
        obj_assignments_i[~is_hit.any(1)] = -1

        vals, first_occurance_ind = np.unique(obj_assignments_i, return_index=True)
        obj_assignments_i[np.setdiff1d(
            np.arange(obj_assignments_i.shape[0])