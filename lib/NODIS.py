
"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM
from lib.fpn.nms.functions.nms import apply_nms

from lib.fpn.box_utils import bbox_preds, center_size, bbox_overlaps
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, \
    Flattener
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
import math
from lib.ODE import odeBlock, odeFunc1, odeFunc2
import time

def filter_det(scores, boxes, start_ind=0, max_per_img=100, thresh=0.001, pre_nms_topn=6000,
               post_nms_topn=300, nms_thresh=0.3, nms_filter_duplicates=True):
    """
    Filters the detections for a single image
    :param scores: [num_rois, num_classes]
    :param boxes: [num_rois, num_classes, 4]. Assumes the boxes have been clamped
    :param max_per_img: Max detections per image
    :param thresh: Threshold for calling it a good box
    :param nms_filter_duplicates: True if we shouldn't allow for mulitple detections of the
           same box (with different labels)
    :return: A numpy concatenated array with up to 100 detections/img [num_im, x1, y1, x2, y2, score, cls]
    """

    valid_cls = (scores[:, 1:].data.max(0)[0] > thresh).nonzero() + 1
    if valid_cls.dim() == 0:
        return None

    nms_mask = scores.data.clone()
    nms_mask.zero_()

    for c_i in valid_cls.squeeze(1).cpu():
        scores_ci = scores.data[:, c_i]
        boxes_ci = boxes.data[:, c_i]
        keep = apply_nms(scores_ci, boxes_ci,
                         pre_nms_topn=pre_nms_topn, post_nms_topn=post_nms_topn,
                         nms_thresh=nms_thresh)
        nms_mask[:, c_i][keep] = 1

    #dists_all = Variable(nms_mask * scores.data, volatile=True)
    with torch.no_grad():
        dists_all = Variable(nms_mask * scores.data)

    if nms_filter_duplicates:
        scores_pre, labels_pre = dists_all.data.max(1)
        inds_all = scores_pre.nonzero()
        assert inds_all.dim() != 0
        inds_all = inds_all.squeeze(1)

        labels_all = labels_pre[inds_all]
        scores_all = scores_pre[inds_all]
    else:
        nz = nms_mask.nonzero()
        assert nz.dim() != 0
        inds_all = nz[:, 0]
        labels_all = nz[:, 1]
        scores_all = scores.data.view(-1)[inds_all * scores.data.size(1) + labels_all]

    # dists_all = dists_all[inds_all]
    # dists_all[:,0] = 1.0-dists_all.sum(1)

    # # Limit to max per image detections
    vs, idx = torch.sort(scores_all, dim=0, descending=True)
    idx = idx[vs > thresh]
    if max_per_img < idx.size(0):
        idx = idx[:max_per_img]

    inds_all = inds_all[idx] + start_ind
    #!!!!scores_all = Variable(scores_all[idx], volatile=True)
    #!!!!labels_all = Variable(labels_all[idx], volatile=True)
    with torch.no_grad():
        scores_all = Variable(scores_all[idx])
        labels_all = Variable(labels_all[idx])
    # dists_all = dists_all[idx]

    return inds_all, scores_all, labels_all

def _sort_by_score(im_inds, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_im = im_inds[-1].item() + 1
    rois_per_image = scores.new(num_im)  ###
    lengths = []
    for i, s, e in enumerate_by_image(im_inds):  # i img_index s:start_idx e:end_idx
        rois_per_image[i] = 2 * (s - e) * num_im + i
        lengths.append(e - s)
    lengths = sorted(lengths, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(lengths)  # move it to TxB form
    inds = torch.LongTensor(inds).cuda(im_inds.get_device())

    # ~~~~~~~~~~~~~~~~
    # HACKY CODE ALERT!!!
    # we're sorting by confidence which is in the range (0,1), but more importantly by longest
    # img....
    # ~~~~~~~~~~~~~~~~
    roi_order = scores - 2 * rois_per_image[im_inds]
    _, perm = torch.sort(roi_order, 0, descending=True)
    perm = perm[inds]
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed


MODES = ('sgdet', 'sgcls', 'predcls')


class O_NODE(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, classes, rel_classes, mode='sgdet',
                 embed_dim=200, obj_dim=2048, order='confidence'):
        super(O_NODE, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode

        self.embed_dim = embed_dim

        self.obj_dim = obj_dim

        #----------add sget nms
        self.nms_filter_duplicates = True
        self.max_per_img =64#default 64
        self.thresh = 0.01#############0.001training 0.01test

        assert order in ('size', 'confidence', 'random', 'leftright')
        self.order = order

        # EMBEDDINGS
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()


        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])

        self.decoder_lin1 = nn.Linear(self.obj_dim + self.embed_dim + 128, 1024)
        self.decoder_lin2 = odeBlock(odeFunc2(use_cuda=True))
        self.decoder_lin3 = nn.Linear(1024, self.num_classes)

    def sort_rois(self, batch_idx, confidence, box_priors):
        """
        :param batch_idx: tensor with what index we're on
        :param confidence: tensor with confidences between [0,1)
        :param boxes: tensor with (x1, y1, x2, y2)
        :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
        """
        cxcywh = center_size(box_priors)
        if self.order == 'size':
            sizes = cxcywh[:, 2] * cxcywh[:, 3]
            # sizes = (box_priors[:, 2.0] - box_priors[:, 0] + 1) * (box_priors[:, 3] - box_priors[:, 1] + 1)
            assert sizes.min() > 0.0
            scores = sizes / (sizes.max() + 1)
        elif self.order == 'confidence':
            scores = confidence
        elif self.order == 'random':
            scores = torch.FloatTensor(np.random.rand(batch_idx.size(0))).cuda(batch_idx.get_device())
        elif self.order == 'leftright':
            centers = cxcywh[:, 0]
            scores = centers / (centers.max() + 1)
        else:
            raise ValueError("invalid mode {}".format(self.order))
        return _sort_by_score(batch_idx, scores)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def nms_boxes(self, obj_dists, rois, box_deltas, im_sizes):
        """
        Performs NMS on the boxes
        :param obj_dists: [#rois, #classes]
        :param rois: [#rois, 5]
        :param box_deltas: [#rois, #classes, 4]
        :param im_sizes: sizes of images
        :return
            nms_inds [#nms]
            nms_scores [#nms]
            nms_labels [#nms]
            nms_boxes_assign [#nms, 4]
            nms_boxes  [#nms, #classes, 4]. classid=0 is the box prior.
        """
        # Now produce the boxes
        # box deltas is (num_rois, num_classes, 4) but rois is only #(num_rois, 4)
        boxes = bbox_preds(rois[:, None, 1:].expand_as(box_deltas).contiguous().view(-1, 4),
                           box_deltas.view(-1, 4)).view(*box_deltas.size())

        # Clip the boxes and get the best N dets per image.
        inds = rois[:, 0].long().contiguous()
        dets = []

        for i, s, e in enumerate_by_image(inds.data):
            h, w = im_sizes[i, :2]
            boxes[s:e, :, 0].data.clamp_(min=0, max=w - 1)
            boxes[s:e, :, 1].data.clamp_(min=0, max=h - 1)
            boxes[s:e, :, 2].data.clamp_(min=0, max=w - 1)
            boxes[s:e, :, 3].data.clamp_(min=0, max=h - 1)

            d_filtered = filter_det(
                F.softmax(obj_dists[s:e], 1), boxes[s:e], start_ind=s,
                nms_filter_duplicates=self.nms_filter_duplicates,
                max_per_img=self.max_per_img,
                thresh=self.thresh,
            )

            if d_filtered is not None:
                dets.append(d_filtered)
        if len(dets) == 0:
            print("nothing was detected", flush=True)
            return None
        nms_inds, nms_scores, nms_labels = [torch.cat(x, 0) for x in zip(*dets)]
        twod_inds = nms_inds * boxes.size(1) + nms_labels.data
        nms_boxes_assign = boxes.view(-1, 4)[twod_inds]

        nms_boxes = torch.cat((rois[:, 1:][nms_inds][:, None], boxes[nms_inds][:, 1:]), 1)
        return nms_inds, nms_scores, nms_labels, nms_boxes_assign, nms_boxes, inds[nms_inds]


    def forward(self, obj_fmaps, obj_logits, im_inds, obj_labels=None, box_priors=None, boxes_per_cls=None, batch_size=None,
                rois=None, od_box_deltas=None, im_sizes=None, image_offset=None, gt_classes=None, gt_boxes=None, ):
        """
        Forward pass through the object and edge context
        :param obj_priors:
        :param obj_fmaps:
        :param im_inds:
        :param obj_labels:
        :param boxes:
        :return:
        """
        obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight
        pos_embed = self.pos_embed(Variable(center_size(box_priors)))
        obj_pre_rep = torch.cat((obj_fmaps, obj_embed, pos_embed), 1)


        if self.mode == 'predcls':
            obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_classes))
        else:
            if self.mode == 'sgcls':

                obj_dists2 = self.decoder_lin1(obj_pre_rep)
                obj_dists2 = self.decoder_lin2(obj_dists2.view(-1, 1, 1024), 1)

                obj_dists2 = obj_dists2[1]

                obj_dists2 = self.decoder_lin3(obj_dists2.view(-1, 1024))

            else:
                # this is for sgdet

                obj_dists2 = self.decoder_lin1(obj_pre_rep)

                perm, inv_perm, ls_transposed = self.sort_rois(im_inds.data, None, box_priors)
                obj_dists2 = obj_dists2[perm].contiguous()
                obj_dists2 = PackedSequence(obj_dists2, torch.tensor(ls_transposed))
                obj_dists2, lengths1 = pad_packed_sequence(obj_dists2, batch_first=False)


                obj_dists2 = self.decoder_lin2(obj_dists2.view(-1, batch_size, 1024), batch_size)[1]


                obj_dists2, _ = pack_padded_sequence(obj_dists2, lengths1, batch_first=False)
                obj_dists2 = self.decoder_lin3(obj_dists2.view(-1, 1024))
                obj_dists2 = obj_dists2[inv_perm]


                if (not self.training and not self.mode == 'gtbox') or self.mode in ('sgdet', 'refinerels'):
                    # try: dont apply nms here, but after own obj_classifier
                    nms_inds, nms_scores, nms_preds, nms_boxes_assign, nms_boxes, nms_imgs = self.nms_boxes(
                        obj_dists2.clone().detach(),
                        rois,
                        od_box_deltas.clone().detach(), im_sizes,
                    )
                    im_inds = nms_imgs + image_offset
                    obj_dists2 = obj_dists2[nms_inds]
                    obj_fmap = obj_fmaps[nms_inds]
                    box_deltas = od_box_deltas[nms_inds]
                    box_priors = nms_boxes[:, 0]
                    rois = rois[nms_inds]

                    if self.training and not self.mode == 'gtbox':
                        # NOTE: If we're doing this during training, we need to assign labels here.
                        pred_to_gtbox = bbox_overlaps(box_priors, gt_boxes).data
                        pred_to_gtbox[im_inds.data[:, None] != gt_classes.data[None, :, 0]] = 0.0

                        max_overlaps, argmax_overlaps = pred_to_gtbox.max(1)
                        rm_obj_labels = gt_classes[:, 1][argmax_overlaps]
                        rm_obj_labels[max_overlaps < 0.5] = 0
                    else:
                        rm_obj_labels = None

        if self.mode == 'sgdet' and not self.training:  # have tried in training
            # NMS here for baseline

            probs = F.softmax(obj_dists2, 1)
            nms_mask = obj_dists2.data.clone()
            nms_mask.zero_()
            for c_i in range(1, obj_dists2.size(1)):
                scores_ci = probs.data[:, c_i]
                boxes_ci = nms_boxes.data[:, c_i]

                keep = apply_nms(scores_ci, boxes_ci,
                                 pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                                 nms_thresh=0.5)#nms_thresh= 0.3 default
                nms_mask[:, c_i][keep] = 1

            obj_preds = Variable(nms_mask * probs.data, volatile=True)[:, 1:].max(1)[1] + 1  # this for sgdet test

            #obj_preds=obj_dists2[:,1:].max(1)[1] + 1
        else:
            if self.mode == 'sgdet':
                # use gt
                obj_preds = rm_obj_labels if rm_obj_labels is not None else obj_dists2[:, 1:].max(1)[1] + 1
                # use_predicted label
                # obj_preds = obj_dists2[:, 1:].max(1)[1] + 1
            else:
                obj_preds = obj_labels if obj_labels is not None else obj_dists2[:, 1:].max(1)[1] + 1

        if self.mode == 'sgdet':
            return obj_dists2, obj_preds, im_inds, box_priors, rm_obj_labels, rois, nms_boxes
        else:
            return obj_dists2, obj_preds



class NODIS(nn.Module):
    """
    RELATIONSHIPS
    """

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, require_overlap_det=True,
                 embed_dim=200, use_resnet=False, order='confidence', thresh=0.01, use_proposals=False):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(NODIS, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        self.pooling_size = 7
        self.embed_dim = embed_dim

        self.obj_dim = 2048 if use_resnet else 4096


        self.order = 'random'

        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
        )

        self.context = O_NODE(self.classes, self.rel_classes, mode=self.mode, embed_dim=self.embed_dim, obj_dim=self.obj_dim, order=order)

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512)

        if use_resnet:
            self.roi_fmap = nn.Sequential(
                resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else:
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier
            self.roi_avg_pool = nn.AvgPool2d(kernel_size=7, stride=0)
        ###################################
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed2.weight.data = embed_vecs.clone()

        self.lstm_visual = nn.LSTM(input_size=1536, hidden_size=512)
        self.lstm_semantic = nn.LSTM(input_size=400, hidden_size=512)
        self.odeBlock = odeBlock(odeFunc1(bidirectional=True))

        self.fc_predicate = nn.Sequential(nn.Linear(1024, 512),
                                          nn.ReLU(inplace=False),
                                          nn.Linear(512, 51),
                                          nn.ReLU(inplace=False))

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def init_hidden(self, batch_size, bidirectional):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if bidirectional:
            return (torch.zeros(2, batch_size, 256).cuda(),
                    torch.zeros(2, batch_size, 256).cuda())
        else:
            return (torch.zeros(1, batch_size, 512).cuda(),
                    torch.zeros(1, batch_size, 512).cuda())

    def sort_rois(self, batch_idx, confidence, box_priors):
        """
        :param batch_idx: tensor with what index we're on
        :param confidence: tensor with confidences between [0,1)
        :param boxes: tensor with (x1, y1, x2, y2)
        :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
        """
        cxcywh = center_size(box_priors)
        if self.order == 'size':
            sizes = cxcywh[:, 2] * cxcywh[:, 3]
            # sizes = (box_priors[:, 2.0] - box_priors[:, 0] + 1) * (box_priors[:, 3] - box_priors[:, 1] + 1)
            assert sizes.min() > 0.0
            scores = sizes / (sizes.max() + 1)
        elif self.order == 'confidence':
            scores = confidence
        elif self.order == 'random':
            scores = torch.FloatTensor(np.random.rand(batch_idx.size(0))).cuda(batch_idx.get_device())
        elif self.order == 'leftright':
            centers = cxcywh[:, 0]
            scores = centers / (centers.max() + 1)
        else:
            raise ValueError("invalid mode {}".format(self.order))
        return _sort_by_score(batch_idx, scores)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes, union_rois = self.union_boxes(features, rois, pair_inds)
        return self.roi_avg_pool(uboxes), union_rois

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def obj_avg_pool(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_avg_pool(feature_pool)

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2.0] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2.0] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels

            if test:
            prob dists, boxes, img inds, maxscores, classes

        """
        '---------gt_rel process----------'

        batch_size = x.shape[0]

        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)
        if result.is_none():
            return ValueError("heck")


        # Prevent gradients from flowing back into score_fc from elsewhere   the last 3 is bullshit
        if self.mode == 'sgdet':
            im_inds = result.im_inds - image_offset  # all indices
            boxes = result.od_box_priors  # all boxes
            rois = torch.cat((im_inds[:, None].float(), boxes), 1)
            result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)
            result.rm_obj_dists, result.obj_preds,  im_inds, result.rm_box_priors, result.rm_obj_labels, rois, result.boxes_all, \
                = self.context(
                result.obj_fmap,
                result.rm_obj_dists.detach(),
                im_inds, result.rm_obj_labels if self.training or self.mode == 'predcls' else None,
                boxes.data, result.boxes_all, batch_size,
                rois, result.od_box_deltas.detach(), im_sizes, image_offset, gt_classes, gt_boxes)

            boxes = result.rm_box_priors
        else:
            #sgcls and predcls
            im_inds = result.im_inds - image_offset
            boxes = result.rm_box_priors
            rois = torch.cat((im_inds[:, None].float(), boxes), 1)
            result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)

            result.rm_obj_dists, result.obj_preds = self.context(
                result.obj_fmap,
                result.rm_obj_dists.detach(),
                im_inds, result.rm_obj_labels if self.training or self.mode == 'predcls' else None,
                boxes.data, result.boxes_all, batch_size,
                None, None, None, None, None, None)

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)

        # visual part
        obj_pooling = self.obj_avg_pool(result.fmap.detach(), rois).view(-1, 512)
        subj_rep = obj_pooling[rel_inds[:, 1]]
        obj_rep = obj_pooling[rel_inds[:, 2]]
        vr, union_rois = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:])
        vr = vr.view(-1, 512)
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
        # semantic part
        subj_class = result.obj_preds[rel_inds[:, 1]]
        obj_class = result.obj_preds[rel_inds[:, 2]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        # padding
        perm, inv_perm, ls_transposed = self.sort_rois(rel_inds[:, 0].data, None, union_rois[:, 1:])
        x_visual_rep = x_visual[perm].contiguous()
        x_semantic_rep = x_semantic[perm].contiguous()

        visual_input = PackedSequence(x_visual_rep, torch.tensor(ls_transposed))
        inputs1, lengths1 = pad_packed_sequence(visual_input, batch_first=False)
        semantic_input = PackedSequence(x_semantic_rep, torch.tensor(ls_transposed))
        inputs2, lengths2 = pad_packed_sequence(semantic_input, batch_first=False)


        self.hidden_state_visual = self.init_hidden(batch_size, bidirectional=False)
        self.hidden_state_semantic = self.init_hidden(batch_size, bidirectional=False)

        output1, self.hidden_state_visual = self.lstm_visual(inputs1, self.hidden_state_visual)
        output2, self.hidden_state_semantic = self.lstm_semantic(inputs2, self.hidden_state_semantic)        
        inputs = torch.cat((output1, output2), 2)


        x_fusion = self.odeBlock(inputs, batch_size)
        x_fusion = x_fusion[1]

        x_fusion, _ = pack_padded_sequence(x_fusion, lengths1, batch_first=False)
        x_out = self.fc_predicate(x_fusion)
        result.rel_dists = x_out[inv_perm]  # for evaluation and crossentropy

        if self.training:
            return result

        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)
        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs