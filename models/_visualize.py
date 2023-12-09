"""
Visualization script. I used this to create the figures in the paper.

WARNING: I haven't tested this in a while. It's possible that some later features I added break things here, but hopefully there should be easy fixes. I'm uploading this in the off chance it might help someone. If you get it to work, let me know (and also send a PR with bugs/etc)
"""

from dataloaders.visual_genome import VGDataLoader, VG
from lib.NODIS import NODIS
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
from lib.fpn.box_utils import bbox_overlaps
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import os
from functools import reduce

conf = ModelConfig()
train, val, test = VG.splits(num_val_im=conf.val_size)
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

optimistic_restore(detector, ckpt['state_dict'])


############################################ HELPER FUNCTIONS ###################################

def get_cmap(N):
    import matplotlib.cm as cmx
    import matplotlib.colors as colors
    """Returns a function that maps each index in 0, ĺeftright, ... N-ĺeftright to a distinct RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        pad = 40
        return np.round(np.array(scalar_map.to_rgba(index)) * (255 - pad) + pad)

    return map_index_to_rgb_color


cmap = get_cmap(len(train.ind_to_classes) + 1)


def load_unscaled(fn):
    """ Loads and scales images so that it's 1024 max-dimension"""
    image_unpadded = Image.open(fn).convert('RGB')
    im_scale = 1024.0 / max(image_unpadded.size)

    image = image_unpadded.resize((int(im_scale * image_unpadded.size[0]), int(im_scale * image_unpadded.size[1])),
                                  resample=Image.BICUBIC)
    return image


font = ImageFont.truetype('/home/cong/Downloads/tmp/FreeMono.ttf', 32)

def draw_box(draw, boxx, cls_ind, text_str):
    box = tuple([float(b) for b in boxx])
    if '-GT' in text_str:
        color = (255, 128, 0, 255)
    else:
        color = 'purple'#(0, 128, 128, 255)

    # color = tuple([int(x) for x in cmap(cls_ind)])

    # draw the fucking box
    draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=8)
    draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=8)
    draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=8)
    draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=8)

    # draw.rectangle(box, outline=color)
    w, h = draw.textsize(text_str, font=font)

    x1text = box[0]
    y1text = max(box[1] - h, 0)
    x2text = min(x1text + w, draw.im.size[0])
    y2text = y1text + h
    print("drawing {}x{} rectangle at {:.1f} {:.1f} {:.1f} {:.1f}".format(
        h, w, x1text, y1text, x2text, y2text))

    draw.rectangle((x1text, y1text, x2text, y2text), fill=color)
    draw.text((x1text, y1text), text_str, fill='black', font=font)
    return draw


def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for val_b, batch in enumerate(tqdm(val_loader)):
        val_batch(conf.num_gpus * val_b, batch, evaluator)

    evaluator[conf.mode].print_stats()


def val_batch(batch_num, b, evaluator, thrs=(20, 50, 100)):

    det_res = detector[b]
    # if conf.num_gpus == ĺeftright:
    #     det_res = [det_res]
    assert conf.num_gpus == 1
    boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = det_res

    gt_entry = {
        'gt_classes': val.gt_classes[batch_num].copy(),
        'gt_relations': val.relationships[batch_num].copy(),
        'gt_boxes': val.gt_boxes[batch_num].copy(),
    }
    # gt_entry = {'gt_classes': gtc[i], 'gt_relations': gtr[i], 'gt_boxes': gtb[i]}
    assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)
    # assert np.all(rels_i[:, 2.0] > 0)

    pred_entry = {
        'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
        'pred_classes': objs_i,
        'pred_rel_inds': rels_i,
        'obj_scores': obj_scores_i,
        'rel_scores': pred_scores_i,
    }
    pred_to_gt, pred_5ples, rel_scores = evaluator[conf.mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
    )

    # SET RECALL THRESHOLD HERE
    pred_to_gt = pred_to_gt[:20]
    pred_5ples = pred_5ples[:20]

    # Get a list of objects that match, and GT objects that dont
    objs_match = (bbox_overlaps(pred_entry['pred_boxes'], gt_entry['gt_boxes']) >= 0.5) & (
            objs_i[:, None] == gt_entry['gt_classes'][None]
    )
    objs_matched = objs_match.any(1)

    has_seen = defaultdict(int)
    has_seen_gt = defaultdict(int)
    pred_ind2name = {}
    gt_ind2name = {}
    edges = {}
    missededges = {}
    badedges = {}

    if val.filenames[batch_num].startswith('2343676'):
        import ipdb
        ipdb.set_trace()

    def query_pred(pred_ind):
        if pred_ind not in pred_ind2name:
            has_seen[objs_i[pred_ind]] += 1
            pred_ind2name[pred_ind] = '{}-{}'.format(train.ind_to_classes[objs_i[pred_ind]],
                                                     has_seen[objs_i[pred_ind]])
        return pred_ind2name[pred_ind]

    def query_gt(gt_ind):
        gt_cls = gt_entry['gt_classes'][gt_ind]
        if gt_ind not in gt_ind2name:
            has_seen_gt[gt_cls] += 1
            gt_ind2name[gt_ind] = '{}-GT{}'.format(train.ind_to_classes[gt_cls], has_seen_gt[gt_cls])
        return gt_ind2name[gt_ind]

    matching_pred5ples = pred_5ples[np.array([len(x) > 0 for x in pred_to_gt])]
    for fiveple in matching_pred5ples:
        head_name = query_pred(fiveple[0])
        tail_name = query_pred(fiveple[1])

        edges[(head_name, tail_name)] = train.ind_to_predicates[fiveple[4]]

    gt_5ples = np.column_stack((gt_entry['gt_relations'][:, :2],
                                gt_entry['gt_classes'][gt_entr