"""
Configuration file!
"""
import os
from argparse import ArgumentParser
import numpy as np

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')

def path(fn):
    return os.path.join(DATA_PATH, fn)

def stanford_path(fn):
    return os.path.join(DATA_PATH, 'stanford_filtered', fn)

# =============================================================================
# Update these with where your data is stored ~~~~~~~~~~~~~~~~~~~~~~~~~

VG_IMAGES = '/home/cong/Dokumente/tmp/data/visualgenome/images'
RCNN_CHECKPOINT_FN = path('faster_rcnn_500k.h5')

IM_DATA_FN = stanford_path('image_data.json')
VG_SGG_FN = stanford_path('VG-SGG.h5')
VG_SGG_DICT_FN = stanford_path('VG-SGG-dicts.json')
PROPOSAL_FN = stanford_path('proposals.h5')

COCO_PATH = '/home/cong/Dokumente/mscoco'
# =============================================================================
# =============================================================================


MODES = ('sgdet', 'sgcls', 'predcls')

BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592#592      # Our images will be resized to this res without padding

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
RPN_FG_FRACTION = 0.5
FG_FRACTION = 0.25
# Total number of examples
RPN_BATCHSIZE = 256
ROIS_PER_IMG = 256
REL_FG_FRACTION = 0.25
RELS_PER_IMG = 256

RELS_PER_IMG_REFINE = 64

BATCHNORM_MOMENTUM = 0.01
ANCHOR_SIZE = 16

ANCHOR_RATIOS = (0.23232838, 0.63365731, 1.28478321, 3.15089189) #(0.5, ĺeftright, 2.0)
ANCHOR_SCALES = (2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731) #(4, 8, 16, 32)

class ModelConfig(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """
        Defaults
        """
        self.coco = None
        self.ckpt = None
        self.save_dir = None
        self.lr = None
        self.batch_size = None
        self.val_size = None
        self.l2 = None
        self.num_gpus = None
        self.num_workers = None
        self.print_interval = None
        self.gt_box = None
        self.mode = None
        self.refine = None
        self.ad3 = False
        self.test = False
        self.adam = True
        self.multi_pred= False
        self.cache = None
        self.use_proposals=False
        self.use_resnet=False
        self.num_epochs=None
        self.old_feats=False
        self.order=None
        self.det_ckpt=None
        self.rec_dropout = None
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())

        print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
        for x, y in self.args.items():
            print("{} : {}".format(x, y))

        self.__dict__.update(self.args)

        if len(self.ckpt) != 0:
            self.ckpt = os.path.join(ROOT_PATH, self.ckpt)
        else:
            self.ckpt = None

        if len(self.cache) != 0:
            self.cache = os.path.join(ROOT_PATH, self.cache)
        else:
            self.cache = None

        if len(self.save_dir) == 0:
            self.save_dir = None
        else:
            self.save_dir = os.path.join(ROOT_PATH, self.save_dir)
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

        assert self.val_size >= 0

        if self.mode not in MODES:
            raise ValueError("Invalid mode: mode must be in {}".format(MODES))


        if self.ckpt is not None and not os.path.exists(self.ckpt):
            raise ValueError("Ckpt file ({}) doesnt exist".format(self.ckpt))

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')


        # Options to deprecate
        parser.add_argument('-coco', dest='coco', help='Use COCO (default to VG)', action='store_true')
        parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str, default='')
        parser.add_argument('-det_ckpt', dest='det_ckpt', help='Filename to load detection parameters from', type=str, default='')

        parser.add_argument('-save_dir', dest='save_dir',
                            help='Directory to save things to, such as checkpoints/save', default='', type=str)

        parser.add_argument('-ngpu', dest='num_gpus', help='cuantos GPUs tienes', type=int, default=3)
        parser.add_argument('-nwork', dest='num_workers', help='num processes to