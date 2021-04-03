from config import COCO_PATH, IM_SCALE, BOX_SCALE
import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
from lib.fpn.anchor_targets import anchor_target_layer
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, RandomOrder, Hue, random_crop
import numpy as np
from dataloaders.blob import Blob
import torch

class CocoDetection(Dataset):
    """
    Adapted from the torchvision code
    """

    def __init__(self, mode):
        """
        :param mode: train2014 or val2014
        """
        self.mode = mode
        self.root = os.path.join(COCO_PATH, mode)
        self.ann_file = os.path.join(COCO_PATH, 'annotations', 'instances_{}.json'.format(mode))
        self.coco = COCO(self.ann_file)
        self.ids = [k for k in self.coco.imgs.keys() if len(self.coco.imgToAnns[k]) > 0]


        tform = []
        if self.is_train:
             tform.append(RandomOrder([
                 Grayscale(),
                 Brightness(),
                 Contrast(),
                 Sharpness(),
                 Hue(),
             ]))

        tform += [
            SquarePad(),
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       