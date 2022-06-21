# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch.nn
from PIL import ImageFilter
import random
import os
import json
from PIL import Image
import torchvision.datasets as datasets
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import warnings
import math
import copy
from typing import Tuple, List, Optional
from torch import Tensor
from collections.abc import Sequence

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def load_pseudo_bbox(bbox_dir):
    final_dict = {}
    file_list = os.listdir(bbox_dir)
    for i, now_name in enumerate(file_list):
        if i % 1000 == 0:
            print(f'load [{i}/{len(file_list)}] json!')
        now_json_file = os.path.join(bbox_dir, now_name)
        with open(now_json_file, 'r') as fp:
            name_bbox_dict = json.load(fp)
        final_dict.update(name_bbox_dict)

    return final_dict

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))
        self.domain_dict = {"autumn": 0, "dim": 1, "grass": 2, "outdoor": 3, "rock": 4, "water": 5}

    def __getitem__(self, idx):

        path = self.imgs[idx][0]
        label = self.imgs[idx][1]
        img = self.loader(path)

        domain = self.domain_dict[path.split('/')[-1].split('_')[0]]

        if self.transform is not None:
            # random resize and crop
            img_q = self.transform(img)
            img_k = self.transform(img)

        return [img_q, img_k], label, domain
