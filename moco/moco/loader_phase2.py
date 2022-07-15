# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch.nn
from PIL import ImageFilter
import random
import os
from torchvision.datasets.folder import pil_loader
"""
修改Dataset以适应Track1数据路径
"""
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

def make_datasets(data_dir):
    imgs = []
    for root, dirs, files in os.walk(data_dir):
        if len(dirs) == 0:
            for file in files:
                imgs.append(os.path.join(root, file))
    return imgs


import torch.utils.data as data
class CustomImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__()
        self.imgs = make_datasets(root)
        self.loader = pil_loader
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        path = self.imgs[idx]
        img = self.loader(path)
        if self.transform is not None:
            # random resize and crop
            img_q = self.transform(img)
            img_k = self.transform(img)
        return [img_q, img_k]
