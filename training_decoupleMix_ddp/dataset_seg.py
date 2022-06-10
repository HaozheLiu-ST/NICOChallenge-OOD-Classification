#encoding:utf-8
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageFile
import json
import glob
import torch
import numpy as np
from math import sqrt
import random
import sys
from torchvision.utils import save_image
import augmentations
ImageFile.LOAD_TRUNCATED_IMAGES = True


"""
Domain label only given in Track 1 (i.e. domain generalization task)
"""
Domain_dict={"autumn":0, "dim":1, "grass":2, "outdoor":3, "rock":4, "water":5}

"""
Augmentation
"""
def data_augmentation(cfg):
    if cfg=='randomcrop':
        aug = get_randomcrop_transformer()
    if cfg=='pairrandomcrop':
        aug = get_pair_randomcrop_transformer
    return aug
def get_randomcrop_transformer(img_to_tensor=True):
    if img_to_tensor:
        return transforms.Compose([
            transforms.RandomResizedCrop(512, scale=(0.8, 1)),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(.4, .4, .4, .4),
            transforms.RandomGrayscale(0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(512, scale=(0.8, 1)),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(.4, .4, .4, .4),
            transforms.RandomGrayscale(0.3),
        ])
def get_pair_randomcrop_transformer(image, mask, img_to_tensor=True):
    # Random crop
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1), ratio = (3/4, 4/3))
    image = TF.resized_crop(image, i, j, h, w, size=(512,512))
    mask = TF.resized_crop(mask, i, j, h, w, size=(512,512))

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
    if img_to_tensor:
        image = aug(image,transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    mask = TF.to_tensor(mask)
    mask = torch.where(mask>0.5, torch.tensor([1.0]), torch.tensor([0.0]))

    return image, mask
def aug(image, preprocess,aug_prob_coeff=1,mixture_width=3,mixture_depth=-1,aug_severity=1):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations_all

  ws = np.float32(
      np.random.dirichlet([aug_prob_coeff] * mixture_width))
  m = np.float32(np.random.beta(aug_prob_coeff, aug_prob_coeff))

  mix = torch.zeros_like(preprocess(image).type(torch.float32))

  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug).type(torch.float32)

  mixed = (1 - m) * preprocess(image).type(torch.float32) + m * mix
  return mixed

"""
Dataset 
"""
class Train_Dataset(Dataset):
    def __init__(self, data, task='track1', root='', transformer=None):
        self.data = data
        self._task = task
        self.root = root
        self.transformer = transformer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.root + self.data[item]['image_path']
        label = self.data[item]['label']
        seg_path = self.root + self.data[item]['image_seg_path']
        if self._task == 'track1':
            domain = self.data[item]['domain']
        else:
            domain = 'None'
        img = Image.open(img_path).convert('RGB')
        seg_map = Image.open(seg_path)
        img, seg_map = self.transformer(img, seg_map, img_to_tensor=False)
        img= aug(img, transforms.PILToTensor())
        return img, seg_map, label, domain

class Valid_Cross_Dataset(Dataset):
    def __init__(self, data, train_task='track1', root='', img_transformer=None):
        self.data_oringin = data
        self.root = root
        self._image_transformer = img_transformer
        self.label_dict = json.load(open('/apdcephfs/share_1290796/Datasets/NICO/dataset_json/'+train_task+'_id_mapping.json', 'r'))

        self.data = []
        for i in range (len(self.data_oringin)):
            if self.data_oringin[i]['label_name'] in self.label_dict:
                self.data.append(self.data_oringin[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.root + self.data[item]['image_path']
        label_name = self.data[item]['label_name']
        label = self.label_dict[label_name]

        img = Image.open(img_path).convert('RGB')
        seg_path = self.root + self.data[item]['image_seg_path']
        seg_map = Image.open(seg_path)
        img, seg_map = self._image_transformer(img, seg_map)
        return img, seg_map, label



def get_dataset_train_cross(train_dataset_name, valid_dataset_name, root='/apdcephfs/share_1290796/Datasets/NICO/nico_datasets/', batchsize=32):
    
    data_train = json.load(open('/apdcephfs/share_1290796/Datasets/NICO/dataset_json/'+train_dataset_name + '_train_with_mask_label.json', 'r'))
    data_test = json.load(open('/apdcephfs/share_1290796/Datasets/NICO/dataset_json/'+valid_dataset_name + '_train_with_mask_label.json', 'r'))

    data_aug = get_pair_randomcrop_transformer

    random.shuffle(data_test)
    split_num = int(0.25*len(data_test))
    data_test = data_test[:split_num]

    train_dataset = Train_Dataset(data_train, train_dataset_name, root, data_aug)
    valid_dataset = Valid_Cross_Dataset(data_test, train_dataset_name, root, data_aug)

    return train_dataset, valid_dataset
