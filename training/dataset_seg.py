#encoding:utf-8
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageFile
import json
import glob
import torch
import numpy as np
import random
from torchvision.utils import save_image
import augmentations
ImageFile.LOAD_TRUNCATED_IMAGES = True


"""
Domain label only given in Track 1 (i.e. domain generalization task)
"""
Domain_dict={"autumn":0, "dim":1, "grass":2, "outdoor":3, "rock":4, "water":5}



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

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed

class DG_Train_Dataset(Dataset):
    '''
    Track 1 training set
    '''
    def __init__(self, data, use_seg=False, img_transformer=None):
        self.data = data
        self._use_seg = use_seg
        self._image_transformer = img_transformer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.data[item]['image_path']
        label = self.data[item]['label']
        domain = self.data[item]['domain']
        img = Image.open(img_path).convert('RGB')

        if self._use_seg:
            seg_path = self.data[item]['image_seg_path']
            seg_map = Image.open(seg_path)
            if self._image_transformer!=None:
                img, seg_map = self._image_transformer(img, seg_map)
            return img, seg_map, label, domain
        else:
            if self._image_transformer!=None:
                img = self._image_transformer(img)
            return img, label, domain

class OOD_Train_Dataset(Dataset):
    '''
    Track 2 training set
    '''

    def __init__(self, data, use_seg=False, img_transformer=None):
        self.data = data
        self._use_seg = use_seg
        self._image_transformer = img_transformer

    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        img_path = self.data[item]['image_path']
        label = self.data[item]['label']
        img = Image.open(img_path).convert('RGB')

        if self._use_seg:
            seg_path = self.data[item]['image_seg_path']
            seg_map = Image.open(seg_path)
            if self._image_transformer!=None:
                img, seg_map = self._image_transformer(img, seg_map)
            return img, seg_map, label
        else:
            if self._image_transformer!=None:
                img = self._image_transformer(img)
            return img, label

class Train_Dataset(Dataset):
    def __init__(self, data, task='track1', use_seg=False, img_transformer=None):
        self.data = data
        self._task = task
        self._use_seg = use_seg
        self._image_transformer = img_transformer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.data[item]['image_path']
        label = self.data[item]['label']
        if self._task == 'track1':
            domain = self.data[item]['domain']
        img = Image.open(img_path).convert('RGB')

        if self._use_seg:
            seg_path = self.data[item]['image_seg_path']
            seg_map = Image.open(seg_path)
            if self._image_transformer!=None:
                img, seg_map = self._image_transformer(img, seg_map)
            if self._task == 'track1':
                return img, seg_map, label, domain
            return img, seg_map, label
        else:
            if self._image_transformer!=None:
                img = self._image_transformer(img)
            if self._task == 'track1':
                return img, label, domain
            return img, label
class Train_Dataset_DeepAug(Dataset):
    def __init__(self, data, task='track1', use_seg=False, img_transformer=None):
        self.data = data
        self._use_seg = use_seg
        self._task = task
        self._image_transformer = img_transformer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.data[item]['image_path']
        label = self.data[item]['label']
        if self._task == 'track1':
            domain = self.data[item]['domain']
            img_path_ = '/apdcephfs/share_1290796/waltszhang/NICO_challenge/cae_pretrain/Datasets/'+img_path.split('/apdcephfs/share_1290796/Datasets/NICO/')[1][:-4]+'.png'
        if self._task == 'track2':
            img_path_ = '/apdcephfs/share_1290796/waltszhang/NICO_challenge/cae_pretrain/Datasets/'+img_path.split('/apdcephfs/share_1290796/Datasets/NICO/')[1][:-4]+'.png'


        img = Image.open(img_path_).convert('RGB')

        if self._use_seg:
            seg_path = self.data[item]['image_seg_path']
            if self._task == 'track1':
                seg_path_ = '/apdcephfs/share_1290796/waltszhang/NICO_challenge/cae_pretrain/Datasets/Track1/seg_maps/train/'+seg_path.split('/apdcephfs/share_1290796/jinhengxie/NICO/train/WSSS/experiments/predictions/nico_track1_r50_test_dim_2048@train@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/')[1][:-4]+'.png'
            if self._task == 'track2':
                seg_path_ = '/apdcephfs/share_1290796/waltszhang/NICO_challenge/cae_pretrain/Datasets/Track2/seg_maps/train/'+seg_path.split('/apdcephfs/share_1290796/jinhengxie/NICO/temp/experiments/predictions/nico_track2_r50_test_dim_2048@train@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/')[1][:-4]+'.png'

            seg_map = Image.open(seg_path_).convert('L')
            img, seg_map = self._image_transformer(img, seg_map, True)
            if self._task == 'track1':
                return img, seg_map, label, domain
            return img, seg_map, label
        else:
            img = self._image_transformer(image)
            if self._task == 'track1':
                return img, label, domain
            return img, label

def get_dataset_train(dataset_name, batchsize=32, use_seg=False, cfg='randomcrop'):
    if use_seg:
        data = json.load(open(dataset_name + '_train_with_mask_label.json', 'r'))
    else:
        data = json.load(open(dataset_name + '_train_label.json', 'r'))
    random.shuffle(data)
    split_num = int(0.8*len(data))
    train_data = data[:split_num]
    valid_data = data[split_num:]
    data_aug = data_augmentation(cfg)

    data_loader_train = DataLoader(Train_Dataset(train_data,dataset_name,use_seg,data_aug), batch_size=batchsize, shuffle=True, num_workers=16,
                                   pin_memory=True, drop_last=True)
    data_loader_valid = DataLoader(Train_Dataset(valid_data,dataset_name,use_seg,data_aug), batch_size=batchsize, shuffle=False, num_workers=16,
                                   pin_memory=True, drop_last=False)
    return data_loader_train, data_loader_valid

def get_dataset_train_deepaug(dataset_name, batchsize=32, use_seg=False, cfg='randomcrop'):
    if use_seg:
        data = json.load(open(dataset_name + '_train_with_mask_label.json', 'r'))
    else:
        data = json.load(open(dataset_name + '_train_label.json', 'r'))
    random.shuffle(data)
    split_num = int(0.8*len(data))
    train_data = data[:split_num]
    valid_data = data[split_num:]
    data_aug = data_augmentation(cfg)

    train_set = Train_Dataset(train_data,dataset_name,use_seg,data_aug)
    deepaug_set = Train_Dataset_DeepAug(train_data,dataset_name,use_seg,data_aug)
    train_dataset = torch.utils.data.ConcatDataset([train_set, deepaug_set])
    valid_dataset = Train_Dataset(valid_data,dataset_name,use_seg,data_aug)

    data_loader_train = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=16,
                                   pin_memory=True, drop_last=True)
    data_loader_valid = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=16,
                                   pin_memory=True, drop_last=False)
    return data_loader_train, data_loader_valid

def data_augmentation(cfg):
    if cfg=='randomcrop':
        aug = get_randomcrop_transformer()
    if cfg=='pairrandomcrop':
        aug = get_pair_randomcrop_transformer
    return aug

def get_randomcrop_transformer():
    return transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.8, 1)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(.4, .4, .4, .4),
        transforms.RandomGrayscale(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
def get_pair_randomcrop_transformer(image, mask, deepaug=False):
    if not deepaug:
        # Random crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1), ratio = (3/4, 4/3))
        image = TF.resized_crop(image, i, j, h, w, size=(448,448))
        mask = TF.resized_crop(mask, i, j, h, w, size=(448,448))

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
    image = aug(image,transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    mask = TF.to_tensor(mask)

    mask = torch.where(mask>0.5, torch.tensor([1.0]), torch.tensor([0.0]))

    return image, mask


if __name__ =='__main__':
    train,valid= get_dataset_train_deepaug('track1',1,True,'pairrandomcrop')
