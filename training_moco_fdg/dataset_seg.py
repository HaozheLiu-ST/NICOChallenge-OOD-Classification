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
import cv2
from tqdm import tqdm
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
def get_pair_randomcrop_transformer(image, mask, deepaug=False, img_to_tensor=True):
    if not deepaug:
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
def colorful_spectrum_mix(img1, img2, alpha=1.0, ratio=1.0):
    """Input image size: ndarray of [H, W, C]"""
    lam = np.random.uniform(0, alpha)

    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    assert img1.shape == img2.shape
    h, w, c = img1.shape

    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))
    
    
    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)

    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))

    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))
    img21 = Image.fromarray(img21)
    img12 = Image.fromarray(img12)
    return img21, img12
def colorful_spectrum_mix_torch(img1, img2, alpha=1.0, ratio=1.0):
    """Input image size: PIL of [H, W, C]"""
    img1 = F.pil_to_tensor(img1)
    img2 = F.pil_to_tensor(img2)

    # lam = random.uniform(0, alpha)
    lam = 0.8

    assert img1.shape == img2.shape
    c, h, w = img1.shape

    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2


    img1_fft = torch.fft.fft2(img1)
    img2_fft = torch.fft.fft2(img2)
    
    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)
    
    for i in range (img1_abs.shape[0]):
        img1_abs[i] = torch.fft.fftshift(img1_abs[i])
        img2_abs[i] = torch.fft.fftshift(img2_abs[i])
    
    
    
    img1_abs_ = torch.clone(img1_abs)
    img2_abs_ = torch.clone(img2_abs)

    img1_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        torch.add(torch.mul(lam, img2_abs_[:,h_start:h_start + h_crop, w_start:w_start + w_crop]), torch.mul((1 - lam), img1_abs_[:,
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]))
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        torch.add(torch.mul(lam, img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]), torch.mul((1 - lam), img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]))

    for i in range (img1_abs.shape[0]):
        img1_abs[i] = torch.fft.ifftshift(img1_abs[i])
        img2_abs = torch.fft.ifftshift(img2_abs)

    img21 = torch.mul(img1_abs, torch.exp(1j *img1_pha))
    img12 = torch.mul(img2_abs, torch.exp(1j *img2_pha))
    
    img21 = torch.real(torch.fft.ifft2(img21))
    img12 = torch.real(torch.fft.ifft2(img12))
    img21 = torch.clamp(img21, 0, 255).type(torch.uint8)
    img12 = torch.clamp(img12, 0, 255).type(torch.uint8)
    img21 = TF.to_pil_image(img21)
    img12 = TF.to_pil_image(img12)

    return img21, img12

"""
Dataset 
"""
class Train_Dataset(Dataset):
    def __init__(self, data, task='track1', use_seg=False, root='/apdcephfs/share_1290796/Datasets/NICO/', img_transformer=None):
        self.data = data
        self._task = task
        self._use_seg = use_seg
        self.root = root
        self._image_transformer = img_transformer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.root + self.data[item]['image_path']
        label = self.data[item]['label']
        if self._task == 'track1':
            domain = self.data[item]['domain']
        img = Image.open(img_path).convert('RGB')

        if self._use_seg:
            seg_path = self.root + self.data[item]['image_seg_path']
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
class Valid_Cross_Dataset(Dataset):
    def __init__(self, data, train_task='track1', use_seg=False, img_transformer=None, root='/apdcephfs/share_1290796/Datasets/NICO/'):
        self.data_oringin = data
        self._task = train_task
        self.root = root
        self._use_seg = use_seg
        self._image_transformer = img_transformer
        self.label_dict = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/datasets/'+train_task+'_id_mapping.json', 'r'))

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
        if self._use_seg:
            seg_path = self.root + self.data[item]['image_seg_path']
            seg_map = Image.open(seg_path)
            if self._image_transformer!=None:
                img, seg_map = self._image_transformer(img, seg_map)
            return img, seg_map, label
        else:
            if self._image_transformer!=None:
                img = self._image_transformer(img)
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
class FourierDGDataset_Seg(Dataset):
    def __init__(self, data, task, root='/apdcephfs/share_1290796/Datasets/NICO/', transformer=None):
        
        self.data = data
        self._task = task
        self.root = root
        if self._task == 'track1':
            self.names = [[],[],[],[],[],[]]
            self.labels = [[],[],[],[],[],[]]
            self.seg_maps = [[],[],[],[],[],[]]
            for i in range(len(self.data)):
                self.names[int(self.data[i]['domain'])].append(self.data[i]['image_path'])
                self.labels[int(self.data[i]['domain'])].append(self.data[i]['label'])
                self.seg_maps[int(self.data[i]['domain'])].append(self.data[i]['image_seg_path'])

        self.transformer = transformer
        self.post_transform =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])])

        self.single_transform = get_randomcrop_transformer(img_to_tensor=False)

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
        img_o = Image.open(img_path).convert('RGB')
        seg_map = Image.open(seg_path)

        img_o, seg_map = self.transformer(img_o, seg_map, img_to_tensor=False)
        img_s, seg_map_s, label_s, domain_s = self.sample_image(domain)
        img_o, img_s = aug(img_o, transforms.PILToTensor()), aug(img_s, transforms.PILToTensor())

        img = [img_o, img_s]
        label = [label, label_s] 
        domain = [domain, domain_s]
        seg_maps = [seg_map, seg_map_s]

        return img, seg_maps, label, domain

    def sample_image(self, domain):
        if domain != 'None':
            domain_idx = random.randint(0, 5)
            img_idx = random.randint(0, len(self.names[domain_idx])-1)
            imgn_ame_sampled = self.root + self.names[domain_idx][img_idx]
            seg_ame_sampled = self.root + self.seg_maps[domain_idx][img_idx]
            label_sampled = self.labels[domain_idx][img_idx]
        else:
            domain_idx = 'None'
            idx = random.randint(0, len(self.data)-1)
            imgn_ame_sampled = self.root + self.data[idx]['image_path']
            seg_ame_sampled = self.root + self.data[idx]['image_seg_path']
            label_sampled = self.data[idx]['label']

        img_sampled = Image.open(imgn_ame_sampled).convert('RGB')
        seg_sampled = Image.open(seg_ame_sampled)
        img_sampled, seg_sampled = self.transformer(img_sampled, seg_sampled, img_to_tensor=False)
        return img_sampled, seg_sampled, label_sampled, domain_idx


"""
Dataloader
"""
def get_fourier_train_dataloader(dataset_name, root, batchsize=64, use_seg=True, cfg='pairrandomcrop'):
    if use_seg:
        data = json.load(open(dataset_name + '_train_with_mask_label.json', 'r'))
    else:
        data = json.load(open(dataset_name + '_train_label.json', 'r'))
    random.shuffle(data)
    split_num = int(0.8*len(data))
    train_data = data[:split_num]
    valid_data = data[split_num:]
    data_aug = data_augmentation(cfg)

    train_dataset = FourierDGDataset_Seg(train_data, dataset_name, root, data_aug)
    valid_dataset_ft = FourierDGDataset_Seg(valid_data, dataset_name, root, data_aug)
    valid_dataset_test = Train_Dataset(valid_data, dataset_name, use_seg, root, data_aug)

    data_loader_train = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    data_loader_valid_ft = DataLoader(valid_dataset_ft, batch_size=batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=False) # for fine-tune
    data_loader_valid_test = DataLoader(valid_dataset_test, batch_size=batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=False) # for test
    return data_loader_train, data_loader_valid_ft, data_loader_valid_test

def get_fourier_train_dataloader_cross(train_dataset_name, valid_dataset_name, root, batchsize=64, use_seg=True, cfg='pairrandomcrop'):
    if use_seg:
        data_train = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/datasets/'+train_dataset_name + '_train_with_mask_label.json', 'r'))
        data_test = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/datasets/'+valid_dataset_name + '_train_with_mask_label.json', 'r'))
    else:
        data_train = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/datasets/'+train_dataset_name + '_train_label.json', 'r'))
        data_test = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/datasets/'+valid_dataset_name + '_train_label.json', 'r'))

    data_aug = data_augmentation(cfg)

    random.shuffle(data_test)
    split_num = int(0.25*len(data_test))
    data_test = data_test[:split_num]

    train_dataset = FourierDGDataset_Seg(data_train, train_dataset_name, root, data_aug)
    valid_dataset = Valid_Cross_Dataset(data_test, train_dataset_name, use_seg, data_aug)
    
    data_loader_train = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    data_loader_valid = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=False) # for test
    return data_loader_train, data_loader_valid

def get_dataset_train_cross(train_dataset_name, valid_dataset_name, root, batchsize=32, use_seg=False, cfg='randomcrop'):
    
    if use_seg:
        data_train = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/datasets/'+train_dataset_name + '_train_with_mask_label.json', 'r'))
        data_test = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/datasets/'+valid_dataset_name + '_train_with_mask_label.json', 'r'))
    else:
        data_train = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/datasets/'+train_dataset_name + '_train_label.json', 'r'))
        data_test = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/datasets/'+valid_dataset_name + '_train_label.json', 'r'))

    data_aug = data_augmentation(cfg)

    random.shuffle(data_test)
    split_num = int(0.2*len(data_test))
    data_test = data_test[:split_num]

    train_dataset = Train_Dataset(data_train, train_dataset_name, use_seg, root, data_aug)
    valid_dataset = Valid_Cross_Dataset(data_test, train_dataset_name, use_seg, data_aug)
    data_loader_train = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=8,
                                   pin_memory=True, drop_last=True)
    data_loader_valid = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=8,
                                   pin_memory=True, drop_last=False)
    return data_loader_train, data_loader_valid
def get_dataset_train(train_dataset_name, root, batchsize=32, use_seg=False, cfg='randomcrop'):
    
    if use_seg:
        data = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/datasets/'+train_dataset_name + '_train_with_mask_label.json', 'r'))
    else:
        data = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/datasets/'+train_dataset_name + '_train_label.json', 'r'))

    data_aug = data_augmentation(cfg)

    random.shuffle(data)
    split_num = int(0.8*len(data))
    data_train = data[:split_num]
    data_test = data[split_num:]

    train_dataset = Train_Dataset(data_train, train_dataset_name, use_seg, root, data_aug)
    valid_dataset = Train_Dataset(data_test, train_dataset_name, use_seg, root, data_aug)
    
    data_loader_train = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=8,
                                   pin_memory=True, drop_last=True)
    data_loader_valid = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=8,
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

    data_loader_train = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=8,
                                   pin_memory=True, drop_last=True)
    data_loader_valid = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=8,
                                   pin_memory=True, drop_last=False)
    return data_loader_train, data_loader_valid

if __name__ =='__main__':
    """
    Track1 :
    train:  88866
    test:  13907
    for track2 valid: 64309 
    """
    train, valid = get_fourier_train_dataloader_cross('track1','track2', '/apdcephfs/share_1290796/Datasets/NICO/', 1,True,'pairrandomcrop')
    print (len(train), len(valid))
    """
    Track2:
    train:  57266
    test:  8715
    for track1 valid: 41400 
    """
    train, valid= get_fourier_train_dataloader_cross('track2','track1', '/apdcephfs/share_1290796/Datasets/NICO/', 1,True,'pairrandomcrop')
    print (len(train), len(valid))
