#encoding:utf-8
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageFile
import json
import torch
import numpy as np
import random
import utils.augmentations as augmentations
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
Augmentation
"""
def data_augmentation(use_seg, img_size):
    if use_seg:
        return get_pair_randomcrop_transformer
    else:
        return get_randomcrop_transformer(img_size)
def get_pair_randomcrop_transformer(img_size, image, mask, img_to_tensor=True):
    # Random crop
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1), ratio = (3/4, 4/3))
    image = TF.resized_crop(image, i, j, h, w, size=(img_size,img_size))
    mask = TF.resized_crop(mask, i, j, h, w, size=(img_size,img_size))

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
def get_randomcrop_transformer(img_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
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
class Valid_Dataset(Dataset):
    def __init__(self, img_size, data, use_seg=True, img_transformer=None, root='./data/'):
        self.data = data
        self.root = root
        self._use_seg = use_seg
        self._image_transformer = img_transformer
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.root + self.data[item]['image_path']
        label_name = self.data[item]['label_name']
        label = self.data[item]['label']

        img = Image.open(img_path).convert('RGB')
        if self._use_seg:
            seg_path = self.data[item]['image_seg_path']
            seg_map = Image.open(seg_path)
            if self._image_transformer!=None:
                img, seg_map = self._image_transformer(self.img_size, img, seg_map)
            return img, seg_map, label
        else:
            if self._image_transformer!=None:
                img = self._image_transformer(img)
            return img, label
class Train_Dataset(Dataset):
    def __init__(self, img_size, data, task='track1', use_seg=True, root='./data/', transformer=None):
        self.data = data
        self._task = task
        self.root = root
        self.transformer = transformer
        self._use_seg = use_seg
        self.img_size = img_size
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.root + self.data[item]['image_path']
        label = self.data[item]['label']
        if self._task == 'track1':
            domain = self.data[item]['domain']
        else:
            domain = 'None'
        img = Image.open(img_path).convert('RGB')

        if self._use_seg:
            seg_path = self.data[item]['image_seg_path']
            seg_map = Image.open(seg_path)
            if self.transformer!=None:
                img, seg_map = self.transformer(self.img_size, img, seg_map, img_to_tensor=False)
                img= aug(img, transforms.PILToTensor())
            return img, seg_map, label, domain
        else:
            if self.transformer!=None:
                img = self.transformer(img)
            return img, label, domain
class FDG_Dataset_Seg(Dataset):
    def __init__(self, img_size, data, task, root='./data/', transformer=None):

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
        self.img_size = img_size
        self.transformer = transformer
        self.post_transform =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        img_path = self.root + self.data[item]['image_path']
        label = self.data[item]['label']
        seg_path = self.data[item]['image_seg_path']
        if self._task == 'track1':
            domain = self.data[item]['domain']
        else:
            domain = 'None'
        img_o = Image.open(img_path).convert('RGB')
        seg_map = Image.open(seg_path)

        img_o, seg_map = self.transformer(self.img_size, img_o, seg_map, img_to_tensor=False)
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
        img_sampled, seg_sampled = self.transformer(self.img_size, img_sampled, seg_sampled, img_to_tensor=False)
        return img_sampled, seg_sampled, label_sampled, domain_idx
class Test_Dataset(Dataset):
    def __init__(self, data, img_transformer=None, root='./data/'):
        self.data = data
        self.root = root
        self._image_transformer = img_transformer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        img_path = self.root+self.data[item]['image_path']
        img_name = self.data[item]['image_name']

        img = Image.open(img_path).convert('RGB')
        img = self._image_transformer(img)
        return img_name, img


def get_dataset_train(img_size, train_dataset_name, scheme, use_seg=True ,root='./data/', json_path='./dataset_json/'):
    
    data_train = json.load(open(json_path+train_dataset_name + '_train_with_mask_label.json', 'r'))
    data_aug = data_augmentation(use_seg, img_size)

    random.shuffle(data_train)
    split_num = int(0.2*len(data_train))
    data_test = data_train[:split_num]

    if scheme =='fdg':
        train_dataset = FDG_Dataset_Seg(img_size, data_train, train_dataset_name, root, data_aug)
        valid_dataset = Valid_Dataset(img_size, data_test, use_seg, data_aug, root)
    elif scheme == 'decouple':
        train_dataset = Train_Dataset(img_size, data_train, train_dataset_name, use_seg, root, data_aug)
        valid_dataset = Valid_Dataset(img_size, data_test, use_seg, data_aug, root)

    return train_dataset, valid_dataset


def get_dataset_test(args, img_size=512):
    test_data = json.load(open(args.json_path+args.track + '_test_label.json', 'r'))

    if args.cfg=='augmix':
        data_aug = transforms.Compose([
            transforms.CenterCrop((image_size, image_size)),
            transforms.PILToTensor()
            ])
    else:
        data_aug = transforms.Compose([
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    test_dataset = Test_Dataset(test_data,data_aug)
    data_loader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32,
                                   pin_memory=True, drop_last=False)
    return data_loader_test