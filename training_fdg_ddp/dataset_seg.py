#encoding:utf-8
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageFile
import json
import torch
import numpy as np
import random
import augmentations
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
Augmentation
"""
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

"""
Dataset
"""
class Valid_Cross_Dataset(Dataset):
    def __init__(self, data, train_task='track1', use_seg=False, img_transformer=None, root='/apdcephfs/share_1290796/Datasets/NICO/nico_datasets/'):
        self.data_oringin = data
        self._task = train_task
        self.root = root
        self._use_seg = use_seg
        self._image_transformer = img_transformer
        self.label_dict = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train_dg_ddp/datasets/'+train_task+'_id_mapping.json', 'r'))

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
class FourierDGDataset_Seg(Dataset):
    def __init__(self, data, task, root='/apdcephfs/share_1290796/Datasets/NICO/nico_datasets/', transformer=None):

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


def get_fourier_train_dataset_cross(train_dataset_name, valid_dataset_name, root, batchsize=64, use_seg=True):
    if use_seg:
        data_train = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train_dg_ddp/datasets/'+train_dataset_name + '_train_with_mask_label.json', 'r'))
        data_test = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train_dg_ddp/datasets/'+valid_dataset_name + '_train_with_mask_label.json', 'r'))
    else:
        data_train = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train_dg_ddp/datasets/'+train_dataset_name + '_train_label.json', 'r'))
        data_test = json.load(open('/apdcephfs/share_1290796/waltszhang/NICO_challenge/train_dg_ddp/datasets/'+valid_dataset_name + '_train_label.json', 'r'))

    data_aug = get_pair_randomcrop_transformer

    random.shuffle(data_test)
    split_num = int(0.25*len(data_test))
    data_test = data_test[:split_num]

    train_dataset = FourierDGDataset_Seg(data_train, train_dataset_name, root, data_aug)
    valid_dataset = Valid_Cross_Dataset(data_test, train_dataset_name, use_seg, data_aug)

    return train_dataset, valid_dataset




if __name__ =='__main__':
    """
    Track1 :
    train:  88866
    test:  13907
    for track2 valid: 64309
    -----------------------
    Track2:
    train:  57266
    test:  8715
    for track1 valid: 41400
    """

