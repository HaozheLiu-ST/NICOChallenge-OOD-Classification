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
from tqdm import tqdm
from torchvision.utils import save_image
import augmentations
ImageFile.LOAD_TRUNCATED_IMAGES = True


"""
Domain label only given in Track 1 (i.e. domain generalization task)
"""
Domain_dict={"autumn":0, "dim":1, "grass":2, "outdoor":3, "rock":4, "water":5}

def dg_data_find(data_dir, label_dir, save_dir):
    train_json = []
    test_json = []
    file_train = open(save_dir+'track1_train_label.json', 'w')
    file_test = open(save_dir+'track1_test_label.json', 'w')
    label_dict = json.load(open(label_dir, 'r'))

    train_path_list = glob.glob(data_dir+'train/*/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        domain = file.split('/')[-3]
        dict ={}
        dict['image_path']=file
        dict['domain'] = Domain_dict[domain]
        dict['label'] = label_dict[label]
        train_json.append(dict)


    test_path_list = glob.glob(data_dir+'public_test_flat/*.jpg')
    test_path_list.sort()
    for i, file in enumerate(test_path_list):
        name = file.split('/')[-1]
        dict ={}
        dict['image_path']=file
        dict['image_name']=name
        test_json.append(dict)

    print ('train: ', len(train_json))
    print ('test: ', len(test_json))
    json.dump(train_json, file_train, indent=4)
    file_train.close()
    json.dump(test_json, file_test, indent=4)
    file_test.close()
def ood_data_find(data_dir,label_dir,save_dir):
    train_json = []
    test_json = []
    file_train = open(save_dir+'track2_train_label.json', 'w')
    file_test = open(save_dir+'track2_test_label.json', 'w')
    label_dict = json.load(open(label_dir, 'r'))

    train_path_list = glob.glob(data_dir+'train/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        dict ={}
        dict['image_path']=file
        dict['label'] = label_dict[label]
        train_json.append(dict)


    test_path_list = glob.glob(data_dir+'public_test_flat/*.jpg')
    test_path_list.sort()
    for i, file in enumerate(test_path_list):
        name = file.split('/')[-1]
        dict ={}
        dict['image_path']=file
        dict['image_name']=name
        test_json.append(dict)

    print ('train: ', len(train_json))
    print ('test: ', len(test_json))
    json.dump(train_json, file_train, indent=4)
    file_train.close()
    json.dump(test_json, file_test, indent=4)
    file_test.close()

def dg_with_seg_data_find(data_dir, seg_dir, label_dir, save_dir):
    train_json = []
    test_json = []
    file_train = open(save_dir+'track1_train_with_mask_label.json', 'w')
    file_test = open(save_dir+'track1_test_with_mask_label.json', 'w')
    label_dict = json.load(open(label_dir, 'r'))

    train_path_list = glob.glob(data_dir+'train/*/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        domain = file.split('/')[-3]
        seg_path = seg_dir+'nico_track1_r50_test_dim_2048@train@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/'+file.split(data_dir+'train/')[1][:-4]+'.png'
        name = '-'.join(file.split(data_dir)[1].split('/'))
        dict ={}
        dict['image_path']=file
        dict['image_seg_path']=seg_path
        dict['domain'] = Domain_dict[domain]
        dict['image_name'] = name
        dict['label'] = label_dict[label]
        train_json.append(dict)

    test_path_list = glob.glob(data_dir+'public_test_flat/*.jpg')
    test_path_list.sort()
    for i, file in enumerate(test_path_list):
        name = file.split('/')[-1]
        seg_path = seg_dir+'nico_track1_r50_test_dim_2048@test@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/'+file.split(data_dir+'public_test_flat/')[1][:-4]+'.png'
        dict ={}
        dict['image_path']=file
        dict['image_seg_path']=seg_path
        dict['image_name']=name
        test_json.append(dict)

    print ('train: ', len(train_json))
    print ('test: ', len(test_json))
    json.dump(train_json, file_train, indent=4)
    file_train.close()
    json.dump(test_json, file_test, indent=4)
    file_test.close()
def ood_with_seg_data_find(data_dir, seg_dir, label_dir, save_dir):
    train_json = []
    test_json = []
    file_train = open(save_dir+'track2_train_with_mask_label.json', 'w')
    file_test = open(save_dir+'track2_test_with_mask_label.json', 'w')
    label_dict = json.load(open(label_dir, 'r'))

    train_path_list = glob.glob(data_dir+'train/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        seg_path = seg_dir+'nico_track2_r50_test_dim_2048@train@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/'+file.split(data_dir+'train/')[1][:-4]+'.png'
        name = '-'.join(file.split(data_dir)[1].split('/'))
        dict ={}
        dict['image_path']=file
        dict['image_seg_path']=seg_path
        dict['image_name'] = name
        dict['label'] = label_dict[label]
        train_json.append(dict)

    test_path_list = glob.glob(data_dir+'public_test_flat/*.jpg')
    test_path_list.sort()
    for i, file in enumerate(test_path_list):
        name = file.split('/')[-1]
        seg_path = seg_dir+'nico_track2_r50_test_dim_2048@test@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/'+file.split(data_dir+'public_test_flat/')[1][:-4]+'.png'
        dict ={}
        dict['image_path']=file
        dict['image_seg_path']=seg_path
        dict['image_name']=name
        test_json.append(dict)

    print ('train: ', len(train_json))
    print ('test: ', len(test_json))
    json.dump(train_json, file_train, indent=4)
    file_train.close()
    json.dump(test_json, file_test, indent=4)
    file_test.close()

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
        # self.images = []
        # self.labels = []
        # self.domains = []
        # self.seg_maps=[]
        # for i in range(len(data)):
        #     self.images.append(Image.open(data[i]['image_path']).convert('RGB'))
        #     self.labels.append(data[i]['label'])
        #     self.domains.append(data[i]['domain'])
        #     if self._use_seg:
        #         self.seg_maps.append(Image.open(data[i]['image_seg_path']))
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

        # img = self.images[item]
        # label = self.labels[item]
        # domain = self.domains[item]
        # if self._use_seg:
        #     seg_map = self.seg_maps[item]
        #     if self._image_transformer!=None:
        #         img, seg_map = self._image_transformer(img, seg_map)
        #     return img, seg_map, label, domain
        # else:
        #     if self._image_transformer!=None:
        #         img = self._image_transformer(img)
        #     return img, label, domain
class OOD_Train_Dataset(Dataset):
    '''
    Track 2 training set
    '''

    def __init__(self, data, use_seg=False, img_transformer=None):
        self.data = data
        self._use_seg = use_seg
        self._image_transformer = img_transformer
        # self.images = []
        # self.labels = []
        # self.seg_maps=[]
        # for i in range(len(data)):
        #     self.images.append(Image.open(data[i]['image_path']).convert('RGB'))
        #     self.labels.append(data[i]['label'])
        #     if self._use_seg:
        #         self.seg_maps.append(Image.open(data[i]['image_seg_path']))

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

        # img = self.images[item]
        # label = self.labels[item]
        # if self._use_seg:
        #     seg_map = self.seg_maps[item]
        #     if self._image_transformer!=None:
        #         img, seg_map = self._image_transformer(img, seg_map)
        #     return img, seg_map, label
        # else:
        #     if self._image_transformer!=None:
        #         img = self._image_transformer(img)
        #     return img, label

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

class Test_Dataset(Dataset):
    def __init__(self, data, img_transformer=None):
        self.data = data
        self._image_transformer = img_transformer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        img_path = self.data[item]['image_path']
        img_name = self.data[item]['image_name']
        img = Image.open(img_path).convert('RGB')
        w,h = img.size
        # if w<=h and h>=256:
        #     img = img.resize((256,h))
        # elif w>h and w>=256:
        #     img = img.resize((w,256))
        # else:
        #     img = img.resize((256,256))

        img = self._image_transformer(img)
        return img_name, img

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

    data_loader_train = DataLoader(Train_Dataset(train_data,dataset_name,use_seg,data_aug), batch_size=batchsize, shuffle=True, num_workers=32,
                                   pin_memory=True, drop_last=True)
    data_loader_valid = DataLoader(Train_Dataset(valid_data,dataset_name,use_seg,data_aug), batch_size=batchsize, shuffle=False, num_workers=32,
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
    
    data_loader_train = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=32,
                                   pin_memory=True, drop_last=True)
    data_loader_valid = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=32,
                                   pin_memory=True, drop_last=False)
    return data_loader_train, data_loader_valid

def get_dataset(dataset_name, batchsize, cfg='tencrop'):
    train_data = json.load(open(dataset_name + '_train_label.json', 'r'))
    test_data = json.load(open(dataset_name + '_test_label.json', 'r'))

    data_aug = get_randomcrop_transformer()
    if 'tencrop' in cfg:
        data_aug_test = get_tencrop_transformer()
    if 'ensemble' in cfg:
        data_aug_test = get_ensemble_transformer()

    if 'track1' in dataset_name:
        data_loader_train = DataLoader(DG_Train_Dataset(train_data,data_aug), batch_size=batchsize, shuffle=True, num_workers=32,
                                       pin_memory=True, drop_last=True)
    if 'track2' in dataset_name:
        data_loader_train = DataLoader(OOD_Train_Dataset(train_data,data_aug), batch_size=batchsize, shuffle=True, num_workers=32,
                                       pin_memory=True, drop_last=True)

    data_loader_test = DataLoader(Test_Dataset(test_data,data_aug_test), batch_size=batchsize, shuffle=False, num_workers=32,
                                       pin_memory=True, drop_last=False)
    return data_loader_train,data_loader_test
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
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)

    mask = torch.where(mask>0.5, torch.tensor([1.0]), torch.tensor([0.0]))

    return image, mask


if __name__ =='__main__':
    """
    Track1 :
    train:  88866
    test:  13907
    """
    data_dir1 ='/apdcephfs/share_1290796/Datasets/NICO/Track1/public_dg_0416/'
    label_dir1 ='/apdcephfs/share_1290796/Datasets/NICO/Track1/dg_label_id_mapping.json'
    # # dg_data_find(data_dir1,label_dir1,'./')
    seg_dir1 = '/apdcephfs/share_1290796/jinhengxie/NICO/train/WSSS/experiments/predictions/'
    # dg_with_seg_data_find(data_dir1, seg_dir1, label_dir1, './')
    """
    Track2:
    train:  57266
    test:  8715
    mask dir 'nico_track2_r50_test_dim_2048@test@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4'
    """
    data_dir2 ='/apdcephfs/share_1290796/Datasets/NICO/Track2/public_ood_0412_nodomainlabel/'
    label_dir2 ='/apdcephfs/share_1290796/Datasets/NICO/Track2/ood_label_id_mapping.json'
    # ood_data_find(data_dir2,label_dir2,'./')
    seg_dir2 = '/apdcephfs/share_1290796/jinhengxie/NICO/temp/experiments/predictions/'
    # ood_with_seg_data_find(data_dir2,seg_dir2,label_dir2,'./')
    # train,valid= get_dataset_train('track1',1,True,'pairrandomcrop')
    train,valid= get_dataset_train_deepaug('track1',64,True,'pairrandomcrop')
    print (len(train),len(valid))

    train = tqdm(train)
    for x in train:
        imgs = x[0]
        seg = x[1]
