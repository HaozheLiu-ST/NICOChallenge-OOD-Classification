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


class DG_Train_Dataset(Dataset):
    '''
    Track 1 training set
    '''
    def __init__(self, data, use_seg=False, img_transformer=None):
        self.data = data
        self._use_seg = use_seg
        self.images = []
        self.labels = []
        self.domains = []
        self.seg_maps=[]
        for i in range(len(data)):
            self.images.append(Image.open(data[i]['image_path']).convert('RGB'))
            self.labels.append(data[i]['label'])
            self.domains.append(data[i]['domain'])
            if self._use_seg:
                self.seg_maps.append(Image.open(data[i]['image_seg_path']))
        self._image_transformer = img_transformer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        # img_path = self.data[item]['image_path']
        # label = self.data[item]['label']
        # domain = self.data[item]['domain']
        # img = Image.open(img_path).convert('RGB')
        img = self.images[item]
        label = self.labels[item]
        domain = self.domains[item]
        if self._use_seg:
            seg_map = self.seg_maps[item]
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
        self.images = []
        self.labels = []
        self.seg_maps=[]
        self._use_seg = use_seg
        for i in range(len(data)):
            self.images.append(Image.open(data[i]['image_path']).convert('RGB'))
            self.labels.append(data[i]['label'])
            if self._use_seg:
                self.seg_maps.append(Image.open(data[i]['image_seg_path']))
        self._image_transformer = img_transformer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        # img_path = self.data[item]['image_path']
        # label = self.data[item]['label']
        # img = Image.open(img_path).convert('RGB')

        # if self._use_seg:
        #     seg_path = self.data[item]['image_seg_path']
        #     seg_map = Image.open(seg_path)
        #     if self._image_transformer!=None:
        #         img, seg_map = self._image_transformer(img, seg_map)
        #     return img, seg_map, label
        # else:
        #     if self._image_transformer!=None:
        #         img = self._image_transformer(img)
        #     return img, label
        img = self.images[item]
        label = self.labels[item]
        if self._use_seg:
            seg_map = self.seg_maps[item]
            if self._image_transformer!=None:
                img, seg_map = self._image_transformer(img, seg_map)
            return img, seg_map, label
        else:
            if self._image_transformer!=None:
                img = self._image_transformer(img)
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
    if 'track1' in dataset_name:
        data_loader_train = DataLoader(DG_Train_Dataset(train_data,use_seg,data_aug), batch_size=batchsize, shuffle=True, num_workers=32,
                                       pin_memory=True, drop_last=True)
        data_loader_valid = DataLoader(DG_Train_Dataset(valid_data,use_seg,data_aug), batch_size=batchsize, shuffle=False, num_workers=32,
                                       pin_memory=True, drop_last=False)
    if 'track2' in dataset_name:
        data_loader_train = DataLoader(OOD_Train_Dataset(train_data,use_seg,data_aug), batch_size=batchsize, shuffle=True, num_workers=32,
                                       pin_memory=False, drop_last=True)
        data_loader_valid = DataLoader(OOD_Train_Dataset(valid_data,use_seg,data_aug), batch_size=batchsize, shuffle=False, num_workers=32,
                                       pin_memory=False, drop_last=False)
    return data_loader_train, data_loader_valid
def get_dataset_test(dataset_name, batchsize=32, cfg='tencrop'):
    test_data = json.load(open(dataset_name + '_test_label.json', 'r'))
    data_aug = data_augmentation(cfg)
    if 'track1' in dataset_name:
        data_loader_train = DataLoader(Test_Dataset(test_data,data_aug), batch_size=batchsize, shuffle=False, num_workers=32,
                                       pin_memory=True, drop_last=False)
    if 'track2' in dataset_name:
        data_loader_train = DataLoader(Test_Dataset(test_data,data_aug), batch_size=batchsize, shuffle=False, num_workers=32,
                                       pin_memory=True, drop_last=False)

    return data_loader_train
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
    if cfg=='tencrop':
        aug = get_tencrop_transformer()
    if cfg=='randomcrop':
        aug = get_randomcrop_transformer()
    if cfg=='pairrandomcrop':
        aug = get_pair_randomcrop_transformer
    if cfg == 'None':
        aug = get_test_transformer()
    if cfg == 'mixinginference':
        aug = get_test_transformer()
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
def get_pair_randomcrop_transformer(image, mask):

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

    image = transforms.RandomGrayscale(0.3)(image)

    image = TF.to_tensor(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    mask = TF.to_tensor(mask)
    mask = torch.where(mask>0.5, 1.0, 0.0)
    return image, mask
def get_ensemble_transformer():
    return transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.8, 1)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(.4, .4, .4, .4),
        transforms.RandomGrayscale(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
def get_tencrop_transformer():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(0.3),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Resize((448,448))(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))
    ])
def get_test_transformer(): # hard-coded
    return transforms.Compose([
        transforms.CenterCrop((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

"""
Each batch has same size
"""
def get_resize_transformer_batch(w,h):
    return transforms.Compose([
        transforms.Resize((w,h)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(.4, .4, .4, .4),
        transforms.RandomGrayscale(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
def get_batch_DG(batch):
    w,h = batch[0][0].size
    imgs = []
    labels = []
    domains=[]
    for sample in batch:
        imgs.append(get_resize_transformer_batch(w,h)(sample[0]))
        labels.append(torch.tensor(sample[1], dtype=torch.int32))
        domains.append(torch.tensor(sample[2], dtype=torch.int32))

    imgs_tensor = torch.stack(imgs,0)
    return imgs_tensor, labels, domains
def get_batch_OOD(batch):
    w,h = batch[0][0].size
    imgs = []
    labels = []
    for sample in batch:
        imgs.append(get_resize_transformer_batch(w,h)(sample[0]))
        labels.append(torch.tensor(sample[1], dtype=torch.int32))
    imgs_tensor = torch.stack(imgs,0)
    return imgs_tensor, labels
def get_dataset_train_batch(dataset_name, batchsize=32):
    data = json.load(open(dataset_name + '_train_label.json', 'r'))
    random.shuffle(data)

    split_num = int(0.8*len(data))
    train_data = data[:split_num]
    valid_data = data[split_num:]


    if 'track1' in dataset_name:
        data_loader_train = DataLoader(DG_Train_Dataset(train_data), batch_size=batchsize, collate_fn=get_batch_DG, shuffle=True, num_workers=32,
                                       pin_memory=True, drop_last=True)
        data_loader_valid = DataLoader(DG_Train_Dataset(valid_data), batch_size=batchsize, collate_fn=get_batch_DG, shuffle=False, num_workers=32,
                                       pin_memory=True, drop_last=False)
    if 'track2' in dataset_name:
        data_loader_train = DataLoader(OOD_Train_Dataset(train_data), batch_size=batchsize, collate_fn=get_batch_OOD, shuffle=True, num_workers=32,
                                       pin_memory=True, drop_last=True)
        data_loader_valid = DataLoader(OOD_Train_Dataset(valid_data), batch_size=batchsize, collate_fn=get_batch_OOD, shuffle=False, num_workers=32,
                                       pin_memory=True, drop_last=False)
    return data_loader_train, data_loader_valid

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
    dg_with_seg_data_find(data_dir1, seg_dir1, label_dir1, './')
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
    train,valid= get_dataset_train('track1',1,True,'pairrandomcrop')
    # print (len(train),len(valid))

    # for x in train:
    #     imgs = x[0]
    #     seg = x[1]
    #     labels = x[2]
    #     domains = x[3]
    #     break    
    







