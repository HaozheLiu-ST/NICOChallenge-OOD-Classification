# encoding:utf-8
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
import json
import glob
import random
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
Domain label only given in Track 1 (i.e. domain generalization task)
"""
Domain_dict = {"autumn": 0, "dim": 1, "grass": 2, "outdoor": 3, "rock": 4, "water": 5}


def dg_data_find(data_dir, label_dir, save_dir):
    train_json = []
    test_json = []
    file_train = open(save_dir + 'track1_train_label.json', 'w')
    file_test = open(save_dir + 'track1_test_label.json', 'w')
    label_dict = json.load(open(label_dir, 'r'))

    train_path_list = glob.glob(data_dir + 'train/*/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        domain = file.split('/')[-3]
        dict = {}
        dict['image_path'] = file
        dict['domain'] = Domain_dict[domain]
        dict['label'] = label_dict[label]
        train_json.append(dict)

    test_path_list = glob.glob(data_dir + 'public_test_flat/*.jpg')
    test_path_list.sort()
    for i, file in enumerate(test_path_list):
        dict = {}
        dict['image_path'] = file
        test_json.append(dict)

    print('train: ', len(train_json))
    print('test: ', len(test_json))
    json.dump(train_json, file_train, indent=4)
    file_train.close()
    json.dump(test_json, file_test, indent=4)
    file_test.close()


def ood_data_find(data_dir, label_dir, save_dir):
    train_json = []
    test_json = []
    file_train = open(save_dir + 'track2_train_label.json', 'w')
    file_test = open(save_dir + 'track2_test_label.json', 'w')
    label_dict = json.load(open(label_dir, 'r'))

    train_path_list = glob.glob(data_dir + 'train/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        dict = {}
        dict['image_path'] = file
        dict['label'] = label_dict[label]
        train_json.append(dict)

    test_path_list = glob.glob(data_dir + 'public_test_flat/*.jpg')
    test_path_list.sort()
    for i, file in enumerate(test_path_list):
        dict = {}
        dict['image_path'] = file
        test_json.append(dict)

    print('train: ', len(train_json))
    print('test: ', len(test_json))
    json.dump(train_json, file_train, indent=4)
    file_train.close()
    json.dump(test_json, file_test, indent=4)
    file_test.close()


class DG_Train_Dataset(Dataset):
    '''
    Track 1 training set
    '''

    def __init__(self, data, data_dir, img_transformer=None):
        self.data = data#[:100]
        self.data_dir = data_dir
        self._image_transformer = img_transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.data[item]['image_path'])
        label = self.data[item]['label']
        domain = self.data[item]['domain']
        img = Image.open(img_path).convert('RGB')
        img = self._image_transformer(img)

        return img, label, domain


class OOD_Train_Dataset(Dataset):
    '''
    Track 2 training set
    '''

    def __init__(self, data, data_dir, img_transformer=None):
        self.data = data#[:100]
        self.data_dir = data_dir
        self._image_transformer = img_transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.data[item]['image_path'])
        label = self.data[item]['label']
        img = Image.open(img_path).convert('RGB')
        img = self._image_transformer(img)
        return img, label


class OOD_Dataset_For_CCAM(Dataset):
    '''
    Track 2
    '''

    def __init__(self, data_dir, json_file, domain):
        data = json.load(open(json_file, 'r'))
        random.shuffle(data)
        self.data = data#[:100]
        self.domain = domain
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.data[item]['image_path'])
        # img_path = img_path[len('/apdcephfs/share_1290796/Datasets/NICO/Track2/'):]
        img_id = img_path.split('/')[-1][:-len('.jpg')]
        # img_path = '/data1/xjheng/dataset/NICO++/' + img_path
        img = Image.open(img_path).convert('RGB')

        if self.domain == 'train':
            label = self.data[item]['label']
            return img, img_id, label
        else:
            return img, img_id, None


class DG_Dataset_For_CCAM(Dataset):
    '''
    Track 1
    '''

    def __init__(self, data_dir, json_file, domain):
        data = json.load(open(json_file, 'r'))
        random.shuffle(data)
        self.data = data#[:100]
        self.domain = domain
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = os.path.join(self.data_dir, self.data[item]['image_path'])
        img_id = img_path.split('/')[-1][:-len('.jpg')]
        img = Image.open(img_path).convert('RGB')

        if self.domain == 'train':
            label = self.data[item]['label']
            domain = self.data[item]['domain']
            return img, img_id, label, domain
        else:
            return img, img_id, None, None


class Test_Dataset(Dataset):
    def __init__(self, data, img_transformer=None):
        self.data = data
        self._image_transformer = img_transformer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.data[item]['image_path']
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        if w < h:
            img = img.resize((256, h))
        else:
            img = img.resize((w, 256))
        img = self._image_transformer(img)
        return img


def get_dataset_train(dataset_name, data_dir, json_file, batchsize=32, cfg='randomcrop'):
    data = json.load(open(json_file, 'r'))
    random.shuffle(data)

    split_num = int(0.8 * len(data))
    train_data = data[:split_num]
    valid_data = data[split_num:]

    if 'randomcrop' in cfg:
        data_aug = get_randomcrop_transformer()
    if 'ensemble' in cfg:
        data_aug = get_ensemble_transformer()

    if 'track1' in dataset_name:
        data_loader_train = DataLoader(DG_Train_Dataset(train_data, data_dir, data_aug), batch_size=batchsize,
                                       shuffle=True, num_workers=32,
                                       pin_memory=True, drop_last=True)
        data_loader_valid = DataLoader(DG_Train_Dataset(valid_data, data_dir, data_aug), batch_size=batchsize, shuffle=False,
                                       num_workers=32,
                                       pin_memory=True, drop_last=False)
    if 'track2' in dataset_name:
        data_loader_train = DataLoader(OOD_Train_Dataset(train_data, data_dir, data_aug), batch_size=batchsize, shuffle=True,
                                       num_workers=32,
                                       pin_memory=True, drop_last=True)
        data_loader_valid = DataLoader(OOD_Train_Dataset(valid_data, data_dir, data_aug), batch_size=batchsize, shuffle=False,
                                       num_workers=32,
                                       pin_memory=True, drop_last=False)
    return data_loader_train, data_loader_valid


def get_dataset_test(dataset_name, batchsize=32, cfg='tencrop'):
    test_data = json.load(open(dataset_name + '_test_label.json', 'r'))

    if 'tencrop' in cfg:
        data_aug = get_tencrop_transformer()
    if 'ensemble' in cfg:
        data_aug = get_ensemble_transformer()

    if 'track1' in dataset_name:
        data_loader_train = DataLoader(Test_Dataset(test_data, data_aug), batch_size=batchsize, shuffle=False,
                                       num_workers=32,
                                       pin_memory=True, drop_last=False)
    if 'track2' in dataset_name:
        data_loader_train = DataLoader(Test_Dataset(test_data, data_aug), batch_size=batchsize, shuffle=False,
                                       num_workers=32,
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
        data_loader_train = DataLoader(DG_Train_Dataset(train_data, data_aug), batch_size=batchsize, shuffle=True,
                                       num_workers=32,
                                       pin_memory=True, drop_last=True)
    if 'track2' in dataset_name:
        data_loader_train = DataLoader(OOD_Train_Dataset(train_data, data_aug), batch_size=batchsize, shuffle=True,
                                       num_workers=32,
                                       pin_memory=True, drop_last=True)

    data_loader_test = DataLoader(Test_Dataset(test_data, data_aug_test), batch_size=batchsize, shuffle=False,
                                  num_workers=32,
                                  pin_memory=True, drop_last=False)
    return data_loader_train, data_loader_test


def get_randomcrop_transformer():
    return transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.8, 1)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(.4, .4, .4, .4),
        # transforms.RandomGrayscale(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_ensemble_transformer():
    return transforms.Compose([
        transforms.RandomResizedCrop(448, scale=(0.8, 1)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(.4, .4, .4, .4),
        # transforms.RandomGrayscale(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_tencrop_transformer():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(0.3),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Resize((448, 448))(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))
    ])


if __name__ == '__main__':
    """
    Track1 :
    train:  88866
    test:  13907
    """
    data_dir1 = '/apdcephfs/share_1290796/Datasets/NICO/Track1/public_dg_0416/'
    label_dir1 = '/apdcephfs/share_1290796/Datasets/NICO/Track1/dg_label_id_mapping.json'
    # dg_data_find(data_dir1,label_dir1,'./')
    """
    Track2:
    train:  57266
    test:  8715
    """
    data_dir2 = '/apdcephfs/share_1290796/Datasets/NICO/Track2/public_ood_0412_nodomainlabel/'
    label_dir2 = '/apdcephfs/share_1290796/Datasets/NICO/Track2/ood_label_id_mapping.json'
    # ood_data_find(data_dir2,label_dir2,'./')

    train1, valid1 = get_dataset_train('track1', 1)
    print(len(train1), len(valid1))
    train2, valid2 = get_dataset_train('track2', 1)
    print(len(train2), len(valid2))

    """
    model.eval()
    for i, (inp, label, context) in enumerate(train1):
        label, context = label, context.cuda()
        gt = torch.cat((gt, label, context), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)
    """
