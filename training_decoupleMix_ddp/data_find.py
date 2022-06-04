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

    train_path_list = glob.glob(data_dir+'public_dg_0416/train/*/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        domain = file.split('/')[-3]
        path = 'Track1/'+file.split(data_dir)[1]
        dict ={}
        dict['image_path']=path
        dict['domain'] = Domain_dict[domain]
        dict['label'] = label_dict[label]
        dict['label_name'] = label
        train_json.append(dict)


    test_path_list = glob.glob(data_dir+'public_dg_0416/public_test_flat/*.jpg')
    test_path_list.sort()
    for i, file in enumerate(test_path_list):
        name = file.split('/')[-1]
        path = 'Track1/'+file.split(data_dir)[1]
        dict ={}
        dict['image_path']=path
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

    train_path_list = glob.glob(data_dir+'public_ood_0412_nodomainlabel/train/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        path = 'Track2/'+file.split(data_dir)[1]
        dict ={}
        dict['image_path']=path
        dict['label'] = label_dict[label]
        dict['label_name'] = label
        train_json.append(dict)


    test_path_list = glob.glob(data_dir+'public_ood_0412_nodomainlabel/public_test_flat/*.jpg')
    test_path_list.sort()
    for i, file in enumerate(test_path_list):
        name = file.split('/')[-1]
        path = 'Track2/'+file.split(data_dir)[1]
        dict ={}
        dict['image_path']=path
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

    train_path_list = glob.glob(data_dir+'public_dg_0416/train/*/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        domain = file.split('/')[-3]
        path = 'Track1/'+file.split(data_dir)[1]
        seg_path = 'Track1/seg_map/'+file.split(data_dir+'public_dg_0416/')[1][:-4]+'.png'
        # seg_path = seg_dir+'nico_track1_r50_test_dim_2048@train@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/'+file.split(data_dir+'train/')[1][:-4]+'.png'
        name = '-'.join(file.split(data_dir)[1].split('/'))
        dict ={}
        dict['image_path']=path
        dict['image_seg_path']=seg_path
        dict['domain'] = Domain_dict[domain]
        dict['image_name'] = name
        dict['label'] = label_dict[label]
        dict['label_name'] = label
        train_json.append(dict)

    test_path_list = glob.glob(data_dir+'public_dg_0416/public_test_flat/*.jpg')
    test_path_list.sort()
    for i, file in enumerate(test_path_list):
        name = file.split('/')[-1]
        path = 'Track1/'+file.split(data_dir)[1]
        seg_path = 'Track1/seg_map/'+file.split(data_dir+'public_dg_0416/')[1][:-4]+'.png'
        # seg_path = seg_dir+'nico_track1_r50_test_dim_2048@test@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/'+file.split(data_dir+'public_test_flat/')[1][:-4]+'.png'
        dict ={}
        dict['image_path']=path
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

    train_path_list = glob.glob(data_dir+'public_ood_0412_nodomainlabel/train/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        path = 'Track2/'+file.split(data_dir)[1]
        seg_path = 'Track2/seg_map/'+file.split(data_dir+'public_ood_0412_nodomainlabel/')[1][:-4]+'.png'
        # seg_path = seg_dir+'nico_track2_r50_test_dim_2048@train@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/'+file.split(data_dir+'train/')[1][:-4]+'.png'
        name = '-'.join(file.split(data_dir)[1].split('/'))
        dict ={}
        dict['image_path']=path
        dict['image_seg_path']=seg_path
        dict['image_name'] = name
        dict['label'] = label_dict[label]
        dict['label_name'] = label
        train_json.append(dict)

    test_path_list = glob.glob(data_dir+'public_ood_0412_nodomainlabel/public_test_flat/*.jpg')
    test_path_list.sort()
    for i, file in enumerate(test_path_list):
        name = file.split('/')[-1]
        path = 'Track2/'+ file.split(data_dir)[1]
        seg_path = 'Track2/seg_map/'+ file.split(data_dir+'public_ood_0412_nodomainlabel/')[1][:-4]+'.png'
        # seg_path = seg_dir+'nico_track2_r50_test_dim_2048@test@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/'+file.split(data_dir+'public_test_flat/')[1][:-4]+'.png'
        dict ={}
        dict['image_path']=path
        dict['image_seg_path']=seg_path
        dict['image_name']=name
        test_json.append(dict)

    print ('train: ', len(train_json))
    print ('test: ', len(test_json))
    json.dump(train_json, file_train, indent=4)
    file_train.close()
    json.dump(test_json, file_test, indent=4)
    file_test.close()


if __name__ =='__main__':
    """
    Track1 :
    train:  88866
    test:  13907
    """
    data_dir1 ='/apdcephfs/share_1290796/Datasets/NICO/Track1/'
    label_dir1 ='/apdcephfs/share_1290796/Datasets/NICO/Track1/dg_label_id_mapping.json'
    # dg_data_find(data_dir1,label_dir1,'./datasets/')
    seg_dir1 = '/apdcephfs/share_1290796/jinhengxie/NICO/train/WSSS/experiments/predictions/'
    # dg_with_seg_data_find(data_dir1, seg_dir1, label_dir1, './datasets_cross/')
    """
    Track2:
    train:  57266
    test:  8715
    mask dir 'nico_track2_r50_test_dim_2048@test@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4'
    """
    data_dir2 ='/apdcephfs/share_1290796/Datasets/NICO/Track2/'
    label_dir2 ='/apdcephfs/share_1290796/Datasets/NICO/Track2/ood_label_id_mapping.json'
    # ood_data_find(data_dir2,label_dir2,'./datasets/')
    seg_dir2 = '/apdcephfs/share_1290796/jinhengxie/NICO/temp/experiments/predictions/'
    # ood_with_seg_data_find(data_dir2,seg_dir2,label_dir2,'./datasets_cross/')
    # train,valid= get_dataset_train('track1',1,True,'pairrandomcrop')
    # print (len(train),len(valid))

    # for x in train:
    #     imgs = x[0]
    #     seg = x[1]
    #     labels = x[2]
    #     domains = x[3]
    #     break    
    







