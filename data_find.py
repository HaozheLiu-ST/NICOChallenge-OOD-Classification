#encoding:utf-8
import json
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch Data Finding')
parser.add_argument("--track",default="1",help='[1/2]')
parser.add_argument("--root", default='/data/', type=str, metavar='PATH',help='path to load data')
parser.add_argument("--label_path", default='', type=str, metavar='PATH',help='path to load label')
parser.add_argument("--seg_path", default='', type=str, metavar='PATH',help='path to load mask')
parser.add_argument("--json_path", default='', type=str, metavar='PATH',help='path to save dataset json')

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
        path = file.split(data_dir)[1]
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
        path = file.split(data_dir)[1]
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
        path = file.split(data_dir)[1]
        dict ={}
        dict['image_path']=path
        dict['label'] = label_dict[label]
        dict['label_name'] = label
        train_json.append(dict)


    test_path_list = glob.glob(data_dir+'public_ood_0412_nodomainlabel/public_test_flat/*.jpg')
    test_path_list.sort()
    for i, file in enumerate(test_path_list):
        name = file.split('/')[-1]
        path = file.split(data_dir)[1]
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
    file_train = open(save_dir+'track1_train_with_mask_label.json', 'w')
    label_dict = json.load(open(label_dir, 'r'))

    train_path_list = glob.glob(data_dir+'public_dg_0416/train/*/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        domain = file.split('/')[-3]
        path = file.split(data_dir)[1]
        seg_path = seg_dir+file.split(data_dir+'public_dg_0416/train/')[1][:-4]+'.png'
        name = '-'.join(file.split(data_dir)[1].split('/'))
        dict ={}
        dict['image_path']=path
        dict['image_seg_path']=seg_path
        dict['domain'] = Domain_dict[domain]
        dict['image_name'] = name
        dict['label'] = label_dict[label]
        dict['label_name'] = label
        train_json.append(dict)

    print ('train: ', len(train_json))
    json.dump(train_json, file_train, indent=4)
    file_train.close()
def ood_with_seg_data_find(data_dir, seg_dir, label_dir, save_dir):
    train_json = []
    file_train = open(save_dir+'track2_train_with_mask_label.json', 'w')
    label_dict = json.load(open(label_dir, 'r'))

    train_path_list = glob.glob(data_dir+'public_ood_0412_nodomainlabel/train/*/*.jpg')
    train_path_list.sort()
    for i, file in enumerate(train_path_list):
        label = file.split('/')[-2]
        path = file.split(data_dir)[1]
        seg_path = seg_dir+file.split(data_dir+'public_ood_0412_nodomainlabel/train/')[1][:-4]+'.png'
        name = '-'.join(file.split(data_dir)[1].split('/'))
        dict ={}
        dict['image_path']=path
        dict['image_seg_path']=seg_path
        dict['image_name'] = name
        dict['label'] = label_dict[label]
        dict['label_name'] = label
        train_json.append(dict)

    print ('train: ', len(train_json))
    json.dump(train_json, file_train, indent=4)
    file_train.close()

def main():
    args = parser.parse_args()
    if not os.path.isdir(args.json_path):
        os.makedirs(args.json_path)

    if args.seg_path == '':
        if args.track == '1':
            dg_data_find(args.root,args.label_path,args.json_path)
        else:
            ood_data_find(args.root,args.label_path,args.json_path)
    else:
        if args.track == '1':
            dg_with_seg_data_find(args.root, args.seg_path, args.label_path, args.json_path)
        else:
            ood_with_seg_data_find(args.root, args.seg_path, args.label_path, args.json_path)
            
if __name__ =='__main__':
    main()
    """
    Track1 :
    train:  88866
    """
    """
    Track2:
    train:  57266
    """

    







