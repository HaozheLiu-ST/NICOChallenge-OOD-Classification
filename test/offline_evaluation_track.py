# encoding:utf-8
import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import zipfile
from tqdm import tqdm
import datasets
import models as net
import json

sys.path.append("..")  # root dir
os.chdir(sys.path[0])  # current work dir

seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def list_(string):
    return [str(word) for word in string.split(',')]
def list_n(string):
    if string=='None':
        return None
    if string=='adaptive':
        return 'adaptive'
    else:
        return [float(word) for word in string.split(',')]

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',help='[resnet18/resnet34/resnet50d/resnet50/resnext101_64x4d]')
parser.add_argument("--out", type=str, default= 'test', help='output_submit_file_name')
parser.add_argument("--track", type=str, default='track1', help='track1/2')
parser.add_argument("--cfg", type=str, default='tencrop', help='tencrop/ensemble/mixinginference')
parser.add_argument("--model_path", type=str, default= False, help='saved model path ')
parser.add_argument("--ensamble", type=str, default=False, help='ensamle or not')
parser.add_argument("--file_list", type=list_, default=None, help='file list to ensamble')
parser.add_argument('-w', "--weights", type=list_n, default='None', help='weights of each ensamble file ')

def inference(argp):
    model = net.create_net(argp)
    eval_step(model, argp)

def eval_step(net, args):

    model_dict = net.state_dict()
    pretrained_dict = torch.load(args.model_path)
    model_dict.update(pretrained_dict['state_dict'])
    net.load_state_dict(model_dict)
    net.cuda()

    test_data = datasets.get_dataset_test(dataset_name=args.track, batchsize=10, cfg = args.cfg)
    print (len(test_data))
    pred_result = eval_(test_data, net, args)
    out_result(pred_result,args)

def eval_(dataset, net, args):

    result={}
    net.eval()
    data_batch = tqdm(dataset)
    data_batch.set_description("Evaluate")

    if args.cfg=='tencrop':
        with torch.no_grad():
            for iter_, (img_name, img_) in enumerate(data_batch):
                img_ = img_.cuda()
                bs, n_crops, c, h, w = img_.size()
                img_var = img_.view(-1, c, h, w)
                output = net(img_var)

                output_mean = output.view(bs,n_crops,-1).mean(1)
                out_ = torch.argmax(output_mean,-1)
                for i in range (bs):
                    result[img_name[i]]=int(out_[i].cpu().data.numpy())

                # out_ = torch.argmax(output,-1)
                # out_ = out_.view(bs,n_crops).cpu().data.numpy()
                # for i in range (bs):
                #     bins = np.bincount(out_[i])
                #     value = np.argmax(bins)
                #     result[img_name[i]]=int(value)
    elif args.cfg=='mixinginference':
        MC_runs=5
        with torch.no_grad():
            for iter_, (img_name, img_) in enumerate(data_batch):
                img_ = img_.cuda()
                bs, c, h, w  = img_.size()
                out_mc = torch.zeros((bs,MC_runs,60))
                
                for mc in range(MC_runs):
                    index = torch.randperm(bs).cuda()
                    # lam = np.random.beta(2,10)
                    lam = 0.01
                    img_mix = (1-lam)*img_ + lam * img_[index,:]
                    out_mc[:,mc,:] = net(img_mix)
                out_mean = out_mc.mean(1)
                out_ = torch.argmax(out_mean,-1)
                for i in range (bs):
                    result[img_name[i]]=int(out_[i].cpu().data.numpy())
    else:
        with torch.no_grad():
            for iter_, (img_name, img_) in enumerate(data_batch):
                img_ = img_.cuda()
                bs, c, h, w  = img_.size()
                output = net(img_)
                out_ = torch.argmax(output,-1)
                for i in range (bs):
                    result[img_name[i]]=int(out_[i].cpu().data.numpy())
    return result

def out_result(result_dict, args):
    file = open('./results/'+str(args.track)+'/'+args.out+'.json', 'w')
    json.dump(result_dict, file, indent=4)


def voting(args):
    test_data_name={}
    name_list = []
    test_data = json.load(open(args.track + '_test_label.json', 'r'))
    for f in test_data:
        test_data_name[f['image_name']]=[]
        name_list.append(f['image_name'])

    for file in args.file_list:
        dict_ = json.load(open(file, 'r'))
        for key in test_data_name.keys():
            test_data_name[key].append(dict_[key])

    if args.weights == 'adaptive':
        weight_list = []
        avg_ensamble_results = ensamble(test_data_name, None)
        for file in args.file_list:
            dict_ = json.load(open(file, 'r'))
            weight_list.append(acc(name_list, dict_, avg_ensamble_results))
        ensamble_results = ensamble(test_data_name, weight_list)
        print (weight_list)
    else:
        ensamble_results = ensamble(test_data_name, args.weights)
        print (args.weights)

    out_result(ensamble_results,args)
    return ensamble_results

def ensamble(dict_all, w):
    ensamble_results={}
    for key in dict_all.keys():
        bins = np.bincount(dict_all[key],weights=w)
        value = np.argmax(bins)
        ensamble_results[key]=int(value)
    return ensamble_results

def acc(name_list, dict_1, dict_2):
    score = [1 if dict_1[n] == dict_2[n] else 0 for n in name_list]
    acc = np.mean(score)
    return float(acc)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.ensamble:
        voting(args)
    else:
        inference(args)
    


















