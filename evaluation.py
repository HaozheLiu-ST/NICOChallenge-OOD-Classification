# encoding:utf-8
import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms.functional as F
from torchvision import transforms
import zipfile
from tqdm import tqdm
from dataset import get_dataset_test
import json
import augmentations

sys.path.append("..")  # root dir
os.chdir(sys.path[0])  # current work dir

def list_n(string):
    if string=='None':
        return None
    if string=='adaptive':
        return 'adaptive'
    else:
        return [float(word) for word in string.split(',')]

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-a', '--arch', metavar='ARCH', default='densenet121',help='[resnet34/densenet121/wide_resnet50_2]')
parser.add_argument("--out", type=str, default= 'test', help='output_submit_file_name')
parser.add_argument('-b', '--batch_size', default=20, type=int, metavar='N', help='mini-batch size (default: 20)')
parser.add_argument("--track", type=str, default='track1', help='track1/2')
parser.add_argument("--cfg", type=str, default='None', help='[ensemble/mixinginference/None/augmix]')
parser.add_argument("--model_path", type=str, default= False, help='saved model path ')
parser.add_argument("--json_path", default='./dataset_json/', type=str, metavar='PATH',help='path to load json files')
parser.add_argument("--save_path", default='./results/', type=str, metavar='PATH',help='path to load json files')
parser.add_argument('--ckpts',nargs='+',help='the list for model ckpts')
parser.add_argument('-w', "--weights", type=list_n, default='None', help='weights of each ensamble file ')

def inference(args):
    model = models.__dict__[args.arch](pretrained=False,num_classes=60)
    model_dict = net.state_dict()
    pretrained_dict = torch.load(args.model_path)
    model_dict.update(pretrained_dict['state_dict'])

    dict_ = {}
    for k in list(model_dict.keys()):
        if 'module' in k:
            dict_[k[7:]] = model_dict[k]
        else:
            dict_[k] = model_dict[k]
    net.load_state_dict(dict_, True)
    net.cuda()
    test_data = get_dataset_test(args)
    pred_result = eval_step(test_data, net, args)
    out_result(pred_result,args)

def eval_step(dataset, net, args):

    result={}
    net.eval()
    data_batch = tqdm(dataset)
    data_batch.set_description("Evaluate")

    if args.cfg=='mixinginference':
        MC_runs=5
        with torch.no_grad():
            for iter_, (img_name, img_) in enumerate(data_batch):
                img_ = img_.cuda()
                bs, c, h, w  = img_.size()
                out_mc = torch.zeros((bs,MC_runs,60))
                
                for mc in range(MC_runs):
                    index = torch.randperm(bs).cuda()
                    lam = 0.01
                    img_mix = (1-lam)*img_ + lam * img_[index,:]
                    out_mc[:,mc,:] = net(img_mix)
                out_mean = out_mc.mean(1)
                out_ = torch.argmax(out_mean,-1)
                for i in range (bs):
                    result[img_name[i]]=int(out_[i].cpu().data.numpy())
    elif args.cfg=='augmix':
        MC_runs=40
        with torch.no_grad():
            for iter_, (img_name, img_) in enumerate(data_batch):
                img_ = img_.cuda()
                bs, c, h, w  = img_.size()
                out_mc = torch.zeros((bs,MC_runs,60)).cuda()
                
                for mc in range(MC_runs):
                    img_aug = torch.zeros_like(img_.type(torch.float32)).cuda()
                    for b in range(bs):
                        img_pil = F.to_pil_image(img_[b])
                        img_aug[b] = aug(img_pil)
                    out_mc[:,mc,:] = net(img_aug)
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

def aug(image, aug_prob_coeff=1,mixture_width=3,mixture_depth=-1,aug_severity=1):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations_all

  preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
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

def out_result(result_dict, args):
    file = open(args.save_path+str(args.track)+'/'+args.out+'.json', 'w')
    json.dump(result_dict, file, indent=4)

def voting(args):
    test_data_name={}
    name_list = []
    test_data = json.load(open(args.json_path+args.track + '_test_label.json', 'r'))
    for f in test_data:
        test_data_name[f['image_name']]=[]
        name_list.append(f['image_name'])

    if ckpts == None:
        file_list = os.list_dir(args.save_path)
    else:
        file_list = args.ckpts
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
    else:
        ensamble_results = ensamble(test_data_name, args.weights)

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

    if args.cfg=='ensemble':
        voting(args)
    else:
        inference(args)
    


















