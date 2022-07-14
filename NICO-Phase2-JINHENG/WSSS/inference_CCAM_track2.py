# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
# from torch.utils.tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from core.model import *
from core.datasets_nico import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

palette = np.array([[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]])

from utils import check_positive

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--json_file', required=True, type=str, help='json file contains the information of dataset')
parser.add_argument('--data_dir', required=True, type=str)
###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--vis_dir', default='vis_cam', type=str)
parser.add_argument('--pretrained', type=str, required=True,
                        help='adopt different pretrained parameters, [supervised, mocov2, detco]')
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--vis_cam', default=False, type=bool)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    experiment_name = args.tag

    if 'train' in args.domain:
        experiment_name += '@train'
    else:
        experiment_name += '@test'

    experiment_name += '@scale=%s'%args.scales
    
    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')

    cam_path = create_directory(f'{args.vis_dir}/{experiment_name}/')

    model_path = './experiments/models/' + f'{args.tag}.pth'
    print(model_path)

    # mapping
    import json

    with open('ood_label_id_mapping.json', 'r') as file:
        mapping = json.load(file)

    new_mapping = {v: k for k, v in mapping.items()}

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    # for mIoU
    dataset = OOD_Dataset_For_CCAM(args.data_dir, args.json_file, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    model = get_model(args.pretrained)

    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    def get_cam(ori_image, scale):
        # preprocessing
        image = copy.deepcopy(ori_image)
        image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
        
        image = normalize_fn(image)
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        flipped_image = image.flip(-1)
        
        images = torch.stack([image, flipped_image])
        images = images.cuda()
        
        # inferenece
        _, _, cam = model(images, inference=False)
        # flag = check_positive(cam.clone())
        # if flag:
        #     cam = 1 - cam
        # postprocessing
        cams = F.relu(cam)
        # cams = torch.sigmoid(features)
        cams = cams[0] + cams[1].flip(-1)

        return cams

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, label) in enumerate(dataset):
            ori_w, ori_h = ori_image.size

            if label is not None:
                add_dir = pred_dir + f'{new_mapping[label]}/'
            else:
                add_dir = pred_dir

            if not os.path.exists(add_dir):
                os.makedirs(add_dir)
            npy_path = add_dir + image_id + '.npy'

            # if os.path.isfile(npy_path):
            #    continue

            strided_size = get_strided_size((ori_h, ori_w), 4)
            strided_up_size = get_strided_up_size((ori_h, ori_w), 16)

            cams_list = [get_cam(ori_image, scale) for scale in scales]

            strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]
            strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)

            hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
            hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]

            keys = torch.nonzero(torch.from_numpy(np.array([1])))[:, 0]

            strided_cams = strided_cams[keys]
            strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5

            hr_cams = hr_cams[keys]
            hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5

            if args.vis_cam:
                ######################################################################
                cam = torch.sum(hr_cams, dim=0)
                cam = cam.unsqueeze(0).unsqueeze(0)

                cam = make_cam(cam).squeeze()
                cam = get_numpy_from_tensor(cam)

                image = np.array(ori_image)

                h, w, c = image.shape

                cam = (cam * 255).astype(np.uint8)
                cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
                cam = colormap(cam)

                image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)
                # if os.path.isfile(f'{cam_path}/{image_id}.png'):
                #    continue
                if not os.path.exists(cam_path + f'{new_mapping[label]}/'):
                    os.makedirs(cam_path + f'{new_mapping[label]}/')
                cv2.imwrite(f'{cam_path}{new_mapping[label]}/{image_id}.png', image.astype(np.uint8))
                ######################################################################

            # save cams
            keys = np.pad(keys + 1, (1, 0), mode='constant')
            np.save(npy_path, {"keys": keys, "cam": strided_cams.cpu(), "hr_cam": hr_cams.cpu().numpy()})

            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), hr_cams.size()))
            sys.stdout.flush()
        print()
        print('results were saved at {}'.format(pred_dir))
