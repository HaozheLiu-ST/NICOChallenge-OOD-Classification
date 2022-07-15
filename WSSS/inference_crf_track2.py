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
# from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from tools.general.io_utils import *
from tools.general.time_utils import *

from tools.ai.demo_utils import *
from tools.ai.torch_utils import *
from tools.ai.randaugment import *
from core.datasets_nico import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--json_file', required=True, type=str, help='json file contains the information of dataset')
parser.add_argument('--data_dir', required=True, type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--threshold', default=0.25, type=float)
parser.add_argument('--crf_iteration', default=0, type=int)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    ccam_dir = f'./WSSS/experiments/predictions/{args.experiment_name}/'
    pred_dir = create_directory(
        f'./WSSS/experiments/predictions/{args.experiment_name}@t={args.threshold}@ccam_inference_crf={args.crf_iteration}/')

    # mapping
    import json
    with open('./data/ood_label_id_mapping.json', 'r') as file:
        mapping = json.load(file)


    new_mapping = {v: k for k, v in mapping.items()}

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    dataset = OOD_Dataset_For_CCAM(args.data_dir, args.json_file, args.domain)

    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, label) in enumerate(dataset):

            if label is not None:
                add_dir = pred_dir + f'{new_mapping[label]}/'
                add_ccam_dir = ccam_dir + f'{new_mapping[label]}/'
            else:
                add_dir = pred_dir
                add_ccam_dir = ccam_dir

            if not os.path.exists(add_dir):
                os.makedirs(add_dir)

            png_path = add_dir + image_id + '.png'
            # if os.path.isfile(png_path):
            #     continue

            ori_w, ori_h = ori_image.size
            predict_dict = np.load(add_ccam_dir + image_id + '.npy', allow_pickle=True).item()

            keys = predict_dict['keys']
            cams = predict_dict['hr_cam']

            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
            cams = np.argmax(cams, axis=0)

            if args.crf_iteration > 0:
                cams = crf_inference_label(np.asarray(ori_image), cams, n_labels=keys.shape[0], t=args.crf_iteration)

            imageio.imwrite(png_path, (cams * 255).astype(np.uint8))
            # imageio.imwrite(add_dir + image_id + '_raw.png', ori_image)

            sys.stdout.write(
                '\r# CAMs CRF Inference [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100,
                                                                            (ori_h, ori_w), cams.shape))
            sys.stdout.flush()
        print()
        print('results were saved at {}'.format(pred_dir))