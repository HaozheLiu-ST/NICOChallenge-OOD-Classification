import argparse
import os
import random
import time
import warnings
import shutil

from models import create_net, LabelSmoothingCrossEntropy
from util import AverageMeter, ProgressMeter, accuracy
from checkpoint import save_checkpoint, load_checkpoint
from dataset_seg import get_dataset_train,get_dataset_train_deepaug
from train_func import adjust_learning_rate,train,validate
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import random
import custom_resnet 

parser = argparse.ArgumentParser(description='PyTorch NICO Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='[resnet50/resnext101_64x4d]')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument("--log_freq", type=int, default=500,
                    help="log frequency to file")
parser.add_argument("--ckpt", default="./ckpts/",
                    help="folder to output checkpoints")

parser.add_argument("--cos_lr", action='store_true',
                    help='use cosine learning rate')

parser.add_argument("--adam", action='store_true',
                    help='use adam optimizer')

parser.add_argument("--adamw", action='store_true',
                    help='use adamw optimizer')

parser.add_argument("--fine_tune", action='store_true',
                    help='fine_tune validation set')
# parser.add_argument("--augmix", action='store_true',
#                     help='adopting augmix to train the model')

parser.add_argument("--track",default="1",help='[1/2]')

# parser.add_argument("--image_size",default=448,type=int, help='[1024/448/224] only for augmix!')

parser.add_argument("--parallel", action='store_true', help='using multiple GPUs')

parser.add_argument("--deepaug",action='store_true',help='using deep augmentation')

parser.add_argument("--label_smooth", action='store_true', help='using label smooth')

parser.add_argument("--root", default='', type=str, metavar='PATH', 
                    help='root path to load datasets')

parser.add_argument("--pretrained", default='', type=str, metavar='PATH', 
                    help='path to load pretrained model (default: none)')

best_acc1 = 0

def main():
    args = parser.parse_args()
    global best_acc1
    args.ckpt += "NICO"
    args.ckpt += "-" + args.arch
    args.ckpt += "-track" + args.track
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

# |-model initilization
    model = create_net(args)

    if args.pretrained != None:
        print ('load pretrained model')
        checkpoint = torch.load(args.pretrained)
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        model.load_state_dict(state_dict, False)
    if args.parallel:
        model = torch.nn.DataParallel(model)
    model.cuda()

# |-Log files initilization
    log_file = os.path.join(args.ckpt, "log.txt")

    if os.path.exists(log_file):
        args.log_file = open(log_file, mode="a")
    else:
        args.log_file = open(log_file, mode="w")
        args.log_file.write("Network - " + args.arch + "\n")
        args.log_file.write("Learning Rate - " + str(args.lr) + "\n")
        args.log_file.write("Batch Size - " + str(args.batch_size) + "\n")
        args.log_file.write("Weight Decay - " + str(args.weight_decay) + "\n")
        args.log_file.write("--------------------------------------------------" + "\n")
    args.log_file.close()


# |-Training initilization
    if args.label_smooth:
        criterion = LabelSmoothingCrossEntropy().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    if args.adamw:
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, 
                                weight_decay=args.weight_decay)
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.cos_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        # for epoch in range(100):
            # scheduler.step()
# |-Resume initilization

    if args.resume:
        model, optimizer, best_acc1, start_epoch = load_checkpoint(args, model, optimizer)
        args.start_epoch = start_epoch



# |-Data initilization
    if args.deepaug:
        get_data = get_dataset_train_deepaug
    else:
        get_data = get_dataset_train
    if args.track == '2':
        train_loader, val_loader = get_data('track2', args.root, args.batch_size, use_seg=True,cfg='pairrandomcrop')
    else:
        train_loader, val_loader = get_data('track1', args.root, args.batch_size, use_seg=True,cfg='pairrandomcrop')

# |-Resume initilization
    for epoch in range(args.start_epoch, args.epochs):
        args.log_file = open(log_file, mode="a")

# |-Training
        if args.fine_tune:
            train(val_loader, model, criterion, optimizer, epoch, args)
            train(train_loader, model, criterion, optimizer, epoch, args)
        else:
            train(train_loader, model, criterion, optimizer, epoch, args)
            
        if(not args.cos_lr):
            adjust_learning_rate(optimizer, epoch, args)
        else:
            scheduler.step()
            print('[%03d] %.7f'%(epoch, scheduler.get_lr()[0]))

# |-Test
        acc1 = validate(val_loader, model, criterion, args)

# |-Save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        args.log_file.close()
        if args.parallel:
            save_checkpoint({
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.module.state_dict(),
                "best_acc": best_acc1,
                "optimizer" : optimizer.state_dict(),
                }, is_best, epoch, save_path=args.ckpt)
        else:
            save_checkpoint({
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc1,
                "optimizer" : optimizer.state_dict(),
                }, is_best, epoch, save_path=args.ckpt)
if __name__ == '__main__':
    main()