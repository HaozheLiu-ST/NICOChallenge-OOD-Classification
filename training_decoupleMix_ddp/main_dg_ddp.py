import argparse
import os
import torchvision.models as models

from models import create_net, LabelSmoothingCrossEntropy
from checkpoint import save_checkpoint, check_local
from dataset_seg import get_dataset_train_cross
from train_func import adjust_learning_rate,train_dg_pdd,validate_dg_pdd, adjust_learning_rate_ft
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='PyTorch NICO Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='[resnet50/densenet121/mobilenet_v2/wide_resnet50_2]')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--num_workers', default=64, type=int,
                    help='processings for data loader')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
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

parser.add_argument("--track",default="1",help='[1/2]')

parser.add_argument("--label_smooth", action='store_true', help='using label smooth')

parser.add_argument("--pretrained", default= None, type=str, metavar='PATH',
                    help='path to load pretrained model (default: none)')

parser.add_argument("--moco_pretrained", default= None, type=str, metavar='PATH',
                    help='path to load moco pretrained model (default: none)')

parser.add_argument("--root", default='', type=str, metavar='PATH',
                    help='root path to load datasets')

parser.add_argument('--dist-url', default='tcp://127.0.0.1:4455', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
best_acc1 = 0

def main():
    args = parser.parse_args()
    global best_acc1
    args.ckpt += "NICO"
    args.ckpt += "-" + args.arch
    args.ckpt += "-track" + args.track
    student_ckpt = args.ckpt+'/student/'
    teacher_ckpt = args.ckpt+'/teacher/'

    print("Use MODEL: {} for training".format(args.arch))
    print("Use Track: {} for training".format(args.track))

    if not os.path.isdir(student_ckpt):
        os.makedirs(student_ckpt)

    if not os.path.isdir(teacher_ckpt):
        os.makedirs(teacher_ckpt)


# |-DDP initilization
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
    args.rank = args.rank * ngpus_per_node + gpu

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
# |-model initilization
    model = models.__dict__[args.arch](pretrained=False,num_classes=60)
    model_teacher = models.__dict__[args.arch](pretrained=False,num_classes=60)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)


    if args.label_smooth:
        criterion = LabelSmoothingCrossEntropy().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

# |-Check local weight
    model, model_teacher, optimizer, best_acc1, start_epoch = check_local(args,model,model_teacher,optimizer)
    args.start_epoch = start_epoch

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_teacher)
    
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_teacher.cuda(args.gpu)

# |-Training initilization
    if args.cos_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        for epoch in range(args.start_epoch):
            scheduler.step()

# |-Log files initilization
    if args.rank % ngpus_per_node == 0:
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

# |-Data initilization

    if args.track == '2':
        train_dataset, valid_dataset = get_dataset_train_cross('track2','track1', args.root, args.batch_size)
    else:
        train_dataset, valid_dataset = get_dataset_train_cross('track1','track2', args.root, args.batch_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)


# |-Training and online test
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if args.rank % ngpus_per_node == 0:
            args.log_file = open(log_file, mode="a")

        train_dg_pdd(train_loader, model, model_teacher, criterion, optimizer, epoch, args)
        if(args.cos_lr):
            scheduler.step()
            print('[%03d] %.7f'%(epoch, scheduler.get_lr()[0]))
        else:
            adjust_learning_rate(optimizer, epoch, args)
        if args.rank % ngpus_per_node == 0:
            acc1 = validate_dg_pdd(val_loader, model, criterion, args)
# |-Save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            args.log_file.close()
            save_checkpoint({
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc1,
                "optimizer" : optimizer.state_dict(),
                }, is_best, epoch, save_path=args.ckpt+'/student/')
            save_checkpoint({
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model_teacher.state_dict(),
                "best_acc": best_acc1,
                "optimizer" : optimizer.state_dict(),
                }, is_best, epoch, save_path=args.ckpt+'/teacher/')
if __name__ == '__main__':
    main()
