import argparse
import os
import torchvision.models as models
from copy import deepcopy
from utils.checkpoint import save_checkpoint, check_local
from utils.dataset import get_dataset_train
from utils.train_func import train_fdg, train_decouple, validate
from utils.util import warm_update
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='PyTorch NICO Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',help='[densenet121/resnet34/wide_resnet50_2]')
parser.add_argument('--epochs', default=90, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--total_epoch', default=200, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N',help='mini-batch size (default: 256)',dest='batch_size')
parser.add_argument('--num_workers', default=64, type=int,help='processings for data loader')
parser.add_argument("--update_epoch",default=1, type=int, help='per "update_epoch" epoch to use swa update')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)',dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,metavar='N', help='print frequency (default: 10)')
parser.add_argument("--log_freq", type=int, default=500,help="log frequency to file")
parser.add_argument("--ckpt", default="./ckpts/",help="folder to output checkpoints")
parser.add_argument("--scheme", default="fdg",help="[fdg/decouple]")
parser.add_argument("--ratio", type=float, default=1.0, help='control foriour range')
parser.add_argument("--cos_lr", action='store_true',help='use cosine learning rate')
parser.add_argument("--swa", action='store_true', help='use swa')
parser.add_argument("--use_seg", action='store_true', help='use seg')
parser.add_argument("--track",default="1",help='[1/2]')
parser.add_argument("--fine_tune", default= None, type=str, metavar='PATH',help='path to load pretrained model and fintune(default: none)')
parser.add_argument("--pretrained", default= None, type=str, metavar='PATH',help='path to load pretrained model (default: none)')
parser.add_argument("--moco_pretrained", default= None, type=str, metavar='PATH',help='path to load moco pretrained model (default: none)')
parser.add_argument("--root", default='', type=str, metavar='PATH',help='root path to load datasets')
parser.add_argument("--json_path", default='./dataset_json/', type=str, metavar='PATH',help='path to load json files')
parser.add_argument('--img_size',nargs='+',help='the list for model ckpts')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:4455', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')

best_acc1 = 0

def main():
    args = parser.parse_args()
    global best_acc1
    args.ckpt += "NICO"
    args.ckpt += "-" + args.arch
    args.ckpt += "-track" + args.track
    print("Use MODEL: {} for training".format(args.arch))
    print("Use Track: {} for training".format(args.track))
    print("Use Ratio: {} for training".format(args.ratio))
    print("Save Path: {}".format(args.ckpt))
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
    if args.scheme == 'fdg':
        student_ckpt = args.ckpt+'/student/'
        teacher_ckpt = args.ckpt+'/teacher/'    
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
    if args.scheme=='fdg':
        model_teacher = models.__dict__[args.arch](pretrained=False,num_classes=60)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

# |-Check local weight
    if args.scheme=='fdg':
        model, model_teacher, optimizer, best_acc1, start_epoch = check_local(args,model,model_teacher,optimizer)
    else:
        model, optimizer, best_acc1, start_epoch = check_local(args,model,model,optimizer)

    args.start_epoch = start_epoch

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
    if args.scheme=='fdg':
        model_teacher.cuda(args.gpu)

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

# |-Training initilization
    if args.cos_lr:
        for param_group in optimizer.param_groups:
            param_group['lr']=args.lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epoch, last_epoch=-1)
        for i in range(start_epoch):
            scheduler.step()
            
# |-Data initilization
    if args.track == '2':
        train_dataset1, valid_dataset1 = get_dataset_train(int(args.img_size[0]), 'track2', args.scheme, args.use_seg, args.root, json_path=args.json_path)
        train_dataset2, valid_dataset2 = get_dataset_train(int(args.img_size[1]), 'track2', args.scheme, args.use_seg, args.root, json_path=args.json_path)
        train_dataset3, valid_dataset3 = get_dataset_train(int(args.img_size[2]), 'track2', args.scheme, args.use_seg, args.root, json_path=args.json_path)

    else:
        train_dataset1, valid_dataset1 = get_dataset_train(int(args.img_size[0]), 'track1', args.scheme, args.use_seg, args.root, json_path=args.json_path)
        train_dataset2, valid_dataset2 = get_dataset_train(int(args.img_size[1]), 'track1', args.scheme, args.use_seg, args.root, json_path=args.json_path)
        train_dataset3, valid_dataset3 = get_dataset_train(int(args.img_size[2]), 'track1', args.scheme, args.use_seg, args.root, json_path=args.json_path)

    train_sampler1 = torch.utils.data.distributed.DistributedSampler(train_dataset1)
    train_sampler2 = torch.utils.data.distributed.DistributedSampler(train_dataset2)
    train_sampler3 = torch.utils.data.distributed.DistributedSampler(train_dataset3)

    train_loader1 = torch.utils.data.DataLoader(
        train_dataset1, batch_size=args.batch_size, shuffle=(train_sampler1 is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler1)
    train_loader2 = torch.utils.data.DataLoader(
        train_dataset2, batch_size=args.batch_size, shuffle=(train_sampler2 is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler2)
    train_loader3 = torch.utils.data.DataLoader(
        train_dataset3, batch_size=args.batch_size, shuffle=(train_sampler3 is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler3)

    val_loader1 = torch.utils.data.DataLoader(valid_dataset1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader2 = torch.utils.data.DataLoader(valid_dataset2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader3 = torch.utils.data.DataLoader(valid_dataset3, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

# |-Training and online test
    for epoch in range(args.start_epoch, args.epochs):
        
        if epoch<200:
            train_loader = train_loader3
            val_loader = val_loader3
            train_sampler =train_sampler3
        if epoch<120:
            train_loader = train_loader2
            val_loader = val_loader2
            train_sampler =train_sampler2
        if epoch<60:
            train_loader = train_loader1
            val_loader = val_loader1
            train_sampler =train_sampler1

        train_sampler.set_epoch(epoch)
        if args.rank % ngpus_per_node == 0:
            args.log_file = open(log_file, mode="a")
    
        if args.scheme == 'fdg':
            train_fdg(train_loader, model, model_teacher, criterion, optimizer, epoch, args)
        else:
            if args.swa:
                model_t = deepcopy(model)
                train_decouple(train_loader, model, criterion, optimizer, epoch, args)
                if epoch % args.update_epoch == 0:
                    warm_update(model, model_t, 0.0005, epoch+1)
            else:
                train_decouple(train_loader, model, criterion, optimizer, epoch, args)

        if(args.cos_lr):
            scheduler.step()
            print('[%03d] %.7f'%(epoch, scheduler.get_lr()[0]))
        if args.rank % ngpus_per_node == 0:
            acc1 = validate(val_loader, model, criterion, args)
# |-Save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            args.log_file.close()

            if args.scheme == 'fdg':
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
            else:
                save_checkpoint({
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.module.state_dict(),
                    "best_acc": best_acc1,
                    "optimizer" : optimizer.state_dict(),
                    }, is_best, epoch, save_path=args.ckpt)
            
if __name__ == '__main__':
    main()
