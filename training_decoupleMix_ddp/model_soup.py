import argparse
import os
import torchvision.models as models
from checkpoint import save_checkpoint
from dataset_seg import get_dataset_train_cross
from util import accuracy, AverageMeter, ProgressMeter
import torch
import copy
import time
parser = argparse.ArgumentParser(description='PyTorch Model Soup')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='[resnet50/densenet121/mobilenet_v2/wide_resnet50_2]')
parser.add_argument('--ckpts',nargs='+',help='the list for model ckpts')
parser.add_argument("--root", default='', type=str, metavar='PATH',
                    help='root path to load datasets')
parser.add_argument("--track",default="1",help='[1/2]')
parser.add_argument("--save_path",default="./soup_model/",help='the path for gwa')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()


model = models.__dict__[args.arch](pretrained=False,num_classes=60)

args.save_path += "NICO"
args.save_path += "-" + args.arch
args.save_path += "-track" + args.track


if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

if args.track == '2':
    _, valid_dataset = get_dataset_train_cross('track2','track1', args.root, 128)
else:
    _, valid_dataset = get_dataset_train_cross('track1','track2', args.root, 128)

val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

criterion = torch.nn.CrossEntropyLoss().cuda()

args.gpu = 0

def merge_model(model_1, model_2, lam=0.5):
    for ema_param, param in zip(model_1.parameters(), model_2.parameters()):
        ema_param.data.mul_(lam).add_(1 - lam, param.data)


def validate_dg_pdd(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, mask, target = data
            images = images.cuda(args.gpu)
            target = target.cuda(args.gpu)

            # compute outputs
            scores = model(images)
            loss = criterion(scores, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(scores, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                epoch_msg = progress.get_message(i)
                print(epoch_msg)

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))
        epoch_msg = '----------- Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} -----------'.format(top1=top1, top5=top5)
        print(epoch_msg)
    return top1.avg

for idx,model_path in enumerate(args.ckpts):
    if idx == 0:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)
        model_dict.update(pretrained_dict['state_dict'])
        dict_ = {}
        for k in list(model_dict.keys()):
            if 'module' in k:
                dict_[k[7:]] = model_dict[k]
            else:
                dict_[k] = model_dict[k]
        model.load_state_dict(dict_, True)
        model.cuda()
        print("=> loading checkpoint '{}')"
            .format(model_path))
        best_acc = validate_dg_pdd(val_loader,model,criterion,args)
    else:
        model_new = models.__dict__[args.arch](pretrained=False,num_classes=60)
        model_dict = model_new.state_dict()
        pretrained_dict = torch.load(model_path)
        model_dict.update(pretrained_dict['state_dict'])
        dict_ = {}
        for k in list(model_dict.keys()):
            if 'module' in k:
                dict_[k[7:]] = model_dict[k]
            else:
                dict_[k] = model_dict[k]
        model_new.load_state_dict(dict_, True)
        model_new.cuda()

        model_old = copy.deepcopy(model)
        model_old.cuda()
        
        merge_model(model,model_new)
        acc = validate_dg_pdd(val_loader,model,criterion,args)
        if acc< best_acc:
            model = model_old.cuda()
save_checkpoint({
    "epoch":None,
    "arch": args.arch,
    "state_dict":model.state_dict(),
    "best_acc": best_acc,
    "optimizer" : None,
    }, True, 0, save_path=args.save_path)





