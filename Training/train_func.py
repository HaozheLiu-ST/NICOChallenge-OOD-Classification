import os
import random
import time
from util import AverageMeter, ProgressMeter, accuracy
import numpy as np
import torch
import torch.nn as nn
import random



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def Sepmixing(image,mask,label,alpha=2.0,p=0.5):
    if alpha > 0:
        lam_fore = np.random.beta(alpha, alpha)
        lam_back = np.random.rand()
    else:
        lam = 1

    batch_size = image.size()[0]

    index = torch.randperm(batch_size).cuda()

    foreground = image * mask

    background = image * (1-mask)
    if random.random() < 0.5:
        mixed_fore = lam_fore * foreground + (1-lam_fore) * foreground[index,:]

        mixed_back = lam_back * background + (1-lam_back) * background[index,:]

        mixed_x = mixed_fore + mixed_back
    else:
        mixed_x = lam_fore * image + (1 - lam_fore)*image[index,:]
    y_a, y_b = label, label[index]
    return mixed_x, y_a, y_b, lam_fore


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    param_groups = optimizer.param_groups[0]
    curr_lr = param_groups["lr"]

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        if args.track == '2':
            images, mask, target = data
        else:
            images, mask, target, _ = data

        # measure data loading time
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            target = target.cuda()
            images = images.cuda()
            mask = mask.cuda()
        mixed_x, y_a, y_b, lam = Sepmixing(images,mask,target)
        # compute output
        output = model(mixed_x)
        loss = lam * criterion(output, y_a) + (1-lam) * criterion(output, y_b)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            epoch_msg = progress.get_message(i)
            epoch_msg += ("\tLr  {:.4f}".format(curr_lr))
            print(epoch_msg)

        if i % args.log_freq == 0:
            args.log_file.write(epoch_msg + "\n")



def validate(val_loader, model, criterion, args):
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
            if args.track == '2':
                images, mask, target = data
            else:
                images, mask, target, domain = data
            images = images.cuda()
            target = target.cuda()
            # compute outputs
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
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

        args.log_file.write(epoch_msg + "\n")


    return top1.avg
