import os
import random
import time
from util import AverageMeter, ProgressMeter, accuracy, warm_update_teacher, get_current_consistency_weight

from torchvision import transforms
from math import sqrt
import numpy as np
import torch
import torch.nn.functional as F



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_ft(optimizer, epoch, args):
    if (epoch+1) == 64:
        lr = args.lr * (0.1)
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

def spectrum_decouple_mix(img,y,alpha=2.0, ratio=1.0):
    lam_low = np.random.beta(alpha, alpha)
    lam_high = np.random.rand()
    n, c, h, w = img.size()
    index = torch.randperm(n).cuda()
    if random.random() < 0.5:

        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w // 2 - w_crop // 2
        
        img_shuffle = img.detach()[index,:]

        img_fft = torch.fft.fft2(img,dim=(2,3))
        img_fft = torch.fft.fftshift(img_fft,dim=(2,3))

        img_shuffle_fft = torch.fft.fft2(img_shuffle,dim=(2,3))
        img_shuffle_fft = torch.fft.fftshift(img_shuffle_fft,dim=(2,3))

        high_pass_mask = torch.zeros_like(img_fft)

        high_pass_mask[:,:,h_start:h_start + h_crop, w_start:w_start + w_crop] = 1

        low_pass_mask = torch.ones_like(high_pass_mask) - high_pass_mask

        mix_low_fft =  lam_low * img_fft * low_pass_mask  + (1-lam_low) * img_shuffle_fft * low_pass_mask 
        mix_high_fft =  lam_high * img_fft * high_pass_mask + (1-lam_high) * img_shuffle_fft * high_pass_mask

        mix_fft = mix_low_fft + mix_high_fft

        mix_fft = torch.fft.ifftshift(mix_fft,dim=(2,3))
        mix_img = torch.fft.ifft2(mix_fft,dim=(2,3)).float()
    else:
        mix_img = lam_low * img + (1 - lam_low)*img[index,:]
    y_shuffle = y.detach()[index]
    return mix_img, y, y_shuffle, lam_low



def train_dg_pdd(train_loader, model, model_teacher, criterion, optimizer, epoch, args):
    T=10.0
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
    # model_teacher.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        global_step = epoch * len(train_loader) + ( i + 1 )
        images, mask, target, _ = data


        # measure data loading time
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu)
            images = images.cuda(args.gpu)
            mask = mask.cuda(args.gpu)
        if random.random() < 0.5:
            mixed_x, y_a, y_b, lam = Sepmixing(images,mask,target)
        else:
            mixed_x, y_a, y_b, lam = spectrum_decouple_mix(images,target)
        # compute output
        # zero grad
        optimizer.zero_grad()

        # forward
        scores = model(mixed_x)
        # calculate total loss
        total_loss = lam * criterion(scores, y_a) + (1-lam) * criterion(scores, y_b)


        # backward
        total_loss.backward()
        # update
        optimizer.step()

        # update teachers
        warm_update_teacher(model, model_teacher, 0.9995, global_step)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(scores, target, topk=(1, 5))
        losses.update(total_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            epoch_msg = progress.get_message(i)
            epoch_msg += ("\tLr  {:.4f}".format(curr_lr))
            print(epoch_msg)

        if i % args.log_freq == 0 and args.rank % args.ngpus_per_node == 0:
            args.log_file.write(epoch_msg + "\n")

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
        args.log_file.write(epoch_msg + "\n")
    return top1.avg