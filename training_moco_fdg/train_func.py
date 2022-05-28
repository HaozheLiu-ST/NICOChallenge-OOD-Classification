import os
import random
import time
from util import AverageMeter, ProgressMeter, accuracy, preprocess_teacher, warm_update_teacher, get_current_consistency_weight

from torchvision import transforms
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random



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

def colorful_spectrum_mix_torch(img1, img2, alpha=1.0, ratio=1.0):
    """Input image size: PIL of [H, W, C]"""
    lam = random.uniform(0, alpha)

    assert img1.shape == img2.shape
    c, h, w = img1.shape

    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2


    img1_fft = torch.fft.fft2(img1)
    img2_fft = torch.fft.fft2(img2)
    
    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)
    
    for i in range (img1_abs.shape[0]):
        img1_abs[i] = torch.fft.fftshift(img1_abs[i])
        img2_abs[i] = torch.fft.fftshift(img2_abs[i])
    
    
    
    img1_abs_ = torch.clone(img1_abs)
    img2_abs_ = torch.clone(img2_abs)

    img1_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        torch.add(torch.mul(lam, img2_abs_[:,h_start:h_start + h_crop, w_start:w_start + w_crop]), torch.mul((1 - lam), img1_abs_[:,
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]))
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        torch.add(torch.mul(lam, img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop]), torch.mul((1 - lam), img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]))

    for i in range (img1_abs.shape[0]):
        img1_abs[i] = torch.fft.ifftshift(img1_abs[i])
        img2_abs = torch.fft.ifftshift(img2_abs)

    img21 = torch.mul(img1_abs, torch.exp(1j *img1_pha))
    img12 = torch.mul(img2_abs, torch.exp(1j *img2_pha))
    
    img21 = torch.real(torch.fft.ifft2(img21))
    img12 = torch.real(torch.fft.ifft2(img12))
    img21 = torch.clamp(img21, 0, 255).type(torch.uint8).div(255)
    img12 = torch.clamp(img12, 0, 255).type(torch.uint8).div(255)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    img21 = norm(img21)
    img12 = norm(img12)
    return img21, img12

def get_fourir_transfer_data(images, mask, target):

    split_idx = int(images.size(0) / 2)
    assert images.size(0) / 2 == mask.size(0) / 2 == target.size(0) / 2
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])

    images_, images_s = torch.split(images, split_idx)
    mask_, mask_s = torch.split(mask, split_idx)
    target_, target_s = torch.split(target, split_idx)

    B, C, H, W = images_.shape

    images_so_list, images_os_list= [], []
    mask_so_list, mask_os_list= [], []
    target_so_list, target_os_list= [], []

    for i in range(B):
        img_s2o, img_o2s = colorful_spectrum_mix_torch(images_[i], images_s[i])
        images_so_list.append(img_s2o.view(1, C, H, W))
        images_os_list.append(img_o2s.view(1, C, H, W))
        mask_so_list.append(mask_[i].view(1, 1, H, W))
        mask_os_list.append(mask_s[i].view(1, 1, H, W))

        target_so_list.append(target_[i])
        target_os_list.append(target_s[i])

    for i in range(2*B):
        images[i] = norm(images[i].div(255))

    images_aug = torch.cat([torch.cat(images_so_list), torch.cat(images_os_list)])
    mask_aug = torch.cat([torch.cat(mask_so_list), torch.cat(mask_os_list)])
    target_aug = torch.cat([torch.tensor(target_so_list), torch.tensor(target_os_list)]).cuda()
    images = torch.cat([images, images_aug])
    mask = torch.cat([mask, mask_aug])
    target = torch.cat([target, target_aug])
    return images, mask, target

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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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

def train_dg(train_loader, model, model_teacher, criterion, optimizer, epoch, args):
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
    model_teacher.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        global_step = epoch * len(train_loader) + ( i + 1 )
        images, mask, target, _ = data

        images = torch.cat(images, dim=0)
        mask = torch.cat(mask, dim=0)
        target = torch.cat(target, dim=0)


        # measure data loading time
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            target = target.cuda()
            images = images.cuda()
            mask = mask.cuda()

        images, mask, target = get_fourir_transfer_data(images, mask, target)
        mixed_x, y_a, y_b, lam = Sepmixing(images,mask,target)
        # compute output
        # zero grad
        optimizer.zero_grad()

        # forward
        scores = model(mixed_x)
        with torch.no_grad():
            scores_teacher = model_teacher(mixed_x)

        assert mixed_x.size(0) % 2 == 0
        split_idx = int(mixed_x.size(0) / 2)
        scores_ori, scores_aug = torch.split(scores, split_idx)
        scores_ori_tea, scores_aug_tea = torch.split(scores_teacher, split_idx)
        scores_ori_tea, scores_aug_tea = scores_ori_tea.detach(), scores_aug_tea.detach()
        labels_ori_a, labels_aug_a = torch.split(y_a, split_idx)
        labels_ori_b, labels_aug_b = torch.split(y_b, split_idx)

        assert scores_ori.size(0) == scores_aug.size(0)

        # classification loss for original data
        loss_cls = lam * criterion(scores_ori, labels_ori_a) + (1-lam) * criterion(scores_ori, labels_ori_b)

        # classification loss for augmented data
        loss_aug = lam * criterion(scores_aug, labels_aug_a) + (1-lam) * criterion(scores_aug, labels_aug_b)
        # calculate probability
        p_ori, p_aug = F.softmax(scores_ori / T, dim=1), F.softmax(scores_aug / T, dim=1)
        p_ori_tea, p_aug_tea = F.softmax(scores_ori_tea / T, dim=1), F.softmax(scores_aug_tea / T, dim=1)

        # use KLD for consistency loss
        loss_ori_tea = F.kl_div(p_aug.log(), p_ori_tea, reduction='batchmean')
        loss_aug_tea = F.kl_div(p_ori.log(), p_aug_tea, reduction='batchmean')

        # get consistency weight
        const_weight = get_current_consistency_weight(epoch=epoch,
                                                      weight=2.0,
                                                      rampup_length=5,
                                                      rampup_type='sigmoid')

        # calculate total loss
        total_loss = 0.5 * loss_cls + 0.5 * loss_aug + \
                     const_weight * loss_ori_tea + const_weight * loss_aug_tea

        # loss_dict["main"] = loss_cls.item()
        # loss_dict["aug"] = loss_aug.item()
        # loss_dict["ori_tea"] = loss_ori_tea.item()
        # loss_dict["aug_tea"] = loss_aug_tea.item()
        # loss_dict["total"] = total_loss.item()

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

        if i % args.log_freq == 0:
            args.log_file.write(epoch_msg + "\n")

def validate_dg(val_loader, model, criterion, args):
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
            images = images.cuda()
            target = target.cuda()

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