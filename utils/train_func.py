import os
import random
import time
from utils.util import AverageMeter, ProgressMeter, accuracy, warm_update_teacher, get_current_consistency_weight
from torchvision import transforms
from math import sqrt
import numpy as np
import torch
import torch.nn.functional as F


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 60 epochs"""
    lr = args.lr * (0.1 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

"""
domain generalization proposess
"""

def mix_up(img, y, alpha=2.0):
    """
    img: torch tensor with shape [N,C,H,W] with normalization and diving 255
    """
    lam_low = np.random.beta(alpha, alpha)
    n, c, h, w = img.size()
    index = torch.randperm(n).cuda()
    mix_img = lam_low * img + (1 - lam_low)*img[index,:]
    y_shuffle = y.detach()[index]
    return mix_img, y, y_shuffle, lam_low
def sepmixing(image,mask,label,scheme='fdg',alpha=2.0,p=0.5):
    """
    if scheme is 'decouple' image: torch tensor with shape [N,C,H,W] without normalization and diving 255
    """
    if alpha > 0:
        lam_fore = np.random.beta(alpha, alpha)
        lam_back = np.random.rand()
    else:
        lam = 1

    batch_size = image.size()[0]

    if scheme == 'decouple':
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        for i in range(batch_size):
            image[i] = norm(image[i].div(255))

    index = torch.randperm(batch_size).cuda()

    foreground = image * mask
    background = image * (1-mask)
    mixed_fore = lam_fore * foreground + (1-lam_fore) * foreground[index,:]
    mixed_back = lam_back * background + (1-lam_back) * background[index,:]
    mixed_x = mixed_fore + mixed_back
    y_a, y_b = label, label[index]
    return mixed_x, y_a, y_b, lam_fore
def spectrum_decouple_mix(img,y,alpha=2.0, ratio=1.0):
    """
    img: torch tensor with shape [N,C,H,W] without normalization and diving 255
    ratio 0.1-0.9 0.1 
    """
    lam_low = np.random.beta(alpha, alpha)
    lam_high = np.random.rand()
    n, c, h, w = img.size()
    index = torch.randperm(n).cuda()
    h_crop = int(h * ratio)
    w_crop = int(w * ratio)
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2
    
    img_shuffle = img.detach()[index,:]

    img_fft = torch.fft.fft2(img,dim=(2,3))
    img_fft = torch.fft.fftshift(img_fft,dim=(2,3))

    img_shuffle_fft = torch.fft.fft2(img_shuffle,dim=(2,3))
    img_shuffle_fft = torch.fft.fftshift(img_shuffle_fft,dim=(2,3))

    low_pass_mask = torch.zeros_like(img_fft)

    low_pass_mask[:,:,h_start:h_start + h_crop, w_start:w_start + w_crop] = 1

    high_pass_mask = torch.ones_like(low_pass_mask) - low_pass_mask

    mix_low_fft =  lam_low * img_fft * low_pass_mask  + (1-lam_low) * img_shuffle_fft * low_pass_mask 
    mix_high_fft =  lam_high * img_fft * high_pass_mask + (1-lam_high) * img_shuffle_fft * high_pass_mask

    mix_fft = mix_low_fft + mix_high_fft

    mix_fft = torch.fft.ifftshift(mix_fft,dim=(2,3))
    mix_img = torch.fft.ifft2(mix_fft,dim=(2,3)).float()
    y_shuffle = y.detach()[index]

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(n):
        mix_img[i] = norm(mix_img[i].div(255))

    return mix_img, y, y_shuffle, lam_low
def fourier_spectrum_mix(img1, img2, alpha=1.0, ratio=1.0):
    """Input image size: PIL of [H, W, C]"""
    lam = random.uniform(0, alpha)

    assert img1.shape == img2.shape
    c, h, w = img1.shape

    h_crop = int(h * ratio)
    w_crop = int(w * ratio)
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
        img2_abs[i] = torch.fft.ifftshift(img2_abs[i])

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
def get_fourir_transfer_data(images, mask, target, args):

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
        img_s2o, img_o2s = fourier_spectrum_mix(images_[i], images_s[i], ratio = args.ratio)
        images_so_list.append(img_s2o.view(1, C, H, W))
        images_os_list.append(img_o2s.view(1, C, H, W))
        mask_so_list.append(mask_[i].view(1, 1, H, W))
        mask_os_list.append(mask_s[i].view(1, 1, H, W))

        target_so_list.append(target_[i])
        target_os_list.append(target_s[i])

    images_aug = torch.cat([torch.cat(images_so_list), torch.cat(images_os_list)])
    mask_aug = torch.cat([torch.cat(mask_so_list), torch.cat(mask_os_list)])
    target_aug = torch.cat([torch.tensor(target_so_list), torch.tensor(target_os_list)]).cuda()
    images = torch.cat([images, images_aug])
    for i in range(2*B):
        images[i] = norm(images[i].div(255))
    mask = torch.cat([mask, mask_aug])
    target = torch.cat([target, target_aug])
    return images, mask, target


"""
train and validate one epoch 
"""
def train_decouple(train_loader, model, criterion, optimizer, epoch, args):
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

    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        global_step = epoch * len(train_loader) + ( i + 1 )

        if args.use_seg:
            images, mask, target, _ = data
        else:
            images, target, _ = data

        # measure data loading time
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu)
            images = images.cuda(args.gpu)
            if args.use_seg:
                mask = mask.cuda(args.gpu)
        if args.use_seg:
            if random.random() < 0.5:
                mixed_x, y_a, y_b, lam = sepmixing(images,mask,target,scheme=args.scheme)
            else:
                mixed_x, y_a, y_b, lam = spectrum_decouple_mix(images,target, ratio = args.ratio)
        else:
            mixed_x, y_a, y_b, lam = mix_up(images,target)
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
def train_fdg(train_loader, model, model_teacher, criterion, optimizer, epoch, args):
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
            target = target.cuda(args.gpu)
            images = images.cuda(args.gpu)
            mask = mask.cuda(args.gpu)

        images, mask, target = get_fourir_transfer_data(images, mask, target, args)
        mixed_x, y_a, y_b, lam = sepmixing(images,mask,target,scheme=args.scheme)
        # compute output
        # zero grad
        optimizer.zero_grad()

        # forward
        scores = model(mixed_x)

        assert mixed_x.size(0) % 2 == 0
        split_idx = int(mixed_x.size(0) / 2)
        scores_ori, scores_aug = torch.split(scores, split_idx)
        labels_ori_a, labels_aug_a = torch.split(y_a, split_idx)
        labels_ori_b, labels_aug_b = torch.split(y_b, split_idx)

        with torch.no_grad():
            scores_teacher = model_teacher(mixed_x)
            scores_ori_tea, scores_aug_tea = torch.split(scores_teacher, split_idx)
            scores_ori_tea, scores_aug_tea = scores_ori_tea.detach(), scores_aug_tea.detach()

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
            if args.use_seg:
                images, mask, target = data
            else:
                images, target = data
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

        epoch_msg = '----------- Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} -----------'.format(top1=top1, top5=top5)
        print(epoch_msg)
        args.log_file.write(epoch_msg + "\n")
    return top1.avg

