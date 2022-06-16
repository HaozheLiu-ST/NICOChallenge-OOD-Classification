import os
import random
import time
from util import AverageMeter, ProgressMeter, accuracy, warm_update_teacher

from torchvision import transforms
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_ft(optimizer, epoch, args):
    if (epoch+1) == 64:
        lr = args.lr * (0.1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def Sepmixing(image,mask,label,alpha=2.0,p=0.5):
    """
    image: torch tensor with shape [N,C,H,W] without normalization and diving 255
    """
    if alpha > 0:
        lam_fore = np.random.beta(alpha, alpha)
        lam_back = np.random.rand()
    else:
        lam = 1
    batch_size = image.size()[0]
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(batch_size):
        image[i] = norm(image[i].div(255))

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
    """
    img: torch tensor with shape [N,C,H,W] without normalization and diving 255
    ratio 0.1-0.9 0.1 
    """
    lam_low = np.random.beta(alpha, alpha)
    lam_high = np.random.rand()
    n, c, h, w = img.size()
    index = torch.randperm(n).cuda()
    if random.random() < 0.5:

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
    else:
        mix_img = lam_low * img + (1 - lam_low)*img[index,:]
    y_shuffle = y.detach()[index]

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(n):
        mix_img[i] = norm(mix_img[i].div(255))

    return mix_img, y, y_shuffle, lam_low

def MixStyle(x, y, alpha=0.1, eps=1e-6):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """
    """
    Args:
      alpha (float): parameter of the Beta distribution.
      eps (float): scaling parameter to avoid numerical issues.
    """
    beta = torch.distributions.Beta(alpha, alpha)
    B = x.size(0)

    for i in range(B):
        x[i] = x[i].div(255)

    mu = x.mean(dim=[2, 3], keepdim=True)
    var = x.var(dim=[2, 3], keepdim=True)
    sig = (var + eps).sqrt()
    mu, sig = mu.detach(), sig.detach()
    x_normed = (x-mu) / sig

    # lmda = beta.sample((B, 1, 1, 1))
    lmda =  beta.sample()

    perm = torch.randperm(B)
    mu2, sig2 = mu[perm], sig[perm]
    y_shuffle = y.detach()[perm]

    mu_mix = mu*lmda + mu2 * (1-lmda)
    sig_mix = sig*lmda + sig2 * (1-lmda)
    
    mix_image = x_normed*sig_mix + mu_mix
    return mix_image, y, y_shuffle, lmda

def phase_decouple_mix(img,y,alpha=2.0, ratio=1.0):
    """
    img: torch tensor with shape [N,C,H,W] without normalization and diving 255
    """
    lam_phase = np.random.beta(alpha, alpha)
    lam_amplitude = np.random.rand()
    n, c, h, w = img.size()
    index = torch.randperm(n).cuda()
    if random.random() < 0.5:

        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = w // 2 - w_crop // 2
        
        img_shuffle = img.detach()[index,:]

        img_fft = torch.fft.fft2(img,dim=(2,3))

        img_shuffle_fft = torch.fft.fft2(img_shuffle,dim=(2,3))
        
        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)
        
        img_shuffle_abs, img_shuffle_pha = torch.abs(img_shuffle_fft), torch.angle(img_shuffle_fft)

        # high_pass_mask = torch.zeros_like(img_fft)

        # high_pass_mask[:,:,h_start:h_start + h_crop, w_start:w_start + w_crop] = 1

        # low_pass_mask = torch.ones_like(high_pass_mask) - high_pass_mask
        
        mix_pha =  lam_phase * img_pha  + (1-lam_phase) * img_shuffle_pha  
        mix_abs =  lam_amplitude * img_abs + (1-lam_amplitude) * img_shuffle_abs

        mix_abs = torch.fft.ifftshift(mix_abs,dim=(2,3))
    
        mix_img = torch.mul(mix_abs, torch.exp(1j *mix_pha))


        mix_img = torch.real(torch.fft.ifft2(mix_img,dim=(2,3)))
        mix_img = torch.clamp(mix_img, 0, 255).type(torch.uint8)

    else:
        mix_img = lam_phase * img + (1 - lam_phase)*img[index,:]
    
    y_shuffle = y.detach()[index]

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(n):
        mix_img[i] = norm(mix_img[i].div(255))

    return mix_img, y, y_shuffle, lam_phase

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
            if args.style_mix:
                mixed_x, y_a, y_b, lam = MixStyle(images,target)
            else:
                mixed_x, y_a, y_b, lam = Sepmixing(images,mask,target)
        else:
            mixed_x, y_a, y_b, lam = spectrum_decouple_mix(images,target, ratio = args.ratio)
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


# from PIL import Image
# from torchvision.utils import save_image

# def phase_decouple_mix_visual(img,img_shuffle ,alpha=2.0, ratio=1.0):
#     """
#     img: torch tensor with shape [N,C,H,W] without normalization and diving 255
#     """
#     lam_phase = np.random.beta(alpha, alpha)
#     lam_amplitude = np.random.rand()
#     n, c, h, w = img.size()

#     h_crop = int(h * ratio)
#     w_crop = int(w * ratio)
#     h_start = h // 2 - h_crop // 2
#     w_start = w // 2 - w_crop // 2
    
#     img_fft = torch.fft.fft2(img,dim=(2,3))
#     img_shuffle_fft = torch.fft.fft2(img_shuffle,dim=(2,3))
#     img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)
#     img_shuffle_abs, img_shuffle_pha = torch.abs(img_shuffle_fft), torch.angle(img_shuffle_fft)
    
#     mix_pha =  lam_phase * img_pha  + (1-lam_phase) * img_shuffle_pha  
#     mix_abs =  lam_amplitude * img_abs + (1-lam_amplitude) * img_shuffle_abs

#     mix_abs = torch.fft.ifftshift(mix_abs,dim=(2,3))

#     mix_img = torch.mul(mix_abs, torch.exp(1j *mix_pha))

#     mix_img = torch.real(torch.fft.ifft2(mix_img,dim=(2,3)))
#     mix_img = torch.clamp(mix_img, 0, 255).type(torch.uint8)


#     norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     for i in range(n):
#         mix_img[i] = norm(mix_img[i].div(255))

#     return mix_img


# def spectrum_decouple_mix_visual(img, img_shuffle,alpha=2.0, ratio=1.0):
#     """
#     img: torch tensor with shape [N,C,H,W] without normalization and diving 255
#     ratio 0.1-0.9 0.1 
#     """
#     lam_low = np.random.beta(alpha, alpha)
#     lam_high = np.random.rand()
#     n, c, h, w = img.size()

#     h_crop = int(h * ratio)
#     w_crop = int(w * ratio)
#     h_start = h // 2 - h_crop // 2
#     w_start = w // 2 - w_crop // 2
#     print (h_start, h_start + h_crop)
#     img_fft = torch.fft.fft2(img,dim=(2,3))
#     img_fft = torch.fft.fftshift(img_fft,dim=(2,3))

#     img_shuffle_fft = torch.fft.fft2(img_shuffle,dim=(2,3))
#     img_shuffle_fft = torch.fft.fftshift(img_shuffle_fft,dim=(2,3))

#     high_pass_mask = torch.zeros_like(img_fft)

#     high_pass_mask[:,:,h_start:h_start + h_crop, w_start:w_start + w_crop] = 1

#     low_pass_mask = torch.ones_like(high_pass_mask) - high_pass_mask

#     mix_low_fft =  lam_low * img_fft * low_pass_mask  + (1-lam_low) * img_shuffle_fft * low_pass_mask 
#     mix_high_fft =  lam_high * img_fft * high_pass_mask + (1-lam_high) * img_shuffle_fft * high_pass_mask

#     img1_low_fft = img_fft * low_pass_mask
#     img1_low_fft = torch.fft.ifftshift(img1_low_fft,dim=(2,3))
#     img1_low = torch.fft.ifft2(img1_low_fft,dim=(2,3)).float()

#     img1_high_fft = img_fft * high_pass_mask
#     img1_high_fft = torch.fft.ifftshift(img1_high_fft,dim=(2,3))
#     img1_high = torch.fft.ifft2(img1_high_fft,dim=(2,3)).float()

#     img2_low_fft = img_shuffle_fft * low_pass_mask
#     img2_low_fft = torch.fft.ifftshift(img2_low_fft,dim=(2,3))
#     img2_low = torch.fft.ifft2(img2_low_fft,dim=(2,3)).float()

#     img2_high_fft = img_shuffle_fft * high_pass_mask
#     img2_high_fft = torch.fft.ifftshift(img2_high_fft,dim=(2,3))
#     img2_high = torch.fft.ifft2(img2_high_fft,dim=(2,3)).float()

#     mix_fft = mix_low_fft + mix_high_fft
#     mix_fft = torch.fft.ifftshift(mix_fft,dim=(2,3))
#     mix_img = torch.fft.ifft2(mix_fft,dim=(2,3)).float()

#     # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                                  std=[0.229, 0.224, 0.225])
#     for i in range(n):
#         img1_low[i] = img1_low[i].div(255)
#         img1_high[i] = img1_high[i].div(255)
#         img2_low[i] = img2_low[i].div(255)
#         img2_high[i] = img2_high[i].div(255)
#         mix_img[i] = mix_img[i].div(255)
#     print (mix_img.shape)

#     return img1_low, img1_high, img2_low, img2_high, mix_img, lam_low


# if __name__ == '__main__':
#     img1 = Image.open('./flower.jpg').convert('RGB')
#     img2 = Image.open('./fox.jpg').convert('RGB')

#     trans = transforms.Compose([
#             transforms.RandomResizedCrop(512, scale=(0.8, 1)),
#             transforms.RandomHorizontalFlip(),
#             transforms.PILToTensor()])

#     img1 = trans(img1).view(1,3,512,512)
#     img2 = trans(img2).view(1,3,512,512)

#     print (img1.shape)
#     print (img2.shape)

#     index = [1.0]
#     for i in index:
#         path = './visual_results/ratio_'+str(i)+'/'
#         if not os.path.isdir(path):
#             os.makedirs(path)
#         mix_img= phase_decouple_mix_visual(img1, img2, ratio = 1.0)
#         # im1_l, im1_h, im2_l, im2_h, mix_img, lam = spectrum_decouple_mix_visual(img1, img2, ratio=i)
#         # save_image(img1[0].div(255), path+'img1.png')
#         # save_image(img2[0].div(255),path+'img2.png')
#         # save_image(im1_l[0], path+'im1_l.png')
#         # save_image(im1_h[0],path+'im1_h.png')
#         # save_image(im2_l[0], path+'im2_l.png')
#         # save_image(im2_h[0], path+'im2_h.png')
#         save_image(mix_img[0], './mix_img.png')