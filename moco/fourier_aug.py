from math import sqrt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import random

def spectrum_decouple_mix_troch(img, img_shuffle, alpha=2.0, ratio=None):
    """
    img: torch tensor with shape [N,C,H,W] without normalization and diving 255
    ratio 0.1-0.9 0.1
    """
    if ratio is None:
        ratio = np.clip(np.random.normal(0.5, 0.2, 1), 0, 1)

    n, c, h, w = img.size()

    h_crop = int(h * ratio)
    w_crop = int(w * ratio)
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img_fft = torch.fft.fft2(img,dim=(2,3))
    img_fft = torch.fft.fftshift(img_fft,dim=(2,3))

    img_shuffle_fft = torch.fft.fft2(img_shuffle,dim=(2,3))
    img_shuffle_fft = torch.fft.fftshift(img_shuffle_fft,dim=(2,3))

    high_pass_mask = torch.zeros_like(img_fft)

    high_pass_mask[:,:,h_start:h_start + h_crop, w_start:w_start + w_crop] = 1

    low_pass_mask = torch.ones_like(high_pass_mask) - high_pass_mask

    img1_low_fft = img_fft * low_pass_mask
    img1_low_fft = torch.fft.ifftshift(img1_low_fft,dim=(2,3))
    img1_low = torch.fft.ifft2(img1_low_fft,dim=(2,3)).float()

    img2_low_fft = img_shuffle_fft * low_pass_mask
    img2_low_fft = torch.fft.ifftshift(img2_low_fft,dim=(2,3))
    img2_low = torch.fft.ifft2(img2_low_fft,dim=(2,3)).float()

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img1_low = norm(img1_low.div(255))
    img2_low = norm(img2_low.div(255))

    return img1_low, img2_low

def colorful_spectrum_mix_torch(img1, img2, alpha=1.0, ratio=1.0):
    """Input image size: PIL of [H, W, C]"""

    assert img1.shape == img2.shape

    img1_fft = torch.fft.fft2(img1)
    img2_fft = torch.fft.fft2(img2)

    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)

    # for i in range (img1_abs.shape[0]):
    img1_abs = torch.fft.fftshift(img1_abs)
    img2_abs = torch.fft.fftshift(img2_abs)

    # for i in range (img1_abs.shape[0]):
    img1_abs = torch.fft.ifftshift(img1_abs)
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

if __name__ == '__main__':

    img1 = Image.open('./flower.jpg').convert('RGB')
    img2 = Image.open('./fox.jpg').convert('RGB')

    trans = transforms.Compose([
            transforms.RandomResizedCrop(512, scale=(0.8, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor()])

    img1 = trans(img1).view(1,3,512,512)
    img2 = trans(img2).view(1,3,512,512)

    print (img1.shape)
    print (img2.shape)

    index = [1.0]
    for i in index:
        path = './visual_results/ratio_'+str(i)+'/'
        if not os.path.isdir(path):
            os.makedirs(path)
        im1_l, im1_h, im2_l, im2_h, mix_img, lam = spectrum_decouple_mix_visual(img1, img2, ratio=0.2)
        # save_image(img1[0].div(255), path+'img1.png')
        # save_image(img2[0].div(255),path+'img2.png')
        save_image(im1_l[0], path+'im1_l.png')
        save_image(im1_h[0],path+'im1_h.png')
        save_image(im2_l[0], path+'im2_l.png')
        save_image(im2_h[0], path+'im2_h.png')
        # save_image(mix_img[0], './mix_img.png')

    img1 = Image.open('./flower.jpg').convert('RGB')
    img2 = Image.open('./fox.jpg').convert('RGB')
    trans = transforms.Compose([
        transforms.RandomResizedCrop(512, scale=(0.8, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor()])

    img1 = trans(img1)
    img2 = trans(img2)
    img1, img2 = colorful_spectrum_mix_torch(img1, img2)
    save_image(img1, path + 'img1.png')
    save_image(img2, path + 'img2.png')