#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python3 main_dg.py \
-a  resnet50 \
--epochs 180 \
--start-epoch 0 \
--batch-size 32 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt './ckpts/epoch_180_AugMix_DecoupleMix_MOCO_1200_pretrained_FDG_512_lr1e-1_CoseDecay/' \
--track 1 \
--parallel \
--cos_lr \
--cross \
--root '/apdcephfs/share_1290796/Datasets/NICO/' \
--pretrained '/apdcephfs/share_1290796/jinhengxie/NICO/train/Scaled-down-self-supervised-learning-main/moco/checkpoints/finetune/224/resnet50_track1/gpus_4_lr_0.03_bs_128_epochs_1200_path_train_domain_free/checkpoint_1199.pth.tar' ;

CUDA_VISIBLE_DEVICES=3 python3 /apdcephfs/share_1290796/haozheliu/code_X/code/train_for_cifar.py


