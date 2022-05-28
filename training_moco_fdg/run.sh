#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1,2 python3 main_dg.py \
-a  resnet34 \
--epochs 180 \
--start-epoch 0 \
--batch-size 64 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt './ckpts/epoch_180_AugMix_DecoupleMix_MOCO_pretrained_FDG_512_lr1e-1_CoseDecay/' \
--track 1 \
--parallel \
--cos_lr \
--cross \
--root '/apdcephfs/share_1290796/Datasets/NICO/' \
--pretrained '/apdcephfs/share_1290796/jinhengxie/NICO/train/Scaled-down-self-supervised-learning-main/moco/checkpoints/finetune/224/resnet34_track1/gpus_4_lr_0.03_bs_128_epochs_200_path_train_domain_free/checkpoint_0199.pth.tar' \

CUDA_VISIBLE_DEVICES=3 python3 /apdcephfs/share_1290796/haozheliu/code_X/code/train_for_cifar.py


