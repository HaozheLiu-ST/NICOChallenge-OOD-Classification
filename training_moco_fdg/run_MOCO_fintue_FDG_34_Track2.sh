#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3,4 python3 main_dg.py \
-a  resnet34 \
--epochs 80 \
--start-epoch 0 \
--batch-size 64 \
--lr 1e-3 \
--wd 5e-4 \
--ckpt './ckpts/epoch_180_SepMixing_fulldata_MOCO_finetune_FDG_512_Ensemble_lr_1e-1_CoseDecay/' \
--track 2 \
--parallel \
--cross \
--root '/tmp/NICO/' \
--fdg_finetune \
--pretrained '/apdcephfs/share_1290796/waltszhang/NICO_challenge/train/ckpts/epoch_180_SepMixing_fulldata_MOCO_pretrained_512_Ensemble_lr_1e-1/NICO-resnet34-track2/checkpoint_120.pth.tar' ;

CUDA_VISIBLE_DEVICES=3 python3 /apdcephfs/share_1290796/haozheliu/code_X/code/train_for_cifar.py