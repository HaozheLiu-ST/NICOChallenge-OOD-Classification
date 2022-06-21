#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3,4 python3 main.py \
-a  resnet34 \
--epochs 180 \
--start-epoch 0 \
--batch-size 256 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt ./ckpts/epoch_180_full_data_SepMixing_DeepAug_448_Ensemble/ \
--track 2 \
--fine_tune \
--parallel \
--deepaug \

CUDA_VISIBLE_DEVICES=3 python3 /apdcephfs/share_1290796/haozheliu/code_X/code/train_for_cifar.py
