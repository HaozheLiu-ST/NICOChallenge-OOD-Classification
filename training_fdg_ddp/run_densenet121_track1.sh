#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /apdcephfs/share_1290796/waltszhang/NICO_challenge/train_dg_ddp/main_dg_ddp.py \
-a  densenet121 \
--epochs 180 \
--start-epoch 0 \
--batch-size 64 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt '/apdcephfs/share_1290796/waltszhang/NICO_challenge/train_dg_ddp/ckpts/epoch_180_AugMix_DecoupleMix_FDG_512_lr1e-1_CoseDecay/' \
--track 1 \
--cos_lr \
--root '/apdcephfs/share_1290796/Datasets/NICO/nico_datasets/' \
--rank 0 \
--num_workers 64 \
--dist-url 'tcp://127.0.0.1:4455' \
--dist-backend 'nccl' \
--world-size 1 \




