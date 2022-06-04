#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python3 /apdcephfs/share_1290796/haozheliu/NICO/test_decouple/main_dg_ddp.py \
-a  densenet121 \
--epochs 180 \
--start-epoch 0 \
--batch-size 128 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt './ckpts/Adaptive_DecoupleMix/' \
--track 1 \
--cos_lr \
--root '/apdcephfs/share_1290796/Datasets/NICO/nico_datasets/' \
--rank 0 \
--num_workers 64 \
--dist-url 'tcp://127.0.0.1:4455' \
--dist-backend 'nccl' \
--world-size 1 \




