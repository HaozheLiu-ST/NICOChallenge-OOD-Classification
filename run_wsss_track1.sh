#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py \
-a  resnet50 \
--epochs 90 \
--total_epoch 90 \
--start-epoch 0 \
--batch-size 512 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt './ckpts/epoch_90_baseline_mix_up_1e-1_cos_lr/' \
--scheme 'decouple' \
--track 1 \
--root './data/' \
--json_path './dataset_json/' \
--rank 0 \
--num_workers 64 \
--dist-url 'tcp://127.0.0.1:4455' \
--dist-backend 'nccl' \
--world-size 1 \
--cos_lr \
--ratio 1.0 \
--img_size 224 224 224 ;

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 ./WSSS/run_track1.py \
--data_dir ./data/ \
--json_file ./dataset_json/track1_train_label.json \
--ppath ./ckpts/epoch_90_baseline_mix_up_1e-1_cos_lr/NICO-resnet50-track1/model_best_in_090_epochs.pth.tar \
