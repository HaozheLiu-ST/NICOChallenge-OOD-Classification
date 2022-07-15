#!/usr/bin/env bash
export root_path='./data/' ;

python3 ./moco/main_moco_pretraining_phase2.py \
-a densenet121 \
--workers 32 \
--lr 0.03 \
--batch-size 128 \
--epochs 200 \
--input-size 224 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--gpus 0,1,2,3 \
--mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
--track track1 \
"${root_path}public_dg_0416/train/" ;

python3 ./moco/main_moco_pretraining_phase2.py \
-a resnet34 \
--workers 32 \
--lr 0.03 \
--batch-size 128 \
--epochs 200 \
--input-size 224 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--gpus 0,1,2,3 \
--mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
--track track1 \
"${root_path}public_dg_0416/train/" ;

python3 ./moco/main_moco_pretraining_phase2.py \
-a wide_resnet50_2 \
--workers 32 \
--lr 0.03 \
--batch-size 128 \
--epochs 200 \
--input-size 224 \
--dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
--gpus 0,1,2,3 \
--mlp --moco-t 0.2 --moco-k 4096 --aug-plus --cos \
--track track1 \
"${root_path}public_dg_0416/train/" ;