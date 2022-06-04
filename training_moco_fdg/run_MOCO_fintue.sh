#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1,2 python3 main.py \
-a  resnet34 \
--epochs 180 \
--start-epoch 0 \
--batch-size 256 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt './ckpts/epoch_180_SepMixing_fulldata_MOCO_pretrained_512_Ensemble_lr_1e-1/' \
--track 1 \
--parallel \
--cos_lr \
--fine_tune \
--root '/apdcephfs/share_1290796/Datasets/NICO/' \
--pretrained '/apdcephfs/share_1290796/jinhengxie/NICO/train/Scaled-down-self-supervised-learning-main/moco/checkpoints/finetune/224/resnet34_track1/gpus_4_lr_0.03_bs_128_epochs_200_path_train_domain_free/checkpoint_0199.pth.tar' \

CUDA_VISIBLE_DEVICES=1,2,3,4 python3 main.py \
-a  resnet50 \
--epochs 180 \
--start-epoch 0 \
--batch-size 256 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt './ckpts/epoch_180_SepMixing_fulldata_MOCO_pretrained_512_Ensemble_lr_1e-1/' \
--track 2 \
--parallel \
--cos_lr \
--fine_tune \
--root '/apdcephfs/share_1290796/Datasets/NICO/' \
--pretrained '/apdcephfs/share_1290796/jinhengxie/NICO/train/Scaled-down-self-supervised-learning-main/moco/checkpoints/finetune/224/resnet50_track2/gpus_4_lr_0.03_bs_128_epochs_200_path_/checkpoint_0199.pth.tar' \


CUDA_VISIBLE_DEVICES=3 python3 /apdcephfs/share_1290796/haozheliu/code_X/code/train_for_cifar.py


