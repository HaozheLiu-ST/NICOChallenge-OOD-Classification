#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /apdcephfs/share_1290796/waltszhang/NICO_challenge/training_decoupleMix_ddp/main_dg_ddp.py \
-a  densenet121 \
--epochs 500 \
--start-epoch 0 \
--batch-size 128 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt '/apdcephfs/share_1290796/waltszhang/NICO_challenge/training_decoupleMix_ddp/ckpts/epoch_500_AugMix_MOCO_Adaptive_DecoupleMix_1e-1_linear100decay/' \
--track 1 \
--root '/apdcephfs/share_1290796/Datasets/NICO/nico_datasets/' \
--rank 0 \
--num_workers 64 \
--dist-url 'tcp://127.0.0.1:4455' \
--dist-backend 'nccl' \
--world-size 1 \
--moco_pretrained '/apdcephfs/share_1290796/jinhengxie/NICO/train/Scaled-down-self-supervised-learning-main/moco/checkpoints/finetune/224/densenet121_track1/gpus_4_lr_0.03_bs_128_epochs_200_path_train_domain_free/checkpoint_0199.pth.tar' \

