#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=2 python3 offline_evaluation_track.py \
--arch resnet34 \
--out NICO-resnet34_full_data_SepMixing_Pretrained_FGD_512_Ensemble_160_ \
--track track1 \
--cfg None \
--model_path /apdcephfs/share_1290796/haozheliu/NICO/Separate_Mix/ckpts/epoch_180_full_data_SepMixing_DeepAug_448_Ensemble/NICO-resnet34-track1/checkpoint_160.pth.tar 
