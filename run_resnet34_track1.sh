#!/usr/bin/env bash
export root_path='./data/' ;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ./main.py \
-a  resnet34 \
--epochs 120 \
--total_epoch 200 \
--start-epoch 0 \
--batch-size 128 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt './ckpts/epoch_120_MOCO_DecoupleMix_224_448_SWA_1e-1_cos_lr/' \
--track 1 \
--root ${root_path} \
--rank 0 \
--num_workers 64 \
--dist-url 'tcp://127.0.0.1:4455' \
--dist-backend 'nccl' \
--world-size 1 \
--cos_lr \
--ratio 1.0 \
--img_size 224 448 512 \
--scheme 'decouple' \
--use_seg \
--swa \
--moco_pretrained './checkpoints/finetune/224/resnet34_track1_4096_0.03_128/gpus_4_lr_0.03_bs_128_epochs_200/checkpoint_0199.pth.tar' ;

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ./main.py \
-a  resnet34 \
--epochs 200 \
--total_epoch 200 \
--start-epoch 0 \
--batch-size 24 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt './ckpts/epoch_120-200_MOCO_FDG_512_1e-1_cos_lr/' \
--track 1 \
--root ${root_path} \
--rank 0 \
--num_workers 64 \
--dist-url 'tcp://127.0.0.1:4455' \
--dist-backend 'nccl' \
--world-size 1 \
--cos_lr \
--ratio 0.2 \
--scheme 'fdg' \
--use_seg \
--img_size 512 512 512 \
--fine_tune './ckpts/epoch_120_MOCO_DecoupleMix_224_448_SWA_1e-1_cos_lr/NICO-resnet34-track1/model_best_out_090_epochs.pth.tar' ;

CUDA_VISIBLE_DEVICES=1 python3 evaluation.py \
--arch resnet34 \
--out resnet34_fdg_200_bestout90 \
--track track1 \
--cfg mixinginference \
--save_path './results/' \
--model_path './ckpts/epoch_120-200_MOCO_FDG_512_1e-1_cos_lr/NICO-resnet34-track1/teacher/model_best_out_090_epochs.pth.tar' ;

for i in $(seq 0.1 0.1 1.0); 
do 
    echo "Saved Path: ${save_path}ratio_${i}_lr_${learning_rate}";
    echo "Learning Rate: ${learning_rate}";
    echo "Ratio: ${i}";
    echo "Pretrained_Path: ${pretrained_path}"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ./main.py \
        -a  resnet34 \
        --epochs 20 \
        --start-epoch 0 \
        --batch-size 128 \
        --lr 1e-4 \
        --wd 5e-4 \
        --ratio ${i} \
        --ckpt "./ckpts/Adaptive_Decouple_SWA_from_best_epoch/ratio_${i}_lr_1e-4/" \
        --track 1 \
        --root ${root_path} \
        --swa \
        --scheme 'decouple' \
		--use_seg \
        --img_size 512 512 512 \
        --rank 0 \
        --num_workers 32 \
        --dist-url 'tcp://127.0.0.1:4455' \
        --dist-backend 'nccl' \
        --world-size 1 \
        --pretrained './ckpts/epoch_120-200_MOCO_FDG_512_1e-1_cos_lr/NICO-resnet34-track1/teacher/model_best_out_090_epochs.pth.tar' ;
    
    CUDA_VISIBLE_DEVICES=1 python3 evaluation.py \
    --arch resnet34 \
    --out resnet34_swa_ratio_${i}_bestin90 \
    --track track1 \
    --cfg mixinginference \
    --save_path './results/' \
    --model_path './ckpts/Adaptive_Decouple_SWA_from_best_epoch/ratio_${i}_lr_1e-4/NICO-resnet34-track1/model_best_in_090_epochs.pth.tar' ;
done