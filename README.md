# NICOChallenge-OOD-Classification
Official Pytorch Implementation for NICO Challenge

## New Version
* Checkpoint is saved per two epochs 
* Add a new file: model_soup.py

## How to use Model Soups 
* Fine tune a model with different parameters
```
export pretrained_path='/apdcephfs/share_1290796/waltszhang/NICO_challenge/train_dg_ddp/ckpts/epoch_200_60epoch_FDG_512_1e-1_coslr_ratio_0.2/NICO-densenet121-track2/teacher/checkpoint_120.pth.tar'
export save_path='/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/from_120/'
export root_path='/apdcephfs/share_1290796/Datasets/NICO/nico_datasets/'
export learning_rate=1e-2
export ratio=0.5
export epochs=40
for i in $(seq 0.1 0.1 0.9); 
do 
    echo "Saved Path: ${save_path}ratio_${i}_lr_${learning_rate}";
    echo "Learning Rate: ${learning_rate}";
    echo "Ratio: ${i}";
    echo "Pretrained_Path: ${pretrained_path}"
    CUDA_VISIBLE_DEVICES=1,2,3,4 python3 /apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/main_dg_ddp.py \
        -a  densenet121 \
        --epochs ${epochs} \
        --start-epoch 0 \
        --batch-size 128 \
        --lr ${learning_rate} \
        --wd 5e-4 \
        --ratio ${i} \
        --ckpt "${save_path}ratio_${i}_lr_${learning_rate}/" \
        --track 2 \
        --root ${root_path} \
        --rank 0 \
        --num_workers 32 \
        --dist-url 'tcp://127.0.0.1:4455' \
        --dist-backend 'nccl' \
        --world-size 1 \
        --pretrained ${pretrained_path};
done

learning_rate_list=(1e-1 1e-2 1e-3 1e-4)

for element in ${learning_rate_list[@]}
do 
    echo "Saved Path: ${save_path}ratio_${ratio}_lr_${element}";
    echo "Learning Rate: ${element}";
    echo "Ratio: ${ratio}";
    echo "Pretrained_Path: ${pretrained_path}"
    CUDA_VISIBLE_DEVICES=1,2,3,4 python3 /apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/main_dg_ddp.py \
        -a  densenet121 \
        --epochs ${epochs} \
        --start-epoch 0 \
        --batch-size 128 \
        --lr ${element} \
        --wd 5e-4 \
        --ratio ${ratio} \
        --ckpt "${save_path}ratio_${ratio}_lr_${element}/" \
        --track 2 \
        --root ${root_path} \
        --rank 0 \
        --num_workers 32 \
        --dist-url 'tcp://127.0.0.1:4455' \
        --dist-backend 'nccl' \
        --world-size 1 \
        --pretrained ${pretrained_path};
done
```

Then generate an averaging model via greedy search

```CUDA_VISIBLE_DEVICES=1 python model_soup.py \
-a densenet121 \
--ckpts \
/apdcephfs/share_1290796/waltszhang/NICO_challenge/train_dg_ddp/ckpts/epoch_200_60epoch_FDG_512_1e-1_coslr_ratio_0.2/NICO-densenet121-track2/teacher/model_best_out_090_epochs.pth.tar \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_05_lr_1e5NICO-densenet121-track2/student/checkpoint_018.pth.tar  \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_02_lr_1e4NICO-densenet121-track2/student/checkpoint_018.pth.tar  \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_05_lr_1e3NICO-densenet121-track2/student/checkpoint_018.pth.tar  \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_06_lr_1e3NICO-densenet121-track2/student/checkpoint_018.pth.tar  \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_04_lr_1e3NICO-densenet121-track2/student/checkpoint_018.pth.tar  \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_05_lr_1e4NICO-densenet121-track2/student/checkpoint_018.pth.tar  \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_08_lr_1e3NICO-densenet121-track2/student/checkpoint_018.pth.tar  \
/apdcephfs/share_1290796/haozheliu/NICO/Model_Soup/ckpts/ratio_01_lr_1e4NICO-densenet121-track2/student/checkpoint_018.pth.tar  \
--root '/apdcephfs/share_1290796/Datasets/NICO/nico_datasets/' \
--track 2 \
--momentum 0.9 \
--save_path ./ckpts/model_soup_greedy_submodel/
```

And the final model is saved at `./ckpts/model_soup_greedy_submodel/`
