# NICO Challenge
This is an official code for NICO

## Requirements

*   numpy>=1.19.2
*   Pillow>=8.3.1
*   pytorch>=1.2.0
*   torchvision>=0.4.0
*   tqdm>=4.62.3

## Datasets
* Please add more details, including data path, the number of images. 

## Data Augmentation strategy
* Augmix 

## Regularization Strategy
* Ensemble Learning
* Decouple Mixing (Proposed)

## Domain Generalization Strategy 
* MOCO
* Fourier-DG

## Training Configuration
* Epoch = 180 decay_strategy = cose decay lr_base=1e-1
* Image Resolution=512

## Usage
* Train ResNet-34 on for Track 1 with GPU 1,2 

```
bash run.sh
```

Then the model is saved at `/ckpts/epoch_180_AugMix_DecoupleMix_MOCO_pretrained_FDG_512_lr1e-1_CoseDecay/`, where `model_best_out_090_epochs.pth.tar` refers to the model with best clean accuracy.

* or directly give the configuration by

```
CUDA_VISIBLE_DEVICES=1,2 python3 main_dg.py \
-a  resnet34 \
--epochs 180 \
--start-epoch 0 \
--batch-size 64 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt './ckpts/epoch_180_AugMix_DecoupleMix_MOCO_pretrained_FDG_512_lr1e-1_CoseDecay/' \
--track 1 \
--parallel \
--cos_lr \
--cross \
--root '/apdcephfs/share_1290796/Datasets/NICO/' \
--pretrained '/apdcephfs/share_1290796/jinhengxie/NICO/train/Scaled-down-self-supervised-learning-main/moco/checkpoints/finetune/224/resnet34_track1/gpus_4_lr_0.03_bs_128_epochs_200_path_train_domain_free/checkpoint_0199.pth.tar' \
```
