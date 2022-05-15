# Robust Representation via Dynamic Feature Aggregation
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
* DeepAug

## Regularization Strategy
* Ensemble Learning
* Decouple Mixing (Proposed)

## Domain Generalization Strategy 
TODO

## Training Configuration
* Epoch = 180 decay_rate=0.1/60epoch
* Image Resolution=448/512

## Usage
* Train ResNet-34 on for Track 2 with GPU 3,4 

```
bash run_SepMixing.sh
```

Then the model is saved at `/ckpts/epoch_180_full_data_SepMixing_DeepAug_448_Ensemble/NICO-resnet34-track2/`, where `model_best_out_090_epochs.pth.tar` refers to the model with best clean accuracy.

* or directly give the configuration by

```
CUDA_VISIBLE_DEVICES=3,4 python3 main.py \
-a  resnet34 \
--epochs 180 \
--start-epoch 0 \
--batch-size 256 \
--lr 1e-1 \
--wd 5e-4 \
--ckpt ./ckpts/epoch_180_full_data_SepMixing_DeepAug_448_Ensemble/ \
--track 2 \
--fine_tune \
--parallel \
--deepaug \
```
