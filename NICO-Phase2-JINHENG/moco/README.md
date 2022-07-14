# MoCo pretraining
## Track 1
```
bash pretrain_track1_phase2.sh densenet121 path/to/your/data
```
e.g., 
```
bash pretrain_track1_phase2.sh densenet121 /apdcephfs/share_1290796/Datasets/NICO/nico_datasets
```
```
bash pretrain_track1_phase2.sh wide_resnet50_2 path/to/your/data
bash pretrain_track1_phase2.sh resnet34 path/to/your/data
```
output-path-densenet121: 
```
checkpoints/finetune/224/densenet121_track1_4096_0.03_128/gpus_4_lr_0.03_bs_128_epochs_200_path_train/checkpoint_0199.pth.tar
```
output-path-wide_resnet50_2:
```
checkpoints/finetune/224/wide_resnet_50_track1_4096_0.03_128/gpus_4_lr_0.03_bs_128_epochs_200_path_train/checkpoint_0199.pth.tar
```
output-path-resnet34:
```
checkpoints/finetune/224/resnet34_track1_4096_0.03_128/gpus_4_lr_0.03_bs_128_epochs_200_path_train/checkpoint_0199.pth.tar
```

## Track 2
```
bash pretrain_track2_phase2.sh wide_resnet50_2 path/to/your/data
bash pretrain_track2_phase2.sh densenet121 path/to/your/data
bash pretrain_track2_phase2.sh resnet34 path/to/your/data
```
output-path-densenet121: 
```
checkpoints/finetune/224/densenet121_track2_4096_0.03_128/gpus_4_lr_0.03_bs_128_epochs_200_path_train/checkpoint_0199.pth.tar
```
output-path-wide_resnet50_2:
```
checkpoints/finetune/224/wide_resnet_50_track2_4096_0.03_128/gpus_4_lr_0.03_bs_128_epochs_200_path_train/checkpoint_0199.pth.tar
```
output-path-resnet34:
```
checkpoints/finetune/224/resnet34_track2_4096_0.03_128/gpus_4_lr_0.03_bs_128_epochs_200_path_train/checkpoint_0199.pth.tar
```