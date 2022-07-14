# Track1
```
CUDA_VISIBLE_DEVICES=5,6 python3 run_track1.py --data_dir xxx --json_file xxx --ppath xxx
```
e.g.,
```
CUDA_VISIBLE_DEVICES=5,6 python3 run_track1.py --data_dir /apdcephfs/share_1290796/Datasets/NICO/nico_datasets/ --json_file /apdcephfs/share_1290796/Datasets/NICO/dataset_json/track1_train_label.json --ppath /apdcephfs/share_1290796/haozheliu/NICO/ckpts/baseline/NICO-resnet50-track1/model_best_in_090_epochs.pth.tar
```

# Track2
```
CUDA_VISIBLE_DEVICES=5,6 python3 run_track2.py --data_dir xxx --json_file xxx --ppath xxx
```
e.g.,
```
CUDA_VISIBLE_DEVICES=5,6 python3 run_track2.py --data_dir /apdcephfs/share_1290796/Datasets/NICO/nico_datasets/ --json_file /apdcephfs/share_1290796/Datasets/NICO/dataset_json/track2_train_label.json --ppath /apdcephfs/share_1290796/haozheliu/NICO/ckpts/baseline/NICO-resnet50-track1/model_best_in_090_epochs.pth.tar
```
