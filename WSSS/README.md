# Track1
```
CUDA_VISIBLE_DEVICES=5,6 python3 run_track1.py --data_dir xxx --json_file xxx --ppath path/to/the/classifier/model
```
e.g.,
```
CUDA_VISIBLE_DEVICES=5,6 python3 run_track1.py --data_dir /apdcephfs/share_1290796/Datasets/NICO/nico_datasets/ --json_file /apdcephfs/share_1290796/Datasets/NICO/dataset_json/track1_train_label.json --ppath /apdcephfs/share_1290796/haozheliu/NICO/ckpts/baseline/NICO-resnet50-track1/model_best_in_090_epochs.pth.tar
```
output mask:
```
experiments/predictions/nico_track1_r50_test_dim_2048@train@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4
```

# Track2
```
CUDA_VISIBLE_DEVICES=5,6 python3 run_track2.py --data_dir xxx --json_file xxx --ppath xxx
```
e.g.,
```
CUDA_VISIBLE_DEVICES=5,6 python3 run_track2.py --data_dir /apdcephfs/share_1290796/Datasets/NICO/nico_datasets/ --json_file /apdcephfs/share_1290796/Datasets/NICO/dataset_json/track2_train_label.json --ppath /apdcephfs/share_1290796/haozheliu/NICO/ckpts/baseline/NICO-resnet50-track2/model_best_in_090_epochs.pth.tar
```
output mask:
``` 
experiments/predictions/nico_track2_r50_test_dim_2048@train@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4
```