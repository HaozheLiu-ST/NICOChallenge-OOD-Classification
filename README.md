# NICOChallenge-OOD-Classification


## Requirements
A suitable [conda](https://conda.io/) environment named `pytorch1.11` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate pytorch1.11
```

## Data Preparation
We prepare the json files which contain the image paths, labels, etc. for the model training.  
(It is noted that the file structures should be the same with those given in phase I.)  
The json files are saved in `--json_path`, for example `./dataset_json/`  
The data and label_id_mapping are put in `--root` path, for example `./data/`:  

### Track 1
```
$./data/
├── dg_label_id_mapping.json
├── public_dg_0416
│   ├── train
│   │   ├──grass
│   │   │   ├── airplane
│   │   │   │   ├──grass_000001.jpg
│   │   ├── ...
│   ├── public_test_flat
│   │   ├── 00a1befa76cc274de35dda16564d2ecc.jpg
│   │   ├── 00a6d9af238fcbd29748c65b149c9d7b.jpg
│   │   ├── ...
```
### Track 2
```
$./data/
├── ood_label_id_mapping.json
├── public_ood_0412_nodomainlabel
│   ├── train
│   │   ├──airplane
│   │   │   ├── 00b272347bcea2077c2e79449eaf3f1c.jpg
│   │   │   ├── ...
│   │   ├── ...
│   ├── public_test_flat
│   │   ├── 00a1befa76cc274de35dda16564d2ecc.jpg
│   │   ├── 00a6d9af238fcbd29748c65b149c9d7b.jpg
│   │   ├── ...
```
Obtain json files with
```
bash find_data.sh
```

## Training WSSS
We applied resnet50 as baselines to achieve weak-supervised semantic segmenatation.  
The output semantic masks are used in the next training. 

### Track1 
Train models with `bash run_wsss_track1.sh`  

### Track2
Train models with `bash run_wsss_track2.sh`

## Data with Mask Preparation
```
bash find_data_with_mask.sh
```
## Training model with pretrained MOCO V2 
### Track1 
Train models with `bash run_train_track1.sh`
### Track2
Train models with `bash run_train_track2.sh`

## Test 
### Track1 
Train models with `bash run_ensemble_track1.sh`
### Track2
Train models with `bash run_ensemble_track2.sh`


Our Members:
Haozhe Liu*, Wentian Zhang*, Jinheng Xie*, Haoqian Wu, Ziqi Zhang, Yuexiang Li, Yawen Huang, Yefeng Zheng
