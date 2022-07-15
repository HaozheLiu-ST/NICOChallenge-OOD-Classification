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
(It is noted that the file structures should be the same with those given in phase I.) For example:

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
The json files are saved in the `./dataset_json/`

## Training WSSS
We applied resnet34, densenet121 and wide_resnet50_2 as baselines. 

### Track1 
Train a resnet34 with `bash resnet34_wsss_track1.sh `

Train a densenet121 with `bash densenet121_wsss_track1.sh `

Train a wide_resnet50_2 with ` bash wide_resnet50_2_wsss_track1.sh `

### Track2
Train a resnet34 with `bash resnet34_wsss_track2.sh `

Train a densenet121 with `bash densenet121_wsss_track2.sh `

Train a wide_resnet50_2 with `bash wide_resnet50_2_wsss_track2.sh `

## Training MOCO V2

### Track1 
Train a resnet34 with `bash resnet34_moco_track1.sh `

Train a densenet121 with `bash densenet121_moco_track1.sh `

Train a wide_resnet50_2 with ` bash wide_resnet50_2_moco_track1.sh `

### Track2
Train a resnet34 with `bash resnet34_moco_track2.sh `

Train a densenet121 with `bash densenet121_moco_track2.sh `

Train a wide_resnet50_2 with ` bash wide_resnet50_2_moco_track2.sh `

## Training Main

## Test 

## Ensemble


Our Members:
Haozhe Liu*, Wentian Zhang*, Jinheng Xie*, Haoqian Wu, Ziqi Zhang, Yuexiang Li, Yawen Huang, Yefeng Zheng
