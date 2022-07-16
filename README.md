# NICOChallenge-OOD-Classification


## Requirements
A suitable [conda](https://conda.io/) environment named `pytorch` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate pytorch
unzip pydensecrf-master.zip
pip install cython
python3 ./pydensecrf-master/setup.py install
```

## Data Preparation
We prepare the json files which contain the image paths, labels, etc. for the model training.  
(It is noted that the file structures should be the same with those given in phase I.)  
The json files are must saved in **`./dataset_json/`**.  
The data and label_id_mapping are must put in **`./data/`**:  

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
Obtain json files with `bash find_data.sh`

## Train WSSS
We applied resnet50 as baselines to achieve weak-supervised semantic segmenatation.  
The output semantic masks are used in the next training.  
**Note that when "0,1,2,3" gpus are unused, directly carry "Train Moco V2".**

### Track1 
Train models with `bash run_wsss_track1.sh`  
### Track2
Train models with `bash run_wsss_track2.sh`

## Train Moco V2
### Track1 
Train models with `bash run_moco_track1.sh`  
### Track2
Train models with `bash run_moco_track2.sh`  

## Data with Mask Preparation  
After WSSS models are trained, generate data jsons with mask by `bash find_data_with_mask.sh`  

## Train final models  
After WSSS models and MOCO models are trained, carry this step.  
### Track1 
Train models with `bash run_train_track1.sh`
### Track2
Train models with `bash run_train_track2.sh`

## Test 
### Track1 
Generate the final 'prediction.json' by `bash run_ensemble_track1.sh` in './results/track1/'
### Track2
Generate the final 'prediction.json' by `bash run_ensemble_track2.sh` in './results/track2/'
