import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--json_file', required=True, type=str, help='xxx/xx/track2_train_label.json')
parser.add_argument('--ppath', required=True, type=str, help='path to pretrained classifier: xxx/xx/xx.pth.tar')

if __name__ == '__main__':
    args = parser.parse_args()
    tag = 'nico_track2_r50_test_dim_2048'
    vis_cam = False
    data_name = 'track2'
    threshold = 0.55
    crf_iteration = 4
    domain = 'train'
    scales = '0.5,1.0,1.5,2.0'

    # training
    os.system(
        f'python3 ./WSSS/train_CCAM_NICO++_track2.py  --data_dir {args.data_dir} --tag {tag} --batch_size 32 --pretrained nico_t2 --ppath {args.ppath} --json_file {args.json_file} --alpha 0.25')
    print()

    # inference on train set
    # generate ccam
    os.system(f'python3 ./WSSS/inference_CCAM_track2.py --tag {tag} --data_dir {args.data_dir} --json_file {args.json_file} --domain {domain} --pretrained no')
    print()
    # crf
    experiment_name = tag + f'@{domain}@scale={scales}'
    os.system(f'python3 ./WSSS/inference_crf_track2.py --experiment_name {experiment_name} --data_dir {args.data_dir} --domain {domain} --threshold {threshold} --crf_iteration {crf_iteration} --json_file {args.json_file}')
    print()

