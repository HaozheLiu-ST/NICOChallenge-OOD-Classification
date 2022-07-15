#!/usr/bin/env bash
python3 data_find.py \
--track 1 \
--root './data/' \
--label_path './data/dg_label_id_mapping.json' \
--seg_path './WSSS/experiments/predictions/nico_track1_r50_test_dim_2048@train@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/' \
--json_path './dataset_json/' ;

python3 data_find.py \
--track 2 \
--root './data/' \
--label_path './data/ood_label_id_mapping.json' \
--seg_path './WSSS/experiments/predictions/nico_track2_r50_test_dim_2048@train@scale=0.5,1.0,1.5,2.0@t=0.55@ccam_inference_crf=4/' \
--json_path './dataset_json/' ;