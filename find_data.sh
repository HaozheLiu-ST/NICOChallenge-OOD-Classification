#!/usr/bin/env bash
python3 data_find.py \
--track 1 \
--root './data/' \
--label_path './data/dg_label_id_mapping.json' \
--json_path './dataset_json/' ;

python3 data_find.py \
--track 2 \
--root './data/' \
--label_path './data/ood_label_id_mapping.json' \
--json_path './dataset_json/' ;