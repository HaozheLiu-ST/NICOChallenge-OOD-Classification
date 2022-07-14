#!/usr/bin/env bash
python3 data_find.py \
--track 1 \
--data_path './data/' \
--label_path './data/dg_label_id_mapping.json' \
--seg_path 'seg_map/' \
--json_path './datasets/' ;

python3 data_find.py \
--track 2 \
--data_path './data/' \
--label_path './data/ood_label_id_mapping.json' \
--seg_path 'seg_map/' \
--json_path './datasets/' ;