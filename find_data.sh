#!/usr/bin/env bash
python3 data_find.py \
--track 1 \
--data_path './data/Track1/' \
--label_path './data/Track1/dg_label_id_mapping.json' \
--json_path './datasets/' ;

python3 data_find.py \
--track 2 \
--data_path './data/Track2/' \
--label_path './data/Track2/ood_label_id_mapping.json' \
--json_path './datasets/' ;