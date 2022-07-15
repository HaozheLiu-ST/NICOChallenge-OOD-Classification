#!/usr/bin/env bash

python3 evaluation.py \
--track track2 \
--cfg 'ensemble' \
--out 'prediction' \
--save_path './results/' \
--weights adaptive ;