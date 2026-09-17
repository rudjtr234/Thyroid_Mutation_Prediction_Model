#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python main_ensemble.py \
  --data_root /path/to/dataset/uni2_embeddings \
  --model_save_dir outputs/braf_ensemble_v0.1.7 \
  --ensemble_json /path/to/dataset/braf_ensemble_meta_data/ensemble_5models_cv.json \
  --test_json /path/to/dataset/braf_ensemble_meta_data/test_set.json \
  --epochs 100 \
  --lr 1e-4 \
  --bag_size 200 \
  --seed 42 \
  --save_model \
  --generate_plots
