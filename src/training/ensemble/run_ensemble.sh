#!/bin/bash
# 5-Model Ensemble 학습 실행 스크립트
# 실행: cd src/training/ensemble && bash run_ensemble.sh

export CUDA_VISIBLE_DEVICES=0

python main_ensemble.py \
  --data_root /path/to/embeddings \
  --model_save_dir /path/to/outputs/braf_ensemble_v0.x \
  --ensemble_json /path/to/ensemble_5models_cv.json \
  --test_json /path/to/test_set.json \
  --epochs 100 \
  --lr 1e-4 \
  --bag_size 5000 \
  --seed 42 \
  --save_model \
  --generate_plots
