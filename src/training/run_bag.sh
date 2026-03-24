#!/bin/bash
# 단일 모델 5-Fold CV 학습 실행 스크립트
# 실행: cd src/training && bash run_bag.sh

export CUDA_VISIBLE_DEVICES=0

# ===== ABMIL =====
# python main.py \
#   --model_name abmil \
#   --data_root /path/to/embeddings \
#   --model_save_dir /path/to/outputs/model_abmil \
#   --cv_split_file /path/to/cv_splits.json \
#   --epochs 100 --lr 1e-5 --bag_size 3000 --seed 42 \
#   --save_model --save_best_only --generate_plots

# ===== CLAM-SB =====
# python main.py \
#   --model_name clam_sb \
#   --data_root /path/to/embeddings \
#   --model_save_dir /path/to/outputs/model_clam_sb \
#   --cv_split_file /path/to/cv_splits.json \
#   --epochs 100 --lr 1e-5 --bag_size 3000 --seed 42 \
#   --save_model --save_best_only --generate_plots

# ===== DSMIL =====
# python main.py \
#   --model_name dsmil \
#   --data_root /path/to/embeddings \
#   --model_save_dir /path/to/outputs/model_dsmil \
#   --cv_split_file /path/to/cv_splits.json \
#   --epochs 100 --lr 1e-5 --bag_size 3000 --seed 42 \
#   --save_model --save_best_only --generate_plots

# ===== ACMIL =====
# python main.py \
#   --model_name acmil \
#   --data_root /path/to/embeddings \
#   --model_save_dir /path/to/outputs/model_acmil \
#   --cv_split_file /path/to/cv_splits.json \
#   --epochs 100 --lr 1e-5 --bag_size 3000 --seed 42 \
#   --save_model --save_best_only --generate_plots

# ===== TransMIL =====
# python main.py \
#   --model_name transmil \
#   --transmil_max_tokens 2048 \
#   --data_root /path/to/embeddings \
#   --model_save_dir /path/to/outputs/model_transmil \
#   --cv_split_file /path/to/cv_splits.json \
#   --epochs 100 --lr 1e-5 --bag_size 3000 --seed 42 \
#   --save_model --save_best_only --generate_plots
