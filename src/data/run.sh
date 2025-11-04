#!/bin/bash
# BRAF meta 데이터 임베딩
export CUDA_VISIBLE_DEVICES=3,4,5
torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    preprocess_data.py \
    --tile_dir /data/143/member/kwk/dl/thyroid/image/slide-v1-240412/patch/Train/preprocess_data/braf_non_meta_v0.2.0 \
    --out_dir /data/143/member/jks/Thyroid_Mutation_dataset/embeddings/preprocess_data/braf_non_meta_v0.2.0 \
    --batch_size 1024
