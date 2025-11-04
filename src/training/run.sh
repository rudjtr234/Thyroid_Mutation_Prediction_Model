
#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python train_WSI.py \
    --data_root /data/143/member/jks/Thyroid_Mutation_dataset/embeddings \
    --cv_split_file /home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model_v2/src/utils/cv_splits/cv_splits_k5_seed42_.json_v0.2.0 \
    --model_save_dir /home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model_v2/outputs/Thyroid_prediction_model_v0.3.4 \
    --epochs 100 \
    --lr 7e-6 \
    --patience 8 \
    --seed 42 \
    --save_model \
    --save_best_only \
    --debug
