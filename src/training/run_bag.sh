
CUDA_VISIBLE_DEVICES=6 python train_bag.py \
  --data_root /data/143/member/jks/Thyroid_Mutation_dataset/embeddings \
  --model_save_dir /home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model_v2/outputs/Thyroid_prediction_model_v0.5.1 \
  --cv_split_file /home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model_v2/src/utils/cv_splits/cv_splits/cv_splits_balanced_k5_seed42_v0.3.0.json \
  --epochs 100 \
  --lr 1e-5 \
  --bag_size 1000 \
  --seed 42 \
  --save_model \
  --save_best_only \
  --debug
