
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0

# ===== v0.13.x : H-optimus-0 + ABMIL =====
# python main.py \
#   --model_name abmil \
#   --data_root /path/to/dataset/h_optimus_embeddings \
#   --model_save_dir outputs/Thyroid_prediction_model_v0.13.0 \
#   --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_k5_seed42.json \
#   --epochs 100 --lr 1e-5 --bag_size 500 --seed 42 \
#   --save_model --save_best_only --debug --generate_plots

# # ===== v0.13.6~ : H-optimus-0 + ABMIL (1000 WSI, v0.4.0_20x256) =====
# python main.py \
#   --model_name abmil \
#   --data_root /path/to/dataset/h_optimus_embeddings \
#   --model_save_dir outputs/Thyroid_prediction_model_v0.13.11 \
#   --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_1000wsi_k5_seed42.json \
#   --epochs 100 --lr 1e-5 --bag_size 5000 --seed 42 \
#   --save_model --save_best_only --debug --generate_plots

# ===== v0.14.x : H-optimus-0 + CLAM-SB =====
# python main.py \
#   --model_name clam_sb \
#   --data_root /path/to/dataset/h_optimus_embeddings \
#   --model_save_dir outputs/Thyroid_prediction_model_v0.14.5 \
#   --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_k5_seed42.json \
#   --epochs 100 --lr 1e-5 --bag_size 5000 --seed 42 \
#   --save_model --save_best_only --debug --generate_plots

# # ===== v0.14.6~ : H-optimus-0 + CLAM-SB (1000 WSI, v0.4.0_20x256) =====
# python main.py \
#   --model_name clam_sb \
#   --data_root /path/to/dataset/h_optimus_embeddings \
#   --model_save_dir outputs/Thyroid_prediction_model_v0.14.11 \
#   --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_1000wsi_k5_seed42.json \
#   --epochs 100 --lr 1e-5 --bag_size 5000 --seed 42 \
#   --save_model --save_best_only --debug --generate_plots

# ===== v0.15.x : H-optimus-0 + DSMIL (100 WSI) =====
# python main.py \
#   --model_name dsmil \
#   --data_root /path/to/dataset/h_optimus_embeddings \
#   --model_save_dir outputs/Thyroid_prediction_model_v0.15.4 \
#   --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_k5_seed42.json \
#   --epochs 100 --lr 1e-5 --bag_size 4000 --seed 42 \
#   --save_model --save_best_only --debug --generate_plots

# ===== v0.15.5~ : H-optimus-0 + DSMIL (1000 WSI, v0.4.0_20x256) =====
# python main.py \
#   --model_name dsmil \
#   --data_root /path/to/dataset/h_optimus_embeddings \
#   --model_save_dir outputs/Thyroid_prediction_model_v0.15.0 \
#   --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_1000wsi_k5_seed42.json \
#   --epochs 100 --lr 1e-5 --bag_size 500 --seed 42 \
#   --save_model --save_best_only --debug --generate_plots

# ===== v0.16.x : H-optimus-0 + TransMIL (100 WSI) =====
# python main.py \
#   --model_name transmil \
#   --transmil_max_tokens 2048 \
#   --data_root /path/to/dataset/h_optimus_embeddings \
#   --model_save_dir outputs/Thyroid_prediction_model_v0.16.0 \
#   --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_k5_seed42.json \
#   --epochs 100 --lr 1e-5 --bag_size 500 --seed 42 \
#   --save_model --save_best_only --debug --generate_plots

# ===== v0.16.x~ : H-optimus-0 + TransMIL (1000 WSI, v0.4.0_20x256) =====
# python main.py \
#   --model_name transmil \
#   --transmil_max_tokens 2048 \
#   --data_root /path/to/dataset/h_optimus_embeddings \
#   --model_save_dir outputs/Thyroid_prediction_model_v0.16.9 \
#   --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_1000wsi_k5_seed42.json \
#   --epochs 100 --lr 1e-5 --bag_size 400 --seed 42 \
#   --save_model --save_best_only --debug --generate_plots

# ===== v0.17.x : H-optimus-0 + ACMIL (1000 WSI, v0.4.0_20x256) =====
# python main.py \
#   --model_name acmil \
#   --data_root /path/to/dataset/h_optimus_embeddings \
#   --model_save_dir outputs/Thyroid_prediction_model_v0.17.2 \
#   --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_1000wsi_k5_seed42.json \
#   --epochs 100 --lr 1e-5 --bag_size 2000 --seed 42 \
#   --save_model --save_best_only --debug --generate_plots

# ===== v0.18.x : H-optimus-0 + ABMIL (862 WSI, v0.5.0_40x512) =====
# python main.py \
#   --model_name abmil \
#   --data_root /path/to/dataset/h_optimus_embeddings \
#   --model_save_dir outputs/Thyroid_prediction_model_v0.18.10 \
#   --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_862wsi_40x512_k5_seed42.json \
#   --epochs 100 --lr 1e-5 --bag_size 400 --seed 42 \
#   --save_model --save_best_only --debug --generate_plots \
#   --tcga_embedding_dir /path/to/TCGA-THCA/embedding/h-optimus-0/40x/npy

# ===== v0.20.x : H-optimus-1 + ABMIL (862 balanced, 1724 WSI, v0.1.0_40x512) =====
# v0.18.x와 동일 조건(모델/bag_size/lr/epochs)으로 임베딩만 교체 → H-optimus-0 대비 순수 비교.
# TCGA 외부검증은 뺐다 — TCGA 임베딩이 아직 h-optimus-0라 feature space가 달라 무의미하다.
# TCGA를 h-optimus-1로 다시 뽑은 뒤 --tcga_embedding_dir를 추가할 것.
python main.py \
  --model_name abmil \
  --data_root /path/to/dataset/h_optimus_1_embeddings \
  --model_save_dir outputs/Thyroid_prediction_model_v0.20.0 \
  --cv_split_file src/utils/cv_splits/h-optimus-1/cv_splits_braf_hoptimus1_862wsi_40x512_k5_seed42.json \
  --epochs 100 --lr 1e-5 --bag_size 500 --seed 42 \
  --save_model --save_best_only --debug --generate_plots \
  --tcga_embedding_dir /path/to/TCGA-THCA/embedding/h-optimus-1/40x/npy

# # ===== v0.19.x : H-optimus-0 + ACMIL (862 WSI, v0.5.0_40x512) =====
# python main.py \
#   --model_name acmil \
#   --data_root /path/to/dataset/h_optimus_embeddings \
#   --model_save_dir outputs/Thyroid_prediction_model_v0.19.1 \
#   --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_862wsi_40x512_k5_seed42.json \
#   --epochs 100 --lr 1e-5 --bag_size 1000 --seed 42 \
#   --save_model --save_best_only --debug --generate_plots \
#   --tcga_embedding_dir /path/to/TCGA-THCA/embedding/h-optimus-0/40x/npy
