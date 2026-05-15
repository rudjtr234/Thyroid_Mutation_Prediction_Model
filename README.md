# Thyroid BRAF Mutation Prediction from H&E WSI

갑상선 유두암(PTC) H&E 병리 슬라이드(WSI)로부터 **BRAF V600E 변이 여부를 예측**하는 딥러닝 파이프라인.

```
WSI → Patch (20x 256×256) → Foundation Model Embedding (1536-dim) → MIL Classification
```

---

## Results

### Internal Validation (5-Fold CV)

> H-optimus-0 20x 256×256 · 1000 WSI (500+/500−)

| Version | Model | AUC | Acc | F1 | Sensitivity | Specificity | PPV | NPV |
|---------|-------|-----|-----|----|-------------|-------------|-----|-----|
| **v0.16.5** | **TransMIL** | **0.8955** | 0.8200 | 0.8223 | 0.8320 | 0.8080 | 0.8134 | 0.8278 |
| v0.13.9 | ABMIL | 0.8937 | 0.8180 | 0.8219 | 0.8380 | 0.7980 | 0.8072 | 0.8313 |
| v0.14.11 | CLAM-SB | 0.8850 | 0.8100 | 0.8089 | 0.8040 | 0.8160 | 0.8153 | 0.8070 |

> UNI2-H · ABMIL ×5 Ensemble (best): AUC **0.9232**, Acc 0.850, F1 0.853

### External Validation (TCGA-THCA)

> H-optimus-0 20x 256×256 · 508 WSI (BRAF+ 235 / BRAF- 273)

| Version | Model | AUC | Acc | F1 | Sensitivity | Specificity | PPV | NPV |
|---------|-------|-----|-----|----|-------------|-------------|-----|-----|
| v0.13.9 | ABMIL | 0.7939 | 0.7323 | 0.7143 | 0.7234 | 0.7399 | 0.7054 | 0.7566 |
| **v0.14.11** | **CLAM-SB** | **0.8078** | **0.7618** | **0.7604** | **0.8170** | 0.7143 | 0.7111 | **0.8193** |
| v0.16.5 | TransMIL | 0.7928 | 0.7205 | 0.6966 | 0.6936 | 0.7436 | 0.6996 | 0.7382 |

---

## Quick Start

### 1. Feature Extraction

```bash
# H-optimus-0
cd src/data/h-optimus-0 && bash run.sh

# UNI2-H
cd src/data/uni2-h && bash run.sh
```

### 2. Single Model Training (5-Fold CV)

```bash
export CUDA_VISIBLE_DEVICES=0

python src/training/main.py \
    --model_name abmil \                  # abmil | clam_sb | dsmil | acmil | transmil
    --data_root /path/to/embeddings \
    --model_save_dir ./outputs/model_vX.X.X \
    --cv_split_file src/utils/cv_splits/cv_splits.json \
    --epochs 100 --lr 1e-5 --bag_size 3000 --seed 42 \
    --save_model --save_best_only --generate_plots
```

### 3. Ensemble Training

```bash
export CUDA_VISIBLE_DEVICES=0

python src/training/ensemble/main_ensemble.py \
    --data_root /path/to/uni2_embeddings \
    --model_save_dir ./outputs/braf_ensemble_vX.X.X \
    --ensemble_json /path/to/ensemble_5models_cv.json \
    --test_json /path/to/test_set.json \
    --epochs 100 --lr 1e-4 --bag_size 5000 --seed 42 \
    --save_model --generate_plots
```

### 4. External Validation (TCGA)

```bash
# 1. Patch extraction
cd src/data/tcga && bash run_extract_patches.sh

# 2. Embedding extraction
bash run_extract_features_hoptimus.sh

# 3. Inference
cd src/inference
python tcga_inference.py \
    --ckpt_dir /path/to/checkpoints \
    --embedding_dir /path/to/embedding/npy \
    --label_csv /path/to/braf_slide_labels.csv \
    --out_dir /path/to/output \
    --model_version v0.13.9
```

---

## Requirements

```
Python >= 3.8
PyTorch >= 2.2.0
timm == 0.9.16
openslide-python >= 1.3.1
scikit-learn, numpy, pandas, matplotlib, mlflow
```
