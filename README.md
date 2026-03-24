# Thyroid BRAF Mutation Prediction from H&E WSI

<p align="center">
  <img src="./image/pipeline_overview.png" alt="Pipeline Overview" width="90%"/>
</p>

---

## Overview

갑상선 유두암(PTC) H&E 병리 슬라이드(WSI)로부터 **BRAF V600E 변이 여부를 예측**하는 딥러닝 파이프라인.

BRAF V600E 변이는 PTC의 약 80%에서 발견되는 주요 드라이버 변이로, 예후 및 치료 반응성에 큰 영향을 미친다. 기존 유전자 검사(PCR, NGS)의 비용과 접근성 한계를 보완하기 위해, **H&E 조직 형태만으로 변이를 추정하는 AI 모델**을 개발하였다.

**핵심 접근법:** Pathology Foundation Model 임베딩 (UNI2-H / H-optimus-0 등) → **Multi-MIL Model Zoo** (ABMIL / CLAM-SB / DSMIL / ACMIL / TransMIL) → 단일모델 5-Fold CV 또는 ABMIL 5-Model Ensemble

---

## Pipeline

```text
H&E WSI
  │
  ▼
Patch Extraction ──────────── 20x 256×256 PNG 타일 분할
  │
  ▼
Foundation Model Embedding ── UNI2-H (ViT-H, 1536-dim) · DDP 4-GPU
  │                           H-optimus-0 (ViT-G, 1536-dim) · DDP 3-GPU
  │                           출력: {slide_id}.npy [N, 1536]
  ▼
MIL Training Branch
  ├─ Single Model (5-Fold CV): abmil | clam_sb | dsmil | acmil | transmil
  └─ Ensemble (ABMIL x5): 고유 양성 700 + 공유 음성 700
  ▼
Evaluation ─────────────────── AUC, Acc, Sens, Spec, F1, PPV, NPV
                               Attention/Heatmap 시각화
```

---

## Project Structure

```
thyroid_mutation_public/
├── src/
│   ├── data/                          # 임베딩 추출
│   │   ├── uni2-h/                    # UNI2-H 임베딩 추출 (DDP)
│   │   │   ├── preprocess_data.py
│   │   │   └── run.sh
│   │   └── h-optimus-0/               # H-optimus-0 임베딩 추출 (DDP)
│   │       ├── extract_features.py
│   │       └── run.sh
│   ├── models/                        # 모델 정의
│   │   ├── abmil/                     # ABMIL (Gated Attention MIL)
│   │   ├── clam_sb/                   # CLAM-SB style MIL
│   │   ├── dsmil/                     # DSMIL
│   │   ├── acmil/                     # ACMIL
│   │   ├── transmil/                  # TransMIL
│   │   ├── factory.py                 # 모델 생성/레지스트리
│   │   ├── mil_template.py            # MIL 베이스 클래스
│   │   └── layers.py                  # Attention 레이어, MLP 빌더
│   ├── training/                      # 학습
│   │   ├── ensemble/                  # 앙상블 학습
│   │   │   ├── main_ensemble.py
│   │   │   ├── train_ensemble.py
│   │   │   ├── merge_ensemble.py
│   │   │   └── run_ensemble.sh
│   │   ├── main.py                    # 단일 모델 학습 엔트리포인트
│   │   ├── train_bag.py               # 5-Fold CV 학습 루프
│   │   └── run_bag.sh                 # 실행 스크립트
│   ├── evaluation/                    # 평가 & 시각화
│   │   ├── metric.py
│   │   └── visualization.py
│   └── utils/                         # 유틸리티
│       └── datasets.py
├── configs/
│   └── mil_model_zoo.yaml             # 모델별 preset
├── image/
│   └── pipeline_overview.png
└── requirements.txt
```

---

## Model Architecture

### Common Input

- 입력: `X = {x_i}_{i=1..N}`, `x_i ∈ R^D` (Foundation Model tile embedding)
  - UNI2-H: D = 1536 / H-optimus-0: D = 1536
- 출력: slide-level binary logits (`BRAF+ / BRAF-`)

### Multi-MIL Model Zoo

| Model | 핵심 아이디어 | 구현 파일 |
|-------|-------------|----------|
| `abmil` | Gated attention으로 tile 중요도 학습 후 weighted pooling | `src/models/abmil/model.py` |
| `clam_sb` | Bag branch + max-instance branch 결합 | `src/models/clam_sb/model.py` |
| `dsmil` | Dual-stream (instance + bag) 기반 집계 | `src/models/dsmil/model.py` |
| `acmil` | Multi-branch attention + stochastic top-k masking | `src/models/acmil/model.py` |
| `transmil` | Transformer encoder로 patch 간 전역 상호작용 학습 | `src/models/transmil/model.py` |

---

## Dataset

| 구분 | 설명 | 수량 |
|------|------|------|
| Meta (BRAF+) | BRAF 변이 양성 슬라이드 | ~2,000 WSI |
| Non-Meta (BRAF−) | 변이 음성 슬라이드 | 862 WSI |
| 슬라이드당 패치 수 | 20x 256×256 타일 | 평균 ~22,000개 |
| 임베딩 차원 | Foundation Model feature vector | 1536-dim |

### CV Split JSON 구조

```json
{
  "seed": 42,
  "k_folds": 5,
  "total_wsis": 1000,
  "split_ratio": "6:2:2 (train:val:test)",
  "balanced": true,
  "meta_dir": "/path/to/meta_embeddings/npy",
  "nonmeta_dir": "/path/to/nonmeta_embeddings/npy",
  "folds": [
    {
      "fold": 1,
      "train_wsis": ["slide_001.npy", "..."],
      "val_wsis": ["slide_601.npy", "..."],
      "test_wsis": ["slide_801.npy", "..."],
      "train_count": 600,
      "train_pos_count": 300,
      "train_neg_count": 300,
      "val_count": 200,
      "val_pos_count": 100,
      "val_neg_count": 100,
      "test_count": 200,
      "test_pos_count": 100,
      "test_neg_count": 100
    }
  ]
}
```

---

## Quick Start

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. Feature Extraction

**UNI2-H** 임베딩 추출:
```bash
cd src/data/uni2-h && bash run.sh
```

**H-optimus-0** 임베딩 추출:
```bash
cd src/data/h-optimus-0 && bash run.sh
```

### 3. Single Model Training (5-Fold CV)

```bash
cd src/training && bash run_bag.sh
```

또는 직접 실행:

```bash
python src/training/main.py \
  --model_name abmil \
  --data_root /path/to/embeddings \
  --model_save_dir /path/to/outputs/model_abmil \
  --cv_split_file /path/to/cv_splits.json \
  --epochs 100 --lr 1e-5 --bag_size 3000 --seed 42 \
  --save_model --save_best_only --generate_plots
```

모델 선택: `--model_name {abmil, clam_sb, dsmil, acmil, transmil}`

### 4. Ensemble Training (ABMIL x5)

```bash
cd src/training/ensemble && bash run_ensemble.sh
```

---

## Experimental Results

### UNI2-H + ABMIL 5-Model Ensemble

| Version | Bag Size | Accuracy | AUC | Sensitivity | Specificity | Precision | NPV | F1 |
|---------|----------|----------|-----|-------------|-------------|-----------|-----|-----|
| v0.1.0 | 500 | 0.8250 | 0.9173 | 0.8900 | 0.7600 | 0.7876 | 0.8736 | 0.8357 |
| v0.1.3 | 3000 | 0.8400 | 0.9171 | 0.9000 | 0.7800 | 0.8036 | 0.8864 | 0.8491 |
| **v0.1.5** | **5000** | **0.8500** | **0.9232** | **0.8700** | **0.8300** | **0.8365** | **0.8646** | **0.8529** |

### H-optimus-0 + ABMIL (단일 모델 5-Fold CV, 1000 WSI)

| Version | Bag Size | Accuracy | AUC | Sensitivity | Specificity | Precision | NPV | F1 |
|---------|----------|----------|-----|-------------|-------------|-----------|-----|-----|
| v0.13.6 | 500 | 0.8050 | 0.8916 | 0.8260 | 0.7840 | 0.7956 | 0.8195 | 0.8093 |
| **v0.13.9** | **3000** | **0.8180** | **0.8937** | **0.8380** | **0.7980** | **0.8072** | **0.8313** | **0.8219** |
| v0.13.11 | 5000 | 0.8150 | 0.8920 | 0.8340 | 0.7960 | 0.8056 | 0.8275 | 0.8188 |

### H-optimus-0 + CLAM-SB (단일 모델 5-Fold CV, 1000 WSI)

| Version | Bag Size | Accuracy | AUC | Sensitivity | Specificity | Precision | NPV | F1 |
|---------|----------|----------|-----|-------------|-------------|-----------|-----|-----|
| v0.14.6 | 500 | 0.7910 | 0.8837 | 0.8660 | 0.7160 | 0.7554 | 0.8445 | 0.8057 |
| **v0.14.9** | **3000** | **0.8020** | **0.8834** | **0.8360** | **0.7680** | **0.7834** | **0.8243** | **0.8086** |

---

## Hardware

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GeForce RTX 3080 (10GB) × 4, Tesla V100S-PCIE-32GB × 3 |
| CUDA | 12.2 / Driver 535.104.05 |
| UNI2-H 임베딩 추출 | DDP 4-GPU (RTX 3080), batch_size=512 |
| H-optimus-0 임베딩 추출 | DDP 3-GPU (V100S), batch_size=448 |
| 학습 | Single GPU, batch_size=1 (MIL 표준) |

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.2.0
- timm == 0.9.16
- transformers >= 4.40.0
- scikit-learn, scipy, pandas, numpy
- matplotlib, seaborn
