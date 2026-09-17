# Thyroid BRAF Mutation Prediction from H&E WSI

<!-- TODO: 개요 파이프라인 Figure 삽입 -->
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
  │                           H-optimus-1 (ViT-G, 1536-dim) · WSI 직접 읽기
  │                           출력: {slide_id}.npy [N, 1536]
  ▼
MIL Training Branch
  ├─ Single Model (5-Fold CV): abmil | clam_sb | dsmil | acmil | transmil
  └─ Ensemble (x5): abmil | transmil | acmil | dsmil
  ▼
Evaluation ─────────────────── 내부 CV + TCGA-THCA 외부검증
  │                           AUC, Acc, Sens, Spec, F1, PPV, NPV (Bootstrap 95% CI)
  │                           Attention/Heatmap 시각화
  ▼
External Validation ───────── fold별 (tcga_inference.py)
                              확률 평균 앙상블 (tcga_inference_ensemble.py)
```

---

## Project Structure

```
Thyroid_Mutation_model_v2/
├── src/
│   ├── data/                          # 임베딩 추출
│   │   ├── uni2-h/                    # UNI2-H 임베딩 추출 (DDP)
│   │   │   ├── preprocess_data.py
│   │   │   └── run.sh
│   │   ├── h-optimus-0/               # H-optimus-0 임베딩 추출 (DDP)
│   │   │   ├── extract_features.py
│   │   │   └── run.sh
│   │   ├── h-optimus-1/               # H-optimus-1 임베딩 추출
│   │   │   ├── extract_features.py    #   패치 PNG 기반 (DDP)
│   │   │   └── extract_features_wsi.py #  WSI 직접 읽기 (NFS I/O 회피)
│   │   └── tcga/                      # TCGA-THCA 파이프라인 (외부 검증)
│   │       ├── extract_patches.py     #   SVS → PNG 패치 추출 (20x 256×256, Otsu)
│   │       ├── run_extract_patches.sh
│   │       ├── run_extract_features_hoptimus.sh
│   │       └── run_extract_features_uni2h.sh
│   ├── models/                        # 모델 정의
│   │   ├── abmil/                     # ABMIL (Gated Attention MIL)
│   │   │   └── model.py
│   │   ├── clam_sb/                   # CLAM-SB style MIL
│   │   │   └── model.py
│   │   ├── dsmil/                     # DSMIL
│   │   │   └── model.py
│   │   ├── acmil/                     # ACMIL
│   │   │   └── model.py
│   │   ├── transmil/                  # TransMIL style MIL
│   │   │   └── model.py
│   │   ├── factory.py                 # 모델 생성/레지스트리
│   │   ├── mil_template.py            # MIL 베이스 클래스
│   │   └── layers.py                  # Attention 레이어, MLP 빌더
│   ├── training/                      # 학습
│   │   ├── ensemble/                  # ★ 앙상블 학습 (메인)
│   │   │   ├── main_ensemble.py       #   엔트리포인트
│   │   │   ├── train_ensemble.py      #   5-Model 앙상블 학습 루프
│   │   │   ├── merge_ensemble.py      #   Weight Averaging 병합
│   │   │   ├── run_ensemble.sh        #   실행 스크립트 (UNI2-H)
│   │   │   └── run_ensemble_hoptimus0.sh # 변환→검증→학습 3단계 파이프라인
│   │   ├── main.py                    # 단일 모델 학습 엔트리포인트
│   │   ├── train_bag.py               # 5-Fold CV 학습 루프
│   ├── evaluation/                    # 평가 & 시각화
│   │   ├── metric.py                  # 메트릭 계산
│   │   └── visualization.py           # ROC/PR/학습곡선 + Attention Heatmap
│   ├── inference/                     # 추론
│   │   ├── inference_pipeline.py      # 추론 파이프라인
│   │   ├── tcga_inference.py          # TCGA 외부검증 (fold별)
│   │   └── tcga_inference_ensemble.py # TCGA 외부검증 (확률 평균 앙상블)
│   └── utils/                         # 유틸리티
│       ├── datasets.py                # Bag-level 데이터셋 로더
│       └── cv_splits/                 # K-Fold 분할 생성/변환/검증 스크립트
│           ├── cv_splits.py           #   기본 K-Fold 분할 생성 (8:1:1)
│           ├── make_splits_hoptimus1.py            # H-optimus-1 분할 생성
│           ├── convert_ensemble_json_to_hoptimus.py # 앙상블 JSON 경로 변환
│           └── verify_ensemble_json_hoptimus.py     # 학습 전 사전 검증
├── configs/
│   └── mil_model_zoo.yaml             # 모델별 preset (abmil/clam_sb/dsmil/acmil/transmil)
├── outputs/                           # 학습 결과 (버전별 체크포인트, 시각화)
└── exports/                           # 내보내기 모델
```

---

## Model Architecture

### Common Input

- 입력: `X = {x_i}_{i=1..N}`, `x_i ∈ R^D` (Foundation Model tile embedding)
  - UNI2-H: D = 1536 / H-optimus-0: D = 1536
- 출력: slide-level binary logits (`BRAF+ / BRAF-`)

### Multi-MIL Model Zoo

| Model | 핵심 아이디어 | 구현 파일 |
|------|--------------|----------|
| `abmil` | Gated attention으로 tile 중요도 학습 후 weighted pooling | `src/models/abmil/model.py` |
| `clam_sb` | Bag branch + max-instance branch 결합 | `src/models/clam_sb/model.py` |
| `dsmil` | Dual-stream (instance + bag) 및 critical instance 기반 집계 | `src/models/dsmil/model.py` |
| `acmil` | Multi-branch attention + stochastic top-k masking | `src/models/acmil/model.py` |
| `transmil` | Transformer encoder로 patch 간 전역 상호작용 학습 | `src/models/transmil/model.py` |

---

## Dataset

| 구분 | 설명 | 수량 |
|------|------|------|
| Meta (BRAF+) | BRAF 변이 양성 슬라이드 | 4,038 WSI |
| Non-Meta (BRAF−) | 변이 음성 슬라이드 | 862 WSI |
| 20x Patch Set (v0.2.0) | 선별 밸런스드 패치셋 | 1,000 WSI (500+/500−) |
| 슬라이드당 패치 수 | 20x 256×256 타일 | 평균 ~22,213개 |
| 임베딩 차원 | Foundation Model feature vector | 1536-dim |

### Embedding Dataset Versions

| 버전 | Foundation Model | WSI | 배율 | Patch 크기 | 상태 |
|------|-----------------|-----|------|-----------|-------------|
| v0.1.1_20x256 | UNI2-H | 4,900 (4,038+/862−) | 20x | 256×256 | 완료 |
| v0.2.0_20x256 | UNI2-H | 1,000 (500+/500−) | 20x | 256×256 | 완료 |
| v0.3.0_20x256 | H-optimus-0 | 200 (100+/100−) | 20x | 256×256 | 완료 |
| v0.4.0_20x256 | H-optimus-0 | 1,000 (500+/500−) | 20x | 256×256 | 완료 |
| v0.5.0_40x512 | H-optimus-0 | 4,900 (4,038+/862−) | 40x | 512×512 | **완료** |
| **v0.6.0_20x224** | **H-optimus-0** | **4,900 (4,038+/862−)** | **20x** | **224×224** | **예정** |
| v0.1.0_40x512 | H-optimus-1 | 1,724 (862+/862−) | 40x | 512×512 | 추출 진행 |

- 20x patch `v0.2.0`는 선별된 1,000 WSI(500+/500−) 기준으로 패치 생성이 완료되어 있음.
- H-optimus-0 `v0.5.0_40x512` 전체 완료 (meta 4,038 + non-meta 862).
- `v0.6.0_20x224`: WSI level 0(40x)에서 448×448 read → 224×224 resize 방식, 패치 추출 스크립트 준비 완료.
- H-optimus-1 `v0.1.0_40x512`: 862 balanced (1,724 WSI) 기준, WSI 직접 읽기 방식으로 추출 중. 학습 결과는 아직 없음.

---

## Quick Start

### 1. Feature Extraction

**UNI2-H** 임베딩 추출:
```bash
cd src/data/uni2-h && bash run.sh
```

**H-optimus-0** 임베딩 추출:
```bash
cd src/data/h-optimus-0 && bash run.sh
```

**H-optimus-1** 임베딩 추출 — 패치 PNG 방식 (DDP):
```bash
cd src/data/h-optimus-1
torchrun --nproc_per_node=4 extract_features.py \
    --tile_dir /path/to/patches --out_dir /path/to/output --batch_size 256
```

**H-optimus-1** 임베딩 추출 — WSI 직접 읽기 (대용량 슬라이드 권장):
```bash
# torchrun 불필요. 단일 프로세스에서 GPU 여러 장을 각각 워커로 사용
python src/data/h-optimus-1/extract_features_wsi.py \
    --wsi_dir  /path/to/slide-v1 \
    --tile_dir /path/to/patches/meta \
    --out_dir  /path/to/embeddings/meta \
    --gpus 4 5 6 7
```
> 패치 PNG 수만 개를 여는 대신 WSI 1개만 열어 NFS I/O 병목을 회피한다.
> 타일 좌표는 기존 패치 디렉토리의 파일명에서만 파싱하므로 기존 임베딩과
> 동일한 타일 집합이 보장된다.

### 2. Ensemble Training (Main)

5-Model Ensemble 학습:

```bash
export CUDA_VISIBLE_DEVICES=2

python src/training/ensemble/main_ensemble.py \
    --data_root /path/to/uni2_embeddings \
    --model_save_dir ./outputs/braf_ensemble_vX.X.X \
    --ensemble_json /path/to/ensemble_5models_cv.json \
    --test_json /path/to/test_set.json \
    --epochs 100 --lr 1e-4 --bag_size 5000 --seed 42 \
    --save_model --generate_plots
```

**H-optimus-0 앙상블** — 변환 → 검증 → 학습 3단계 자동 실행:

```bash
bash src/training/ensemble/run_ensemble_hoptimus0.sh
```
> UNI2-H 앙상블 분할 JSON을 H-optimus-0 경로로 변환한 뒤, 모든 `.npy` 경로와
> 임베딩 차원(1536)·좌표 JSON 정합성을 사전 검증한다. 검증 실패 시 학습을
> 시작하지 않고 non-zero exit으로 종료한다.

**앙상블 TCGA 외부검증** (확률 평균):

```bash
python src/inference/tcga_inference_ensemble.py \
    --ckpt_dir outputs/braf_ensemble_hoptimus0_vX.X.X/checkpoints \
    --embedding_dir /path/to/TCGA-THCA/embedding/h-optimus-0/40x/npy \
    --label_csv /path/to/TCGA-THCA/genomic/braf_slide_labels.csv \
    --out_dir outputs/tcga_eval_ensemble_vX.X.X \
    --save_heatmap
```
> `tcga_ensemble_*`(확률 평균 후 계산)와 `tcga_mean_*`(개별 fold 성능의 산술평균)은
> 계산식이 다른 별개 지표이므로 혼용하지 않는다.

### 3. Single Model Training (5-Fold CV)

`run_bag.sh`를 사용하면 학습 → MLflow 업로드 → TCGA 외부검증까지 자동 실행:

```bash
cd src/training && bash run_bag.sh
```

또는 직접 실행:

```bash
export CUDA_VISIBLE_DEVICES=4

python src/training/main.py \
    --model_name abmil \
    --data_root /path/to/embeddings \
    --model_save_dir ./outputs/Thyroid_prediction_model_vX.X.X \
    --cv_split_file src/utils/cv_splits/h-optimus-0/cv_splits_braf_hoptimus0_862wsi_40x512_k5_seed42.json \
    --epochs 100 --lr 1e-5 --bag_size 500 --seed 42 \
    --save_model --save_best_only --generate_plots \
    --tcga_embedding_dir /path/to/TCGA-THCA/embedding/h-optimus-0/40x/npy
```

`--tcga_embedding_dir` 지정 시 학습 완료 후 TCGA 외부검증 + MLflow 통합 업로드 자동 실행.

모델은 `--model_name {abmil,clam_sb,dsmil,acmil,transmil}` 중 선택 가능.

예시:

```bash
# CLAM-SB
python src/training/main.py --model_name clam_sb ...

# DSMIL
python src/training/main.py --model_name dsmil ...

# ACMIL
python src/training/main.py --model_name acmil --acmil_num_branches 4 --acmil_topk_ratio 0.1 ...

# TransMIL
python src/training/main.py --model_name transmil --transmil_max_tokens 2048 ...
```

### 4. MLflow 수동 등록 (`thyr-braf`)

`src/training/register_model.py`는 아래 두 가지 방식으로 등록할 수 있다.

```bash
# outputs 기준 최고 성능 모델 자동 선택 (AUC 우선, F1 보조) 후 등록
python src/training/register_model.py \
    --outputs_dir ./outputs \
    --model_filter abmil \
    --compat v1 \
    --alias production

# 특정 checkpoint 직접 지정
python src/training/register_model.py \
    --model_path ./outputs/Thyroid_prediction_model_v0.11.0/checkpoints/best_model_fold5_auc0.9375.pt \
    --compat v1 \
    --alias staging
```

- `--compat v1`: 모델 버전 태그를 `framework=torchscript`, `task=mil` 중심으로 기록
- `--compat v2`: 기존 확장 태그(`embedding`, `model_arch`, `test_*`, `cv_*`)까지 기록

---

## Recent Updates

### 2026-09-09 — H-optimus-1 임베딩 추출 / WSI 직접 읽기 방식 도입

- **`src/data/h-optimus-1/` 신규**: H-optimus-1 (ViT-G, 1536-dim) 임베딩 추출
  - `extract_features.py`: DDP 기반 패치 PNG 임베딩 추출
    - DataLoader 공유 전략을 `file_system`으로 변경 — 타일 수가 많은 슬라이드에서
      fd 한도를 초과해 `Too many open files`로 죽던 문제 해결
  - `extract_features_wsi.py`: **WSI(.svs) 직접 읽기 방식** (패치 PNG 미사용)
    - 패치 PNG 수만 개를 여는 대신 WSI 1개만 열고 `read_region`으로 타일 추출
      → NFS lookup/getattr 왕복(전체 시간의 94%)이 사라짐
    - 타일 좌표는 기존 패치 디렉토리의 **파일명에서만** 파싱 (내용 미판독)
      → 조직 검출 기준 재현 불필요, 기존 임베딩과 동일한 타일 집합 보장
    - DDP/all_gather 제거, GPU마다 서로 다른 슬라이드를 통째로 처리 (동기화 없음)
    - CPU 리더 스레드 ↔ GPU 연산을 큐로 겹쳐 I/O 대기를 은닉
    - 실측(36,062 타일 슬라이드): 패치 PNG 6분 25초 → WSI 16스레드 실질 ~1.5분
- **`src/utils/cv_splits/make_splits_hoptimus1.py`**: H-optimus-1 40x512
  (862 balanced, 1,724 WSI) 5-Fold CV split 생성

> H-optimus-1은 현재 임베딩 추출 및 split 생성 단계이며, 학습 결과는 아직 없음.

### 2026-08-10 — 앙상블 TCGA 외부검증 (확률 평균) 추가

- **`src/inference/tcga_inference_ensemble.py` 신규**: 앙상블 5개 체크포인트의
  확률을 평균해 외부검증 성능을 계산
  - 기존 `tcga_inference.py`는 체크포인트를 5-Fold CV처럼 순회해 **fold별 성능만**
    리포트하고 확률 평균 로직이 없었음
  - 체크포인트 5개의 `model_id={1..5}` 무결성을 로드 **전에** 검증
  - 확률은 리스트 위치가 아니라 **`slide_id` 기준으로 정렬 집계** (모델별 순서 상이 가능)
  - `tcga_ensemble_*`(확률 평균 후 계산)와 `tcga_mean_*`(개별 fold 성능의 산술평균)은
    계산식이 다른 별개 지표 — 혼용 금지
  - attention heatmap은 1차 확률 추론으로 20장을 선정한 뒤 2차로만 재추론
    (508장 × 5모델 전체에 attention을 뽑지 않아 메모리 절감)

### 2026-09-09 — H-optimus-0 앙상블 3단계 파이프라인

- **`src/training/ensemble/run_ensemble_hoptimus0.sh` 신규**: 변환 → 검증 → 학습을
  한 번에 실행
  1. `convert_ensemble_json_to_hoptimus.py` — UNI2-H 앙상블 fold 분할 JSON을
     H-optimus-0 경로로 변환. `uni2_embeddings`와 `h_optimus_embeddings`는 meta/non_meta
     명명 규칙과 버전 번호가 다르므로 부분 문자열 치환 대신 **dir 전체 값 단위 매핑**을
     사용하고, JSON 구조에 의존하지 않도록 전체를 재귀 순회
  2. `verify_ensemble_json_hoptimus.py` — 학습 시작 전 사전 검증, 실패 시 **non-zero
     exit으로 학습을 시작하지 않음**
     - 모든 train/val/test `.npy` 경로 존재 여부 전수 확인
     - `.npy` 마지막 차원이 1536인지 mmap으로 확인 (전체 배열을 메모리에 올리지 않음)
     - heatmap 생성 대상 test WSI의 좌표 JSON 존재 및 `patch_coords` 길이와
       대응 `.npy`의 `shape[0]` 일치 확인
  3. `main_ensemble.py` — `--model_name`으로 abmil/transmil/acmil/dsmil 선택 학습
- ABMIL 외 **TransMIL / ACMIL / DSMIL 앙상블 실험** 추가 (v0.2.x / v0.3.x / v0.4.x)

### 2026-05-20 — Attention Heatmap 라벨별 색상 분리 / MLflow HTML 통합 / 파이프라인 순서 재구성

- **`src/evaluation/visualization.py`**: `create_attention_heatmap_colormap(label=None)` 인터페이스 변경
  - BRAF+(label=1): 파랑→청록→초록→노랑→빨강 (기존 유지)
  - BRAF-(label=0): 흰색→연파랑→하늘→짙은파랑 (신규)
  - colorbar 라벨도 라벨별로 분리 (`High = Red (BRAF+)` / `High = Deep Blue (BRAF-)`)
- **`src/inference/tcga_inference.py`**:
  - `plot_attention_heatmap()`에 `label` 파라미터 추가 → `run_inference()`에서 실제 라벨 전달
  - HTML 요약표에서 95% CI 컬럼 제거 (AUC CI, Acc CI → 간결화)
  - `run_inference()` 반환값 누락 버그 수정 (`return fold_results`)
- **`src/training/mlflow_utils.py`**:
  - `upload_to_mlflow(tcga_fold_results=...)` 파라미터 추가
  - 내부 CV HTML에 TCGA 외부검증 결과 섹션을 합쳐 단일 HTML로 업로드 (색상 구분: 초록=내부, 파랑=TCGA)
- **`src/training/main.py`**: TCGA 외부검증을 MLflow 업로드 **전**에 먼저 실행하여 결과를 HTML에 통합
- **`src/training/run_bag.sh`**: v0.18.4 → v0.18.7, bag_size 4000 → 100, v0.19.x ACMIL 40x512 블록 추가(주석)

### 2026-05-11 — v0.18.0 학습 파이프라인 자동화 / TCGA 40x 외부검증 통합

- **40x 512×512 임베딩 완료**: H-optimus-0 v0.5.0 — meta 4,038 WSI / non_meta 862 WSI
- **v0.18.0 실험**: H-optimus-0 + ABMIL, 862:862 balanced, 40x 512×512, bag_size=500, 5-fold CV
  - CV split: `cv_splits_braf_hoptimus0_862wsi_40x512_k5_seed42.json` (1724 WSI, ~7:1:2)
- **학습 자동화**: `bash run_bag.sh` 한 번으로 학습 → MLflow 업로드 → TCGA 외부검증 → 결과 업로드 완료
- **MLflow 단일 run 통합**: 내부 CV 결과 + TCGA 외부검증 결과를 하나의 run에 통합
  - `tcga_summary/tcga_summary.html` — fold별 TCGA 지표 HTML 표
  - `tcga_heatmaps/` — BRAF+/BRAF- 각 10장 SVS overlay heatmap (확신도 상위 선별)
  - `visualizations/` — 내부 best-fold BRAF+/BRAF- 각 10장 overlay heatmap
- **TCGA 40x 외부검증**: `/path/to/TCGA-THCA/embedding/h-optimus-0/40x/npy` 자동 연결
- **SVS overlay heatmap**: TCGA UUID 하위 폴더 구조 자동 탐색 (`/path/to/TCGA-THCA/raw/{UUID}/`)
- **20x 224×224 패치 추출 스크립트 준비**: 40x level0에서 448×448 read → 224×224 resize (v0.3.0 예정)

### 2026-05-08 — ACMIL 실험 추가 / CI 지표 / Attention Heatmap

- `run_bag.sh`: ACMIL v0.17.x 실험 추가 (H-optimus-0 20x, 1000 WSI)
- `src/evaluation/metric.py`: Bootstrap 95% CI 계산 추가 (`bootstrap_ci()`)
- `src/training/mlflow_utils.py`: MLflow에 CI 지표 로깅 + HTML 테이블 95% CI 행 추가
- `src/inference/tcga_inference.py`:
  - Attention heatmap 생성 (`--save_heatmap`, `--coord_dir` 옵션)
  - AUC/Acc 95% CI 계산 및 출력
  - MLflow에 CI 지표 + heatmap artifact 업로드

### 2026-04-24 — TCGA-THCA 외부 검증 추론 완료

- `src/inference/tcga_inference.py` 신규: TCGA 임베딩 → MIL 모델 추론 + MLflow 자동 업로드
  - fold별 개별 결과 출력 (confusion matrix, ROC curve, prob distribution)
  - 패치 수 1,000 미만 슬라이드 자동 필터링 (6개 제외 → 508개 사용)
- H-optimus-0 20x 기반 상위 3개 모델 외부 검증 결과 (TCGA-THCA, 508 WSI):

| 버전 | 모델 | AUC | Acc | F1 | Sensitivity | Specificity | PPV | NPV |
|------|------|-----|-----|----|-------------|-------------|-----|-----|
| v0.13.9 | ABMIL | 0.7939 | 0.7323 | 0.7143 | 0.7234 | 0.7399 | 0.7054 | 0.7566 |
| v0.14.11 | CLAM-SB | **0.8078** | **0.7618** | **0.7604** | **0.8170** | 0.7143 | 0.7111 | **0.8193** |
| v0.16.5 | TransMIL | 0.7928 | 0.7205 | 0.6966 | 0.6936 | **0.7436** | 0.6996 | 0.7382 |

### 2026-04-13 — TCGA-THCA H-optimus-0 20x 임베딩 추출 완료

- TCGA-THCA 514 WSI 임베딩 추출 완료 (H-optimus-0, 20x 256×256)
  - 출력: `/path/to/TCGA-THCA/embedding/h-optimus-0/20x/` (총 26,629,986 패치)
  - RTX 3080 × 3 (GPU 0,1,5), UUID 지정, batch_size=64
- TCGA-THCA 40x 512×512 패치 추출 완료 (514 WSI, 6,781,950 패치, 소요 90시간)
  - 가장자리 패딩 제거 / alpha=0 필터 / 흑백 배경 제거 개선 적용

### 2026-04-01 — H-optimus-0 40x 512 임베딩 추출 (v0.5.0) + TransMIL 실험

- `src/data/h-optimus-0/run.sh`: 원본 40x 512×512 패치 기반 임베딩 추출로 전환 (v0.5.0_40x512)
  - 패치 소스: `meta_braf_patch_v0.1.0` (4,038 WSI) / `non_meta_braf_patch_v0.1.0` (862 WSI)
  - `CUDA_VISIBLE_DEVICES` UUID 방식으로 V100S 3장 강제 지정
- TransMIL 1000 WSI 실험 (v0.16.x): best AUC 0.8955 (v0.16.5, bag_size=5000)
- MLflow Description 동적 생성: `data_root` 경로에서 임베딩 모델/배율 자동 추론

### 2026-03-30 — TCGA-THCA 외부 검증 파이프라인 추가

- `src/data/tcga/` 신규: TCGA-THCA SVS → 패치 추출 + H-optimus-0 / UNI2-H 임베딩 추출 스크립트
  - `extract_patches.py`: CSV 기반 SVS 일괄 처리, 20x 256×256, Otsu 조직 마스킹, 멀티프로세싱
  - 임베딩 출력 경로: `/path/to/TCGA-THCA/embedding/{h-optimus-0,uni2-h}`

### 2026-03-24 — H-optimus-0 실험 결과 및 Multi-MIL Model Zoo

- ABMIL 경로를 `src/models/abmil/` 패키지로 리팩터링
- MIL SOTA 계열 모델 추가: `clam_sb`, `dsmil`, `acmil`, `transmil`
- `src/models/factory.py` 기반 모델 생성/registry 통합
- 단일 학습 CLI 확장: `--model_name` 및 모델별 override 인자 지원
- H-optimus-0 1000 WSI 실험 결과 추가 (ABMIL best AUC 0.8937, CLAM-SB best AUC 0.8834)
- `src/training/register_model.py`에 outputs 자동 선택 + `--compat {v1,v2}` 등록 모드 추가

---

## Experimental Results

### UNI2-H + ABMIL 5-Model Ensemble (Best)

> Ensemble(Test) 기준 최고 성능 — v0.1.5 (bag_size=5000)

| Version | Bag Size | Accuracy | AUC | Sensitivity | Specificity | Precision | NPV | F1 |
|---------|----------|----------|-----|-------------|-------------|-----------|-----|-----|
| **v0.1.5** | **5000** | **0.8500** | **0.9232** | **0.8700** | **0.8300** | **0.8365** | **0.8646** | **0.8529** |

---

### H-optimus-0 + MIL Models — 내부 검증 SOTA (단일 모델 5-Fold CV)

> 데이터: v0.4.0_20x256 (1000 WSI, 500+/500−) · H-optimus-0 20x 256×256 · CV split 6:2:2

| 버전 | 모델 | WSI | Bag Size | AUC | Acc | F1 | Sensitivity | Specificity | PPV | NPV |
|------|------|-----|----------|-----|-----|----|-------------|-------------|-----|-----|
| **v0.16.5** | **TransMIL** | 1000 | 5000 | **0.8955** | 0.8200 | 0.8223 | 0.8320 | 0.8080 | 0.8134 | 0.8278 |
| v0.13.9 | ABMIL | 1000 | 3000 | 0.8937 | 0.8180 | 0.8219 | 0.8380 | 0.7980 | 0.8072 | 0.8313 |
| v0.14.11 | CLAM-SB | 1000 | 5000 | 0.8850 | 0.8100 | 0.8089 | 0.8040 | 0.8160 | 0.8153 | 0.8070 |

---

### TCGA-THCA 외부 검증 (External Validation)

> 데이터: TCGA-THCA 508 WSI (BRAF+ 235 / BRAF- 273)  
> 임베딩: H-optimus-0 20x 256×256 · 내부 학습 데이터와 완전히 독립된 외부 코호트

| 버전 | 모델 | AUC | Acc | F1 | Sensitivity | Specificity | PPV | NPV |
|------|------|-----|-----|----|-------------|-------------|-----|-----|
| v0.13.9 | ABMIL | 0.7939 | 0.7323 | 0.7143 | 0.7234 | 0.7399 | 0.7054 | 0.7566 |
| **v0.14.11** | **CLAM-SB** | **0.8078** | **0.7618** | **0.7604** | **0.8170** | 0.7143 | 0.7111 | **0.8193** |
| v0.16.5 | TransMIL | 0.7928 | 0.7205 | 0.6966 | 0.6936 | 0.7436 | 0.6996 | 0.7382 |

- 외부 검증 기준 **CLAM-SB(v0.14.11)** 가 AUC 0.8078로 최고 일반화 성능
- CLAM-SB Sensitivity 0.817로 내부 검증 수준 유지

---

### H-optimus-0 40x512 + 5-Model Ensemble (확률 평균) — 현재 Best

> 임베딩: H-optimus-0 v0.5.0_40x512 · 앙상블 5개 모델의 softmax 확률 평균 (threshold=0.5)  
> 내부 Test: 공유 테스트셋 200 WSI · 외부: TCGA-THCA 498 WSI (BRAF+ 234 / BRAF− 264)

**내부 검증 (Ensemble Test)**

| 버전 | 모델 | AUC | Acc | F1 | Sensitivity | Specificity | PPV | NPV |
|------|------|-----|-----|----|-------------|-------------|-----|-----|
| **v0.1.5** | **ABMIL** | **0.9286** | **0.8550** | **0.8466** | 0.8000 | **0.9100** | **0.8989** | **0.8198** |
| v0.4.5 | DSMIL | 0.9242 | 0.8050 | 0.8060 | **0.8100** | 0.8000 | 0.8020 | 0.8081 |
| v0.3.8 | ACMIL | 0.9202 | 0.8150 | 0.8042 | 0.7600 | 0.8700 | 0.8539 | 0.7838 |
| v0.2.5 | TransMIL | 0.9174 | 0.8050 | 0.7845 | 0.7100 | 0.9000 | 0.8765 | 0.7563 |

**TCGA-THCA 외부 검증 (Ensemble)**

| 버전 | 모델 | AUC | Acc | F1 | Sensitivity | Specificity | PPV | NPV |
|------|------|-----|-----|----|-------------|-------------|-----|-----|
| **v0.2.5** | **TransMIL** | **0.8492** | 0.7851 | 0.7766 | 0.7949 | **0.7765** | **0.7592** | 0.8103 |
| v0.1.5 | ABMIL | 0.8437 | **0.7932** | **0.7936** | **0.8462** | 0.7462 | 0.7472 | **0.8455** |
| v0.3.8 | ACMIL | 0.8390 | 0.7711 | 0.7625 | 0.7821 | 0.7614 | 0.7439 | 0.7976 |
| v0.4.5 | DSMIL | 0.8351 | 0.7831 | 0.7787 | 0.8120 | 0.7576 | 0.7480 | 0.8197 |

- 외부 검증 AUC **0.8078 → 0.8492** 로 향상 (기존 최고 CLAM-SB 20x 단일모델 대비 **+0.0414**)
- 확률 평균 앙상블이 개별 모델 평균보다 일관되게 우수
  (TransMIL: 개별 평균 AUC 0.8391 → 앙상블 0.8492)
- 내부 최고는 ABMIL(0.9286), 외부 최고는 TransMIL(0.8492) — 내부 성능 순위가
  외부 일반화 순위와 일치하지 않음

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.2.0
- timm == 0.9.16
- transformers >= 4.40.0
- openslide-python >= 1.3.1
- scikit-learn, scipy, pandas, numpy
- matplotlib, seaborn
- MLflow

---

## Hardware

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GeForce RTX 3080 (10GB) × 4, Tesla V100S-PCIE-32GB × 3 |
| CUDA | 12.2 / Driver 535.104.05 |
| UNI2-H 임베딩 추출 | DDP 4-GPU (RTX 3080), batch_size=512 |
| H-optimus-0 임베딩 추출 | DDP 3-GPU (V100S), batch_size=448 |
| 학습 | Single GPU, batch_size=1 (MIL 표준) |
