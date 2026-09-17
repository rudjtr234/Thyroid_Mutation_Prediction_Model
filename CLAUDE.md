# Thyroid BRAF Mutation Prediction — Project Context

## 목적
PTC H&E WSI → BRAF V600E 변이 이진 분류 (BRAF+ / BRAF−)

## 파이프라인
```
WSI → PNG 패치 (20x 256×256 / 40x 512×512 / 20x 224×224) → Foundation Model 임베딩 (1536-dim) → MIL 분류
```

## 기술 스택
- **임베딩**: UNI2-H (ViT-H) · H-optimus-0 / H-optimus-1 (ViT-G) — DDP 멀티GPU
- **모델**: ABMIL / CLAM-SB / DSMIL / ACMIL / TransMIL (`src/models/{model}/model.py`, `factory.py`)
- **학습**: 5-Fold CV (`src/training/main.py`) 또는 ABMIL 5-Model Ensemble (`src/training/ensemble/main_ensemble.py`)
- **MLflow**: `MLFLOW_TRACKING_URI` 환경변수로 지정 (기본 `http://localhost:5000`) · experiment: `braf mutation` · 배포모델: `thyr-braf`

## 디렉토리 구조
```
src/
├── data/
│   ├── uni2-h/          # UNI2-H 임베딩 추출 (DDP, torchrun x4)
│   ├── h-optimus-0/     # H-optimus-0 임베딩 추출 (DDP, torchrun x3)
│   ├── h-optimus-1/     # H-optimus-1 임베딩 추출 (패치 / WSI 직접 추출)
│   ├── kaiko/           # Kaiko midnight 임베딩 추출
│   ├── tcga/            # TCGA-THCA 외부검증: SVS→패치(extract_patches.py) + 임베딩
│   └── dataset.py       # 경로명 기반 라벨 자동 추론
├── models/
│   ├── {abmil,clam_sb,dsmil,acmil,transmil}/model.py
│   ├── factory.py       # --model_name 기반 모델 생성
│   ├── mil_template.py
│   └── layers.py
├── training/
│   ├── ensemble/        # ABMIL x5 앙상블 (main_ensemble.py, run_ensemble.sh)
│   ├── main.py          # 단일모델 학습 (--model_name 선택)
│   ├── train_bag.py     # 5-Fold CV 루프
│   ├── register_model.py # best .pt → thyr-braf 등록
│   └── run_bag.sh       # 실험 실행 스크립트
├── evaluation/          # metric.py, visualization.py
├── inference/           # inference_pipeline.py, export_torchscript.py,
│                        # tcga_inference.py, tcga_inference_ensemble.py
└── utils/
    ├── datasets.py      # Bag-level 데이터셋 로더
    └── cv_splits/       # K-Fold 분할 생성 스크립트
```

> 데이터셋·임베딩·체크포인트 경로는 모두 `/path/to/...` 플레이스홀더로 표기되어 있다.
> 실행 전 각 스크립트 상단 또는 `run_*.sh`의 경로 변수를 실제 환경에 맞게 수정해야 한다.
>
> 위 구조의 `src/data/`(임베딩 추출·TCGA 전처리 스크립트)는 내부 데이터 경로에
> 강하게 결합되어 있어 이 공개 저장소에는 포함하지 않는다. 문서상 파이프라인
> 전체를 설명하기 위해 구조에만 표기했다.

## 데이터 경로 (플레이스홀더)
| 데이터 | 경로 |
|--------|------|
| 내부 UNI2-H 임베딩 | `/path/to/dataset/uni2_embeddings/` |
| 내부 H-optimus-0 임베딩 (20x) | `/path/to/dataset/h_optimus_embeddings/` (v0.3~v0.4) |
| 내부 H-optimus-0 임베딩 (40x) | `/path/to/dataset/h_optimus_embeddings/` (v0.5.0_40x512) |
| 내부 H-optimus-1 임베딩 | `/path/to/dataset/h_optimus_1_embeddings/` |
| 앙상블 메타 JSON | `/path/to/dataset/braf_ensemble_meta_data/` |
| TCGA 원본 SVS | `/path/to/TCGA-THCA/raw/{UUID}/` |
| TCGA 패치 | `/path/to/TCGA-THCA/patch/20x/` |
| TCGA 임베딩 | `/path/to/TCGA-THCA/embedding/h-optimus-0/{20x,40x}/` |

## 라벨 추론 (경로명 기반, `dataset.py`)
- BRAF+(`1`): `final_meta_dataset*`, `/meta/`, `braf_meta`
- BRAF−(`0`): `final_non*meta*`, `/nonmeta/`, `braf_non_meta`

## 앙상블 전략
- 모델별 고유 양성 700장 + 공유 음성 700장 + 공유 테스트셋
- 최종 예측 = 5개 softmax 확률 평균 (threshold=0.5)
- 학습 완료 시 MLflow `braf_ensemble_{version}` 자동 등록

## 베스트 결과 (현재 기준)
| 설정 | AUC | Acc | F1 |
|------|-----|-----|----|
| UNI2-H + ABMIL Ensemble v0.1.5 (bag=5000) | **0.9232** | 0.850 | 0.853 |
| H-optimus-0 + TransMIL v0.16.5 (bag=5000, 1000WSI) | 0.8955 | 0.820 | 0.822 |
| H-optimus-0 + ABMIL v0.13.9 (bag=3000, 1000WSI) | 0.8937 | 0.818 | 0.822 |
| H-optimus-0 + CLAM-SB v0.14.11 (bag=5000, 1000WSI) | 0.8850 | 0.810 | 0.809 |

## TCGA 외부 검증 결과 (H-optimus-0 20x, 508 WSI)
| 설정 | AUC | Acc | F1 |
|------|-----|-----|----|
| H-optimus-0 + CLAM-SB v0.14.11 | **0.8078** | 0.762 | 0.760 |
| H-optimus-0 + ABMIL v0.13.9 | 0.7939 | 0.732 | 0.714 |
| H-optimus-0 + TransMIL v0.16.5 | 0.7928 | 0.721 | 0.697 |

## 실행 환경
- **conda 환경**: `thyroid_mutation`
- **임베딩 추출**: 멀티GPU DDP. 이기종 GPU가 섞인 서버에서는 `CUDA_VISIBLE_DEVICES`에
  GPU 번호 대신 **UUID 지정**을 권장 (번호 지정 시 의도치 않은 GPU에 할당될 수 있음)
- **학습**: Single GPU, `CUDA_VISIBLE_DEVICES=<빈 GPU 번호>`
- 실행 전 `nvidia-smi`로 빈 GPU 확인

## 학습 자동화 파이프라인 (v0.18.0 ~)
`bash src/training/run_bag.sh` 한 번으로 아래가 순서대로 자동 실행됨:
1. 5-Fold CV 학습
2. MLflow 업로드 (내부 CV 결과 + heatmap) → `run_id` 반환
3. TCGA 외부검증 추론 (같은 MLflow run에 이어 붙임)
   - BRAF+ / BRAF- 상위 확신도 각 10장 SVS overlay heatmap 생성
   - `tcga_summary.html` 표 + confusion matrix / ROC / prob distribution 업로드

## 패치 추출 현황
| 버전 | 배율 | 크기 | meta | non_meta | 상태 |
|------|------|------|------|----------|------|
| v0.1.0 | 40x | 512×512 | 4,038 | 862 | 완료 |
| v0.2.0 | 20x | 256×256 | 500 | 500 | 완료 |
| v0.3.0 | 20x | 224×224 | 4,038 | 862 | **예정** (40x 448 read → 224 resize) |

## CV Split 파일
`src/utils/cv_splits/`의 생성 스크립트로 만든다. 생성된 JSON은 실제 슬라이드 ID를
포함하므로 이 저장소에는 포함되지 않는다.

| 스크립트 | 용도 |
|------|--------|
| `cv_splits.py` | 기본 K-Fold 분할 생성 (8:1:1) |
| `make_splits_hoptimus1.py` | H-optimus-1 임베딩 기준 분할 생성 |
| `convert_ensemble_json_to_hoptimus.py` | 앙상블 메타 JSON → H-optimus 경로 변환 |
| `verify_ensemble_json_hoptimus.py` | 변환된 분할의 임베딩 파일 존재 검증 |

## 주의사항
- H-optimus 임베딩 추출 시 이기종 GPU 환경에서는 `CUDA_VISIBLE_DEVICES`에 **UUID 사용**
- 패치 버전: `v0.1.0` = 40x 512×512 (원본), `v0.2.0` = 20x 256×256 (500+/500−), `v0.3.0` = 20x 224×224 (예정, 전체)
- 임베딩 버전 네이밍: `final_{meta,non_meta}_dataset_v{버전}_{배율x크기}/`
- 20x 224 패치 추출: WSI level 0 (40x)에서 448×448 read → 224×224 resize (20x 레벨 없음)
- TCGA SVS: `/path/to/TCGA-THCA/raw/{UUID}/` (UUID 하위 폴더 구조)
- 외부검증 추론: `src/inference/tcga_inference.py` (SVS overlay heatmap, 95% CI, MLflow 업로드 포함)
- MLflow run 구조: 학습 결과 + TCGA 외부검증이 **단일 run**으로 통합 업로드 (단일 HTML에 내부 CV + TCGA 섹션 합산)
- 모든 학습/추론 결과에 Bootstrap 95% CI 자동 계산
- Attention heatmap 색상: BRAF+(label=1) = 파랑→빨강 / BRAF-(label=0) = 흰색→짙은파랑 (`visualization.py`, `tcga_inference.py` 동일 적용)
