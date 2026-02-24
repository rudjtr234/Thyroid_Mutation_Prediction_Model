# Thyroid BRAF Mutation Prediction (공개용)

WSI 패치 임베딩 기반으로 BRAF mutation(`BRAF-=0`, `BRAF+=1`)을 예측하는 공개용 MIL 파이프라인입니다.
단일 모델(5-fold CV)과 5-model ensemble 학습 코드를 함께 포함합니다.

![BRAF Pipeline](./image/pipeline_overview.png)

## 1. 문제 정의
- 태스크: 이진 분류 (`Non-meta/BRAF-=0`, `Meta/BRAF+=1`)
- 입력: 슬라이드 단위 패치 임베딩 배열 (`.npy`)
- 출력: CV 성능 지표, 체크포인트, 시각화, attention heatmap
- 지원 학습 방식:
  - 단일 모델: `ABMIL` 5-fold CV
  - 앙상블 모델: ABMIL 5개 독립 모델 학습 + 확률 평균

## 2. 저장소 구조
- `src/data/preprocess_data.py`: 패치 이미지 -> 임베딩(`UNI2-h`) 추출
- `src/training/main.py`: 단일 모델 학습 엔트리포인트
- `src/training/train_bag.py`: 단일 모델 5-fold CV 학습/평가
- `src/training/mlflow_utils.py`: 단일 모델 MLflow 로깅(선택)
- `src/training/ensemble/main_ensemble.py`: 앙상블 학습 엔트리포인트
- `src/training/ensemble/train_ensemble.py`: 5-model 앙상블 학습/평가
- `src/training/ensemble/mlflow_utils_ensemble.py`: 앙상블 MLflow 로깅(선택)
- `src/training/ensemble/merge_ensemble.py`: 체크포인트 평균 병합 + 평가
- `src/training/ensemble/run_ensemble.sh`: 공개용 앙상블 실행 템플릿
- `src/evaluation/metric.py`: ROC/PR/혼동행렬 기반 지표 계산
- `src/evaluation/visualization.py`: attention heatmap/overlay 생성
- `src/utils/datasets.py`: bag-level dataset loader
- `configs/abmil_config.yaml`: ABMIL 기본 설정
- `image/pipeline_overview.png`: 전체 파이프라인 다이어그램
- `image/example_attention_heatmap.png`: attention heatmap 예시

## 3. 환경 설정
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

권장 환경:
- Python 3.10+
- CUDA 가능한 PyTorch 환경 (임베딩/학습)

## 4. 데이터 구조 예시
임베딩 루트(`--data_root`)는 아래 형태를 가정합니다.

```text
/path/to/embeddings/
  meta/
    TC_XX_0001.npy
    ...
  nonmeta/
    TC_YY_0001.npy
    ...
```

단일 모델 CV split JSON(`--cv_split_file`) 예시:

```json
{
  "folds": [
    {
      "fold": 1,
      "train_wsis": ["TC_XX_0001.npy"],
      "val_wsis": ["TC_YY_0001.npy"],
      "test_wsis": ["TC_ZZ_0001.npy"],
      "train_count": 0,
      "train_pos_count": 0,
      "train_neg_count": 0,
      "val_count": 0,
      "val_pos_count": 0,
      "val_neg_count": 0,
      "test_count": 0,
      "test_pos_count": 0,
      "test_neg_count": 0
    }
  ]
}
```

앙상블 학습은 아래 2개 JSON을 사용합니다.
- `ensemble_5models_cv.json`: 모델별 train/val split
- `test_set.json`: 공통 test set

## 5. 임베딩 생성
`preprocess_data.py`는 DDP(`torchrun`) 기반으로 동작합니다.

```bash
torchrun --nproc_per_node=4 src/data/preprocess_data.py \
  --tile_dir /path/to/tiles \
  --out_dir /path/to/embeddings \
  --batch_size 512
```

## 6. 단일 모델 학습 (5-fold CV)
```bash
python src/training/main.py \
  --data_root /path/to/embeddings \
  --model_save_dir /path/to/outputs/braf_single_v0.1.0 \
  --cv_split_file /path/to/cv_splits_braf.json \
  --epochs 100 \
  --lr 1e-4 \
  --bag_size 2000 \
  --seed 42 \
  --save_model \
  --save_best_only \
  --generate_plots
```

옵션:
- `--test_fold N`: 특정 fold만 실행
- `--debug`: 상세 로그 출력

## 7. 앙상블 학습 (5-model)
직접 실행:

```bash
python src/training/ensemble/main_ensemble.py \
  --data_root /path/to/embeddings \
  --model_save_dir /path/to/outputs/braf_ensemble_v0.1.0 \
  --ensemble_json /path/to/ensemble_5models_cv.json \
  --test_json /path/to/test_set.json \
  --epochs 100 \
  --lr 1e-4 \
  --bag_size 2000 \
  --seed 42 \
  --save_model \
  --generate_plots
```

또는 템플릿 스크립트:

```bash
cd src/training/ensemble
bash run_ensemble.sh
```

## 8. 주요 산출물
단일 모델(`main.py`) 실행 후 `--model_save_dir`:
- `results_cv_summary_optimal.json`
- `attention_scores/attention_scores_fold*.json`
- `checkpoints/*.pt` (`--save_model` 사용 시)
- `visualizations/` (`--generate_plots` 사용 시)

앙상블(`main_ensemble.py`) 실행 후 `--model_save_dir`:
- `ensemble_results.json`
- `attention_scores/attention_scores_model*.json`
- `checkpoints/model_*.pt` (`--save_model` 사용 시)
- `visualizations/` (`--generate_plots` 사용 시)

예시 heatmap:

![Example Attention Heatmap](./image/example_attention_heatmap.png)

## 9. MLflow 연동 (선택)
공개용 코드는 MLflow 서버 정보를 환경변수로 주입합니다.

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="braf mutation"
export MLFLOW_TRACKING_INSECURE_TLS="false"
```

## 10. 보안 주의사항
이 공개용 리포는 내부 호스트/IP/절대경로를 제외한 형태로 유지해야 합니다.
아래 항목은 커밋하지 마세요:
- 비공개 원본 데이터셋
- 내부 인증정보/토큰
- 내부 추적 서버 주소
- 개인/사내 절대경로
