# Thyroid BRAF Mutation Prediction (공개용)

WSI 패치 임베딩 기반으로 BRAF mutation(`BRAF-=0`, `BRAF+=1`)을 예측하는 공개용 MIL 학습 파이프라인입니다.

![BRAF Pipeline](./image/pipeline_overview.png)

## 1. 문제 정의
- 태스크: 이진 분류 (`Non-meta/BRAF-=0`, `Meta/BRAF+=1`)
- 입력: 슬라이드 단위 패치 임베딩 배열 (`.npy`)
- 출력: 5-fold CV 성능 지표, 체크포인트, 시각화, attention heatmap
- 모델: `ABMIL (Gated Attention MIL)`

## 2. 저장소 구조
- `src/data/preprocess_data.py`: 패치 이미지 -> 임베딩(`UNI2-h`) 추출
- `src/training/main.py`: 학습 엔트리포인트
- `src/training/train_bag.py`: 5-fold CV 학습/평가 핵심 파이프라인
- `src/training/mlflow_utils.py`: 선택적 MLflow 로깅
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

CV split JSON(`--cv_split_file`)은 fold마다 파일명 목록을 포함해야 합니다.

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

## 5. 임베딩 생성
`preprocess_data.py`는 DDP(`torchrun`) 기반으로 동작합니다.

```bash
torchrun --nproc_per_node=4 src/data/preprocess_data.py \
  --tile_dir /path/to/tiles \
  --out_dir /path/to/embeddings \
  --batch_size 512
```

## 6. 모델 학습
기본 실행:

```bash
python src/training/main.py \
  --data_root /path/to/embeddings \
  --model_save_dir /path/to/outputs/braf_v0.1.0 \
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

## 7. 주요 산출물
학습 완료 후 `--model_save_dir`에 생성:
- `results_cv_summary_optimal.json`
- `attention_scores/attention_scores_fold*.json`
- `checkpoints/*.pt` (`--save_model` 사용 시)
- `visualizations/` (`--generate_plots` 사용 시)

예시 heatmap:

![Example Attention Heatmap](./image/example_attention_heatmap.png)

## 8. MLflow 연동 (선택)
공개용 코드에서는 MLflow 서버를 환경변수로 주입합니다.

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="braf mutation"
export MLFLOW_TRACKING_INSECURE_TLS="false"
```

## 9. 보안 주의사항
이 공개용 리포는 내부 호스트/IP/절대경로를 제외한 형태로 유지해야 합니다.
아래 항목은 커밋하지 마세요:
- 비공개 원본 데이터셋
- 내부 인증정보/토큰
- 내부 추적 서버 주소
- 개인/사내 절대경로
