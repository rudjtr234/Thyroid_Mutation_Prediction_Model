#!/bin/bash
set -euo pipefail

# 저장소 루트에서 실행해도 동작하도록 스크립트 자체 디렉토리로 이동
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES=2

DATASET_ROOT=/path/to/dataset/root
META_DATA_DIR=${DATASET_ROOT}/braf_ensemble_meta_data
HOPTIMUS_META_NPY=${DATASET_ROOT}/h_optimus_embeddings/final_meta_dataset_v0.5.0_40x512
HOPTIMUS_NONMETA_NPY=${DATASET_ROOT}/h_optimus_embeddings/final_non_meta_dataset_v0.5.0_40x512

ENSEMBLE_JSON_HOPTIMUS=${META_DATA_DIR}/ensemble_5models_cv_hoptimus0.json
TEST_JSON_HOPTIMUS=${META_DATA_DIR}/test_set_hoptimus0.json

# 앙상블 각 모델의 MIL 아키텍처: abmil / clam_sb / dsmil / transmil / acmil
MODEL_NAME=dsmil

# ===== Step 1: uni2 JSON -> H-optimus-0 경로 변환 (WSI 구성/fold 분할은 동일하게 유지) =====
echo "========== Step 1: JSON 변환 (uni2 -> h-optimus-0) =========="
python ../../utils/cv_splits/convert_ensemble_json_to_hoptimus.py \
  --ensemble_json "${META_DATA_DIR}/ensemble_5models_cv.json" \
  --test_json "${META_DATA_DIR}/test_set.json" \
  --out_dir "${META_DATA_DIR}"

# ===== Step 2: 사전 검증 (실패 시 set -e로 즉시 중단, 학습 시작 안 함) =====
echo "========== Step 2: 사전 검증 =========="
python ../../utils/cv_splits/verify_ensemble_json_hoptimus.py \
  --ensemble_json "${ENSEMBLE_JSON_HOPTIMUS}" \
  --test_json "${TEST_JSON_HOPTIMUS}" \
  --json_meta_dir "${HOPTIMUS_META_NPY}/json" \
  --json_nonmeta_dir "${HOPTIMUS_NONMETA_NPY}/json"

# ===== Step 3: 앙상블 학습 (H-optimus-0) =====
echo "========== Step 3: 앙상블 학습 =========="
python main_ensemble.py \
  --data_root "${DATASET_ROOT}/h_optimus_embeddings" \
  --model_save_dir outputs/braf_ensemble_hoptimus0_${MODEL_NAME}_v0.4.6 \
  --ensemble_json "${ENSEMBLE_JSON_HOPTIMUS}" \
  --test_json "${TEST_JSON_HOPTIMUS}" \
  --model_name "${MODEL_NAME}" \
  --json_meta_dir "${HOPTIMUS_META_NPY}/json" \
  --json_nonmeta_dir "${HOPTIMUS_NONMETA_NPY}/json" \
  --svs_base_dir /path/to/slide-v1 \
  --embedding_model h-optimus-0 \
  --epochs 100 \
  --lr 1e-4 \
  --bag_size 100 \
  --seed 42 \
  --save_model \
  --generate_plots \
  --tcga_embedding_dir /path/to/TCGA-THCA/embedding/h-optimus-0/40x/npy
