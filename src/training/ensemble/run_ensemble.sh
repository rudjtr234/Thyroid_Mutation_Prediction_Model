#!/bin/bash

# Example usage for ensemble training (public template)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python main_ensemble.py \
  --data_root "${DATA_ROOT:-/path/to/embeddings}" \
  --model_save_dir "${MODEL_SAVE_DIR:-/path/to/outputs/braf_ensemble_v0.1.0}" \
  --ensemble_json "${ENSEMBLE_JSON:-/path/to/ensemble_5models_cv.json}" \
  --test_json "${TEST_JSON:-/path/to/test_set.json}" \
  --epochs "${EPOCHS:-100}" \
  --lr "${LR:-1e-4}" \
  --bag_size "${BAG_SIZE:-2000}" \
  --seed "${SEED:-42}" \
  --save_model \
  --generate_plots
