# -*- coding: utf-8 -*-
"""
Main entry point for Ensemble Training with MLflow Integration

Workflow:
    1. Parse command-line arguments
    2. Run 5-model ensemble training (train_ensemble.py)
    3. Automatically upload results to MLflow server (mlflow_utils_ensemble.py)

Entry Point:
    Main entry point for BRAF ensemble training with automatic MLflow integration.
"""

import os
import sys

# CRITICAL: GPU 설정을 가장 먼저 해야 함 (torch import 전에)
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    print(f"⚠️  [main_ensemble.py] CUDA_VISIBLE_DEVICES not set, forcing GPU 2")
else:
    print(f"✓ [main_ensemble.py] CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

import argparse
from pathlib import Path

# =========================
# Path Configuration
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(training_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

from train_ensemble import run_ensemble_training
from mlflow_utils_ensemble import upload_ensemble_to_mlflow


def main():
    """
    Main entry point for ensemble training with MLflow integration.
    """

    parser = argparse.ArgumentParser(description='Train 5 Ensemble Models for BRAF Mutation with MLflow')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of embedding data')
    parser.add_argument('--model_save_dir', type=str, required=True,
                        help='Directory to save model checkpoints and results')
    parser.add_argument('--ensemble_json', type=str, required=True,
                        help='Path to ensemble CV split JSON (ensemble_5models_cv.json)')
    parser.add_argument('--test_json', type=str, required=True,
                        help='Path to test set JSON (test_set.json)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--bag_size', type=int, default=2000,
                        help='Bag size for MIL (default: 2000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model checkpoints')
    parser.add_argument('--generate_plots', action='store_true',
                        help='Generate visualization plots')

    args = parser.parse_args()

    # Create output directory
    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)

    # Run training
    print(f"{'='*80}")
    print(f"Starting Ensemble Training Pipeline")
    print(f"{'='*80}\n")

    training_results = run_ensemble_training(args)

    # MLflow upload
    if training_results and training_results.get('json_path'):
        if args.save_model:
            print(f"\n{'='*80}")
            print(f"Uploading to MLflow")
            print(f"{'='*80}\n")

            try:
                upload_ensemble_to_mlflow(
                    model_save_dir=training_results['model_save_dir'],
                    json_path=training_results['json_path'],
                    model_checkpoint_path=training_results.get('model_checkpoint_path'),
                    lr=args.lr,
                    epochs=args.epochs,
                    bag_size=args.bag_size,
                    seed=args.seed
                )
                print(f"\n[✓] MLflow upload completed!")

            except Exception as e:
                print(f"\n[!] MLflow upload failed: {e}")
                import traceback
                traceback.print_exc()

    else:
        print(f"\n[i] Skipping MLflow upload (no results or --save_model not specified)")

    print(f"\n{'='*80}")
    print(f"Pipeline Completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()