# -*- coding: utf-8 -*-
"""
Main training script for WSI-level training with MLflow integration.

Workflow:
  1. Parse command-line arguments
  2. Run 5-fold cross-validation training (train_WSI_v2.py)
  3. Optionally upload results to MLflow
"""

import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from train_WSI_v2 import run_training
from mlflow_utils import upload_to_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train WSI-level model with 5-fold CV and MLflow logging"
    )

    # Data arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing preprocessed NPY embeddings")
    parser.add_argument("--cv_split_file", type=str, required=True,
                        help="Path to CV split JSON file")
    parser.add_argument("--json_meta_dir", type=str, required=True,
                        help="Directory containing JSON metadata for meta (BRAF+) cases")
    parser.add_argument("--json_nonmeta_dir", type=str, required=True,
                        help="Directory containing JSON metadata for nonmeta (BRAF-) cases")
    parser.add_argument("--meta_npy_dir", type=str, required=True,
                        help="Directory containing original meta NPY embeddings")
    parser.add_argument("--nonmeta_npy_dir", type=str, required=True,
                        help="Directory containing original nonmeta NPY embeddings")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Saving arguments
    parser.add_argument("--model_save_dir", type=str, required=True,
                        help="Directory to save model checkpoints and results")
    parser.add_argument("--save_model", action="store_true",
                        help="Save model checkpoints")
    parser.add_argument("--save_best_only", action="store_true",
                        help="Save only best model per fold")

    # Visualization arguments
    parser.add_argument("--generate_plots", action="store_true",
                        help="Generate visualization plots + attention heatmaps")

    # Debug arguments
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with verbose output")

    args = parser.parse_args()

    print("=" * 80)
    print("Starting WSI Training Pipeline")
    print("=" * 80 + "\n")

    training_results = run_training(args)

    if training_results and training_results.get("model_checkpoint_path"):
        if args.save_model:
            print("\n" + "=" * 80)
            print("Uploading to MLflow")
            print("=" * 80 + "\n")
            try:
                upload_to_mlflow(
                    model_save_dir=training_results["model_save_dir"],
                    json_path=training_results["json_path"],
                    model_checkpoint_path=training_results["model_checkpoint_path"],
                    lr=args.lr,
                    epochs=args.epochs,
                    bag_size=None,
                    seed=args.seed,
                    mode="full_npy",
                )
                print("\n[OK] MLflow upload completed!")
            except Exception as exc:
                print("\n[!] MLflow upload failed: " + str(exc))
                print("    You can manually upload using:")
                print("    python outputs/mlflow/mlflow_upload.py")
    else:
        print("\n[i] Skipping MLflow upload (no model checkpoint or --save_model not specified)")

    print("\n" + "=" * 80)
    print("Pipeline Completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
