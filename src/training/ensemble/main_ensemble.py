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
from models.factory import get_available_models


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
    parser.add_argument('--model_name', type=str, default='abmil',
                        choices=get_available_models(),
                        help='MIL model architecture to use for each ensemble member (abmil/clam_sb/dsmil/transmil/acmil)')
    parser.add_argument('--in_dim', type=int, default=1536,
                        help='Input embedding dimension')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of target classes')
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
    parser.add_argument('--json_meta_dir', type=str,
                        default='/path/to/embeddings/meta/json',
                        help='JSON metadata directory for meta cases (heatmap patch coords)')
    parser.add_argument('--json_nonmeta_dir', type=str,
                        default='/path/to/embeddings/nonmeta/json',
                        help='JSON metadata directory for nonmeta cases (heatmap patch coords)')
    parser.add_argument('--svs_base_dir', type=str,
                        default='/path/to/slide-v1',
                        help='SVS base directory for heatmap overlay')
    parser.add_argument('--embedding_model', type=str, default='uni2-h',
                        choices=['uni2-h', 'h-optimus-0'],
                        help='Embedding model used for this training run (for MLflow tagging)')
    parser.add_argument('--tcga_embedding_dir', type=str, default=None,
                        help='TCGA embedding npy 디렉토리 (지정 시 앙상블 학습 후 자동 외부검증)')
    parser.add_argument('--tcga_label_csv', type=str,
                        default='/path/to/TCGA-THCA/genomic/braf_slide_labels.csv',
                        help='TCGA 라벨 CSV 경로')
    parser.add_argument('--tcga_out_dir', type=str, default=None,
                        help='TCGA 외부검증 결과 저장 디렉토리 (기본: model_save_dir/tcga_eval_ensemble)')
    parser.add_argument('--tcga_no_heatmap', action='store_true',
                        help='TCGA attention heatmap 생성 안 함')

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
                run_id = upload_ensemble_to_mlflow(
                    model_save_dir=training_results['model_save_dir'],
                    json_path=training_results['json_path'],
                    model_checkpoint_path=training_results.get('model_checkpoint_path'),
                    lr=args.lr,
                    epochs=args.epochs,
                    bag_size=args.bag_size,
                    seed=args.seed,
                    embedding_model=args.embedding_model,
                    model_name=args.model_name,
                )
                print(f"\n[✓] MLflow upload completed!")

                if run_id:
                    run_id_path = Path(args.model_save_dir) / "mlflow_run_id.txt"
                    run_id_path.write_text(run_id)
                    print(f"[✓] MLflow run_id saved: {run_id_path} ({run_id})")

            except Exception as e:
                run_id = None
                print(f"\n[!] MLflow upload failed: {e}")
                import traceback
                traceback.print_exc()

            # TCGA 외부검증 (앙상블 전용, 단일모델 파이프라인과 완전히 분리된 코드 경로)
            if args.tcga_embedding_dir:
                print(f"\n{'='*80}")
                print(f"TCGA External Validation (Ensemble)")
                print(f"{'='*80}\n")

                try:
                    from inference.tcga_inference_ensemble import run_inference_ensemble

                    ckpt_dir = str(Path(args.model_save_dir) / 'checkpoints')
                    tcga_out = args.tcga_out_dir or str(Path(args.model_save_dir) / 'tcga_eval_ensemble')
                    model_version = Path(args.model_save_dir).name

                    tcga_args = argparse.Namespace(
                        ckpt_dir=ckpt_dir,
                        embedding_dir=args.tcga_embedding_dir,
                        label_csv=args.tcga_label_csv,
                        out_dir=tcga_out,
                        model_version=model_version,
                        embed_model=args.embedding_model,
                        threshold=0.5,
                        gpu=0,
                        min_patches=1000,
                        no_mlflow=False,
                        save_heatmap=not args.tcga_no_heatmap,
                        coord_dir=None,
                        patch_size=512,
                        svs_base_dir='/path/to/TCGA-THCA/raw',
                        mlflow_run_id=None,
                    )

                    run_inference_ensemble(tcga_args, mlflow_run_id=run_id)
                    print(f"\n[✓] TCGA 외부검증 완료! 결과: {tcga_out}")

                except Exception as e:
                    print(f"\n[!] TCGA 외부검증 실패: {e}")
                    import traceback
                    traceback.print_exc()

    else:
        print(f"\n[i] Skipping MLflow upload (no results or --save_model not specified)")

    print(f"\n{'='*80}")
    print(f"Pipeline Completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()