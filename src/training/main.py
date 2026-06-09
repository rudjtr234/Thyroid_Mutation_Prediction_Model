# -*- coding: utf-8 -*-
"""
Main training script with MLflow integration.

This script orchestrates the training pipeline and automatic MLflow logging.

Workflow:
    1. Parse command-line arguments
    2. Run 5-fold cross-validation training (train_bag.py)
    3. Automatically upload results to MLflow server (mlflow_utils.py)

Entry Point:
    This is the main entry point for training with automatic MLflow integration.
"""

import os
import sys

# ⚠️ CRITICAL: GPU 설정을 가장 먼저 해야 함 (torch import 전에)
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    print(f"⚠️  [main.py] CUDA_VISIBLE_DEVICES not set, forcing GPU 2")
else:
    print(f"✓ [main.py] CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

import argparse
from pathlib import Path

# =========================
# Path Configuration
# =========================
# Add src directory to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from train_bag import run_training
from mlflow_utils import upload_to_mlflow
from models.factory import get_available_models


def _str2bool(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected boolean value (true/false).")


def main():
    """
    Main entry point for training with MLflow integration.

    Steps:
    1. Parse arguments
    2. Run training (5-fold CV)
    3. Automatically upload results to MLflow if --save_model is enabled
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train MIL model with 5-fold CV and MLflow logging')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of embedding data')
    parser.add_argument('--model_save_dir', type=str, required=True,
                        help='Directory to save model checkpoints and results')
    parser.add_argument('--cv_split_file', type=str, required=True,
                        help='Path to CV split JSON file')
    parser.add_argument('--model_name', type=str, default='abmil',
                        choices=get_available_models(),
                        help='MIL model name to train')
    parser.add_argument('--in_dim', type=int, default=1536,
                        help='Input embedding dimension')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of target classes')
    parser.add_argument('--model_embed_dim', type=int, default=None,
                        help='Override model embed dimension')
    parser.add_argument('--model_attn_dim', type=int, default=None,
                        help='Override attention hidden dimension (ABMIL/CLAM)')
    parser.add_argument('--model_num_fc_layers', type=int, default=None,
                        help='Override number of FC layers in patch embed MLP')
    parser.add_argument('--model_dropout', type=float, default=None,
                        help='Override dropout')
    parser.add_argument('--model_gate', type=_str2bool, default=None,
                        help='Override gated attention usage (true/false)')
    parser.add_argument('--dsmil_temperature', type=float, default=None,
                        help='Override DSMIL attention temperature')
    parser.add_argument('--transmil_n_heads', type=int, default=None,
                        help='Override TransMIL attention heads')
    parser.add_argument('--transmil_n_layers', type=int, default=None,
                        help='Override TransMIL transformer layers')
    parser.add_argument('--transmil_ffn_dim', type=int, default=None,
                        help='Override TransMIL feed-forward dimension')
    parser.add_argument('--transmil_max_tokens', type=int, default=None,
                        help='Override max tokens for TransMIL self-attention')
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
    parser.add_argument('--save_best_only', action='store_true',
                        help='Save only best model per fold')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--generate_plots', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--test_fold', type=int, default=None,
                        help='Test only specific fold (default: None, test all folds)')
    parser.add_argument('--tcga_embedding_dir', type=str, default=None,
                        help='TCGA embedding npy 디렉토리 (지정 시 학습 후 자동 외부검증)')
    parser.add_argument('--tcga_label_csv', type=str,
                        default=None,
                        help='TCGA 라벨 CSV 경로')
    parser.add_argument('--tcga_out_dir', type=str, default=None,
                        help='TCGA 외부검증 결과 저장 디렉토리 (기본: model_save_dir/tcga_eval)')
    parser.add_argument('--tcga_no_heatmap', action='store_true',
                        help='TCGA attention heatmap 생성 안 함')

    args = parser.parse_args()

    # Run training
    print(f"{'='*80}")
    print(f"Starting Training Pipeline")
    print(f"{'='*80}\n")

    training_results = run_training(args)

    # TCGA 외부검증 (MLflow 업로드 전에 먼저 실행해서 결과를 HTML에 합침)
    tcga_fold_results = None
    if args.tcga_embedding_dir and training_results and training_results.get('model_save_dir'):
        print(f"\n{'='*80}")
        print(f"TCGA External Validation")
        print(f"{'='*80}\n")

        try:
            from inference.tcga_inference import run_inference as tcga_run

            ckpt_dir = str(Path(training_results['model_save_dir']) / 'checkpoints')
            tcga_out = args.tcga_out_dir or str(Path(training_results['model_save_dir']) / 'tcga_eval')
            model_version = Path(training_results['model_save_dir']).name.replace(
                'Thyroid_prediction_model_', '')

            tcga_args = argparse.Namespace(
                ckpt_dir=ckpt_dir,
                embedding_dir=args.tcga_embedding_dir,
                label_csv=args.tcga_label_csv,
                out_dir=tcga_out,
                model_version=model_version,
                embed_model='H-optimus-0',
                threshold=0.5,
                gpu=0,
                min_patches=1000,
                no_mlflow=True,  # MLflow는 아래 upload_to_mlflow에서 통합 처리
                save_heatmap=not args.tcga_no_heatmap,
                coord_dir=str(Path(args.tcga_embedding_dir).parent / 'json'),
                patch_size=512,
                svs_base_dir=None,
            )

            tcga_fold_results = tcga_run(tcga_args, mlflow_run_id=None)
            print(f"\n[✓] TCGA 외부검증 완료! 결과: {tcga_out}")

        except Exception as e:
            print(f"\n[!] TCGA 외부검증 실패: {e}")
            import traceback
            traceback.print_exc()

    # MLflow 업로드 (내부검증 + TCGA 결과 통합)
    if training_results and training_results['model_checkpoint_path']:
        if args.save_model:
            print(f"\n{'='*80}")
            print(f"Uploading to MLflow")
            print(f"{'='*80}\n")

            try:
                mlflow_run_id = upload_to_mlflow(
                    model_save_dir=training_results['model_save_dir'],
                    json_path=training_results['json_path'],
                    model_checkpoint_path=training_results['model_checkpoint_path'],
                    lr=args.lr,
                    epochs=args.epochs,
                    bag_size=args.bag_size,
                    seed=args.seed,
                    model_name=args.model_name,
                    data_root=args.data_root,
                    tcga_fold_results=tcga_fold_results,
                )
                print(f"\n[✓] MLflow upload completed! run_id={mlflow_run_id}")

                # TCGA artifacts(confusion, ROC, heatmap 등)를 같은 run에 이어 붙임
                if tcga_fold_results and args.tcga_embedding_dir:
                    from inference.tcga_inference import upload_all_to_mlflow
                    tcga_out = args.tcga_out_dir or str(Path(training_results['model_save_dir']) / 'tcga_eval')
                    model_version = Path(training_results['model_save_dir']).name.replace(
                        'Thyroid_prediction_model_', '')
                    upload_all_to_mlflow(
                        fold_results=tcga_fold_results,
                        out_dir=tcga_out,
                        model_version=model_version,
                        embed_model='H-optimus-0',
                        patch_config="40x 512x512" if "40x" in args.tcga_embedding_dir else "20x 256x256",
                        run_id=mlflow_run_id,
                    )

            except Exception as e:
                print(f"\n[!] MLflow upload failed: {e}")
                print(f"    You can manually upload using:")
                print(f"    python outputs/mlflow/mlflow_upload.py")
                import traceback
                traceback.print_exc()
                mlflow_run_id = None

    else:
        print(f"\n[i] Skipping MLflow upload (no model checkpoint or --save_model not specified)")
        mlflow_run_id = None

    print(f"\n{'='*80}")
    print(f"Pipeline Completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
