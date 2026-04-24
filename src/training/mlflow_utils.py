"""
MLflow utilities for experiment tracking and model registration.

This module provides functions to upload training results to MLflow tracking server.
All imports and path configurations are done at module level for efficiency.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow

# SSL 인증서 검증 비활성화 (자체 서명된 인증서를 사용하는 MLflow 서버 허용)
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'


# =========================
# Helper Functions
# =========================


def _get_split_metrics(fold_data: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Return metrics for the requested split (train/val/test) with schema fallback."""
    for key in (f"{split}_metrics", f"best_{split}_metrics"):
        if key in fold_data:
            return fold_data.get(key, {}) or {}
    return {}


def _get_history(fold_data: Dict[str, Any]) -> Dict[str, List[float]]:
    """Return training history with schema fallback."""
    return fold_data.get("training_history") or fold_data.get("history") or {}


def _aggregate_split_metrics(folds: List[Dict[str, Any]], split: str) -> Dict[str, Dict[str, float]]:
    """Aggregate mean/std for a given split across folds."""
    metrics = ["accuracy", "auc", "sensitivity", "specificity", "precision", "npv", "f1"]
    collected: Dict[str, List[float]] = {m: [] for m in metrics}

    for fold in folds:
        metrics_dict = _get_split_metrics(fold, split)
        for metric in metrics:
            val = metrics_dict.get(metric)
            if val is None and metric == "precision":
                val = metrics_dict.get("ppv")
            if val is not None:
                collected[metric].append(val)

    summary: Dict[str, Dict[str, float]] = {}
    for metric, values in collected.items():
        if values:
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return summary


def upload_to_mlflow(
    model_save_dir: str,
    json_path: str,
    model_checkpoint_path: Optional[str],
    lr: float,
    epochs: int,
    bag_size: Optional[int],
    seed: int,
    mode: Optional[str] = None,
    model_name: str = "abmil",
    data_root: Optional[str] = None,
):
    """
    MLflow에 학습 결과를 자동으로 업로드하는 함수

    Args:
        model_save_dir: 모델 저장 디렉토리 경로
        json_path: CV 결과 JSON 파일 경로
        model_checkpoint_path: 모델 체크포인트 파일 경로
        lr: Learning rate
        epochs: Epoch 수
        bag_size: Bag size (bagging 모드가 아닐 경우 None)
        seed: Random seed
        mode: 학습 모드(e.g., full_npy, bag 등)
        model_name: MIL model name
    """
    # JSON 로드
    with open(json_path, 'r') as f:
        summary = json.load(f)

    folds = summary.get("folds", [])
    resolved_mode = mode or summary.get("mode") or ("bag" if bag_size else "full_npy")
    resolved_model_name = summary.get("model_name", model_name)

    # 버전 추출 (경로에서)
    version = Path(model_save_dir).name.replace("Thyroid_prediction_model_", "")

    # MLflow 설정
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("braf mutation")

    run_name = f"braf_full_auto_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Params
        params = {
            "version": version,
            "model_name": resolved_model_name,
            "optimizer": "Adam",
            "lr": lr,
            "epochs": epochs,
            "seed": seed,
            "early_stopping": "val_auc",
            "patience": 8,  # EarlyStopping patience (train_bag.py:600)
        }
        if bag_size is not None:
            params["bag_size"] = bag_size
        if resolved_mode:
            params["mode"] = resolved_mode
        mlflow.log_params(params)

        # Description — 임베딩 모델 / 배율 / 패치 크기를 data_root 경로에서 자동 추론
        if data_root:
            dr = data_root.lower()
            if "h_optimus" in dr or "h-optimus" in dr:
                embed_model = "H-optimus-0"
            elif "uni2" in dr:
                embed_model = "UNI2-H"
            elif "kaiko" in dr:
                embed_model = "Kaiko"
            else:
                embed_model = "Unknown"

            if "40x" in dr or "40x512" in dr:
                patch_info = "40x 512×512"
            elif "20x" in dr or "20x256" in dr:
                patch_info = "20x 256×256"
            else:
                patch_info = "20x 256×256"
        else:
            embed_model = "Unknown"
            patch_info = "20x 256×256"

        mlflow.set_tag(
            "Description",
            f"WSI → Patch({patch_info}) → {embed_model} Embedding(1536-dim) → {resolved_model_name} "
            f"→ BRAF V600E Mutation Classification. 5-fold Stratified CV(8:1:1). "
            f"bag_size={bag_size}, lr={lr}, seed={seed}.",
        )

        # Training Curves
        all_train_loss, all_train_auc, all_train_acc = [], [], []
        all_val_loss, all_val_auc, all_val_acc = [], [], []

        max_epochs = 0
        for fold_data in folds:
            history = _get_history(fold_data)
            train_loss = history.get("train_loss", [])
            train_auc = history.get("train_auc", [])
            train_acc = history.get("train_acc", [])
            val_loss = history.get("val_loss", [])
            val_auc = history.get("val_auc", [])
            val_acc = history.get("val_acc", [])

            max_epochs = max(max_epochs, len(train_loss))
            all_train_loss.append(train_loss)
            all_train_auc.append(train_auc)
            all_train_acc.append(train_acc)
            all_val_loss.append(val_loss)
            all_val_auc.append(val_auc)
            all_val_acc.append(val_acc)

        # Epoch별 평균 로깅
        for epoch in range(max_epochs):
            train_loss_at_epoch = [fold[epoch] for fold in all_train_loss if epoch < len(fold)]
            if train_loss_at_epoch:
                mlflow.log_metric("train_loss", float(np.mean(train_loss_at_epoch)), step=epoch)

            train_auc_at_epoch = [fold[epoch] for fold in all_train_auc if epoch < len(fold)]
            if train_auc_at_epoch:
                mlflow.log_metric("train_auc", float(np.mean(train_auc_at_epoch)), step=epoch)

            train_acc_at_epoch = [fold[epoch] for fold in all_train_acc if epoch < len(fold)]
            if train_acc_at_epoch:
                mlflow.log_metric("train_acc", float(np.mean(train_acc_at_epoch)), step=epoch)

            val_loss_at_epoch = [fold[epoch] for fold in all_val_loss if epoch < len(fold)]
            if val_loss_at_epoch:
                mlflow.log_metric("val_loss", float(np.mean(val_loss_at_epoch)), step=epoch)

            val_auc_at_epoch = [fold[epoch] for fold in all_val_auc if epoch < len(fold)]
            if val_auc_at_epoch:
                mlflow.log_metric("val_auc", float(np.mean(val_auc_at_epoch)), step=epoch)

            val_acc_at_epoch = [fold[epoch] for fold in all_val_acc if epoch < len(fold)]
            if val_acc_at_epoch:
                mlflow.log_metric("val_acc", float(np.mean(val_acc_at_epoch)), step=epoch)

        # Split-wise summary logging
        split_summaries = {
            "train": _aggregate_split_metrics(folds, "train"),
            "val": _aggregate_split_metrics(folds, "val"),
            "test": _aggregate_split_metrics(folds, "test"),
        }
        for split_name, metrics_dict in split_summaries.items():
            for metric, stats in metrics_dict.items():
                mlflow.log_metric(f"{split_name}_{metric}_mean", stats["mean"])
                mlflow.log_metric(f"{split_name}_{metric}_std", stats["std"])

        # HTML 테이블 생성
        def _safe_round(val: Optional[float]) -> float:
            return round(float(val), 4) if val is not None else 0.0

        html_parts = []
        html_parts.append("""
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                h1 { text-align: center; color: #333; margin-bottom: 30px; }
                h2 { text-align: center; color: #555; margin-top: 40px; margin-bottom: 15px; }
                table {
                    border-collapse: collapse;
                    width: 90%;
                    margin: 20px auto;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                th {
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px;
                    text-align: center;
                    font-weight: bold;
                }
                td {
                    padding: 10px;
                    text-align: center;
                    border: 1px solid #ddd;
                }
                tr:nth-child(even) { background-color: #f9f9f9; }
                tr:hover { background-color: #f0f0f0; }
                .summary-table th { background-color: #2196F3; }
                .summary-table tr:nth-last-child(2) { font-weight: bold; background-color: #d1ecf1; }
                .summary-table tr:last-child { font-weight: bold; background-color: #fff3cd; }
                hr { margin: 40px auto; width: 90%; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>5-Fold Cross-Validation Results</h1>
        """)

        for fold_data in folds:
            fold_num = fold_data.get("fold", "?")
            fold_table_data = []

            for split_name, split_key in [("Train", "train"), ("Val", "val"), ("Test", "test")]:
                metrics = _get_split_metrics(fold_data, split_key)
                row = {
                    "Split": split_name,
                    "Accuracy": _safe_round(metrics.get("accuracy")),
                    "AUC": _safe_round(metrics.get("auc")),
                    "Sensitivity": _safe_round(metrics.get("sensitivity")),
                    "Specificity": _safe_round(metrics.get("specificity")),
                    "Precision": _safe_round(metrics.get("precision", metrics.get("ppv")) if metrics else None),
                    "NPV": _safe_round(metrics.get("npv")),
                    "F1": _safe_round(metrics.get("f1")),
                }
                # Train/Val만 Loss 추가 (Test는 loss 계산 안 함)
                if split_key in ["train", "val"]:
                    row["Loss"] = _safe_round(metrics.get("loss"))
                fold_table_data.append(row)

            fold_df = pd.DataFrame(fold_table_data)
            html_parts.append(f"<h2>Fold {fold_num}</h2>")
            html_parts.append(fold_df.to_html(index=False, border=1, justify='center'))

        html_parts.append("<hr><h2>Test Results Summary (All Folds)</h2>")

        summary_data = []
        test_summary = split_summaries.get("test", {})

        for fold_data in folds:
            test_m = _get_split_metrics(fold_data, "test")
            summary_data.append({
                "Fold": f"Fold {fold_data.get('fold', '?')}",
                "Accuracy": _safe_round(test_m.get("accuracy")),
                "AUC": _safe_round(test_m.get("auc")),
                "Sensitivity": _safe_round(test_m.get("sensitivity")),
                "Specificity": _safe_round(test_m.get("specificity")),
                "Precision": _safe_round(test_m.get("precision", test_m.get("ppv")) if test_m else None),
                "NPV": _safe_round(test_m.get("npv")),
                "F1": _safe_round(test_m.get("f1")),
            })

        summary_data.append({
            "Fold": "Mean",
            "Accuracy": _safe_round(test_summary.get("accuracy", {}).get("mean")),
            "AUC": _safe_round(test_summary.get("auc", {}).get("mean")),
            "Sensitivity": _safe_round(test_summary.get("sensitivity", {}).get("mean")),
            "Specificity": _safe_round(test_summary.get("specificity", {}).get("mean")),
            "Precision": _safe_round(
                (test_summary.get("precision") or test_summary.get("ppv") or {}).get("mean")
            ),
            "NPV": _safe_round(test_summary.get("npv", {}).get("mean")),
            "F1": _safe_round(test_summary.get("f1", {}).get("mean")),
        })
        summary_data.append({
            "Fold": "Std",
            "Accuracy": _safe_round(test_summary.get("accuracy", {}).get("std")),
            "AUC": _safe_round(test_summary.get("auc", {}).get("std")),
            "Sensitivity": _safe_round(test_summary.get("sensitivity", {}).get("std")),
            "Specificity": _safe_round(test_summary.get("specificity", {}).get("std")),
            "Precision": _safe_round(
                (test_summary.get("precision") or test_summary.get("ppv") or {}).get("std")
            ),
            "NPV": _safe_round(test_summary.get("npv", {}).get("std")),
            "F1": _safe_round(test_summary.get("f1", {}).get("std")),
        })

        summary_df = pd.DataFrame(summary_data)
        html_parts.append(summary_df.to_html(index=False, border=1, justify='center', classes='summary-table'))
        html_parts.append("</body></html>")

        unified_html = "\n".join(html_parts)
        html_path = Path(model_save_dir) / "cv_results_all_folds.html"
        with open(html_path, "w") as f:
            f.write(unified_html)

        mlflow.log_artifact(str(html_path), artifact_path="tables")

        # Artifacts 업로드
        mlflow.log_artifact(str(json_path), artifact_path="results")

        viz_dir = Path(model_save_dir) / "visualizations"
        if viz_dir.exists():
            mlflow.log_artifact(str(viz_dir), artifact_path="visualizations")

        attn_dir = Path(model_save_dir) / "attention_scores"
        if attn_dir.exists():
            mlflow.log_artifact(str(attn_dir), artifact_path="attention")

        # 모델 체크포인트 업로드 (옵션)
        if model_checkpoint_path:
            try:
                mlflow.log_artifact(str(model_checkpoint_path), artifact_path="model")
                pt_name = Path(model_checkpoint_path).name
                print(f"[✓] Model checkpoint uploaded: {pt_name}")
            except Exception as e:
                print(f"[!] Model upload skipped: {e}")


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Upload CV results to MLflow")
    parser.add_argument("--model_save_dir", type=str, required=True,
                        help="Model save directory (e.g., outputs/Thyroid_prediction_model_v0.5.0)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bag_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    save_dir = Path(args.model_save_dir)

    # JSON 자동 탐색
    json_path = save_dir / "cv_results.json"
    if not json_path.exists():
        json_candidates = list(save_dir.glob("*results*.json"))
        if json_candidates:
            json_path = json_candidates[0]
        else:
            print(f"[!] No results JSON found in {save_dir}")
            sys.exit(1)

    # Best 체크포인트 자동 탐색 + config 추출
    ckpt_dir = save_dir / "checkpoints"
    ckpt_path = None
    ckpt_config = {}
    if ckpt_dir.exists():
        best_ckpts = sorted(ckpt_dir.glob("best_*.pt"))
        if best_ckpts:
            ckpt_path = str(best_ckpts[0])
        else:
            all_ckpts = sorted(ckpt_dir.glob("*.pt"))
            if all_ckpts:
                ckpt_path = str(all_ckpts[0])
        if ckpt_path:
            ckpt_config = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("config", {})

    lr = args.lr or ckpt_config.get("lr", 1e-5)
    epochs = args.epochs or 100
    bag_size = args.bag_size or ckpt_config.get("bag_size")
    seed = args.seed or ckpt_config.get("seed", 42)

    print(f"Model dir  : {save_dir}")
    print(f"JSON       : {json_path}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Config     : lr={lr}, epochs={epochs}, bag_size={bag_size}, seed={seed}")

    upload_to_mlflow(
        model_save_dir=str(save_dir),
        json_path=str(json_path),
        model_checkpoint_path=ckpt_path,
        lr=lr,
        epochs=epochs,
        bag_size=bag_size,
        seed=seed,
    )
