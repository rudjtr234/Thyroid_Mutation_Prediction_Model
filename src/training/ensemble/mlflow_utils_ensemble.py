# -*- coding: utf-8 -*-
"""
MLflow utilities for Ensemble experiment tracking.

단일 MLflow run에 5개 모델 결과 + 앙상블 평균을 업로드
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import mlflow

# SSL 인증서 검증 비활성화
os.environ.setdefault('MLFLOW_TRACKING_INSECURE_TLS', 'false')


def _get_model_metrics(model_data: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Return metrics for the requested split (train/val/test)."""
    for key in (f"{split}_metrics", f"best_{split}_metrics"):
        if key in model_data:
            return model_data.get(key, {}) or {}
    return {}


def _get_history(model_data: Dict[str, Any]) -> Dict[str, List[float]]:
    """Return training history."""
    return model_data.get("training_history") or model_data.get("history") or {}


def _aggregate_model_metrics(models: List[Dict[str, Any]], split: str) -> Dict[str, Dict[str, float]]:
    """Aggregate mean/std for a given split across models."""
    metrics = ["accuracy", "auc", "sensitivity", "specificity", "precision", "npv", "f1"]
    collected: Dict[str, List[float]] = {m: [] for m in metrics}

    for model in models:
        metrics_dict = _get_model_metrics(model, split)
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


def upload_ensemble_to_mlflow(
    model_save_dir: str,
    json_path: str,
    model_checkpoint_path: Optional[str],
    lr: float,
    epochs: int,
    bag_size: Optional[int],
    seed: int,
):
    """
    MLflow에 앙상블 학습 결과를 업로드

    Args:
        model_save_dir: 모델 저장 디렉토리 경로
        json_path: ensemble_results.json 파일 경로
        model_checkpoint_path: 모델 체크포인트 파일 경로
        lr: Learning rate
        epochs: Epoch 수
        bag_size: Bag size
        seed: Random seed
    """
    # JSON 로드
    with open(json_path, 'r') as f:
        results = json.load(f)

    models = results.get("models", [])
    ensemble_metrics = results.get("ensemble_metrics", {})
    summary_stats = results.get("summary_statistics", {})

    # 버전 추출
    version = Path(model_save_dir).name

    # MLflow 설정
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'braf mutation')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = f"braf_ensemble_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Params
        params = {
            "version": version,
            "model_type": "ensemble",
            "n_models": len(models),
            "optimizer": "Adam",
            "lr": lr,
            "epochs": epochs,
            "seed": seed,
            "early_stopping": "val_auc",
            "patience": 8,
        }
        if bag_size is not None:
            params["bag_size"] = bag_size
        mlflow.log_params(params)

        # Description
        mlflow.set_tag(
            "Description",
            "BRAF Mutation Ensemble (5 models). "
            "Each model: 700 unique positive + 700 shared negative. "
            "Final prediction: probability averaging. "
            "Model: ABMIL + UNI2-H Embedding(1536-dim)."
        )

        # Training Curves (모델별 평균)
        all_train_loss, all_train_auc, all_val_loss, all_val_auc = [], [], [], []

        max_epochs = 0
        for model_data in models:
            history = _get_history(model_data)
            train_loss = history.get("train_loss", [])
            train_auc = history.get("train_auc", [])
            val_loss = history.get("val_loss", [])
            val_auc = history.get("val_auc", [])

            max_epochs = max(max_epochs, len(train_loss))
            all_train_loss.append(train_loss)
            all_train_auc.append(train_auc)
            all_val_loss.append(val_loss)
            all_val_auc.append(val_auc)

        # Epoch별 평균 로깅
        for epoch in range(max_epochs):
            train_loss_at_epoch = [m[epoch] for m in all_train_loss if epoch < len(m)]
            if train_loss_at_epoch:
                mlflow.log_metric("train_loss", float(np.mean(train_loss_at_epoch)), step=epoch)

            train_auc_at_epoch = [m[epoch] for m in all_train_auc if epoch < len(m)]
            if train_auc_at_epoch:
                mlflow.log_metric("train_auc", float(np.mean(train_auc_at_epoch)), step=epoch)

            val_loss_at_epoch = [m[epoch] for m in all_val_loss if epoch < len(m)]
            if val_loss_at_epoch:
                mlflow.log_metric("val_loss", float(np.mean(val_loss_at_epoch)), step=epoch)

            val_auc_at_epoch = [m[epoch] for m in all_val_auc if epoch < len(m)]
            if val_auc_at_epoch:
                mlflow.log_metric("val_auc", float(np.mean(val_auc_at_epoch)), step=epoch)

        # Split-wise summary logging (개별 모델 평균)
        split_summaries = {
            "train": _aggregate_model_metrics(models, "train"),
            "val": _aggregate_model_metrics(models, "val"),
            "test": _aggregate_model_metrics(models, "test"),
        }
        for split_name, metrics_dict in split_summaries.items():
            for metric, stats in metrics_dict.items():
                mlflow.log_metric(f"{split_name}_{metric}_mean", stats["mean"])
                mlflow.log_metric(f"{split_name}_{metric}_std", stats["std"])

        # Ensemble metrics
        for metric, value in ensemble_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"ensemble_{metric}", float(value))

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
                .summary-table tr:nth-last-child(3) { font-weight: bold; background-color: #d1ecf1; }
                .summary-table tr:nth-last-child(2) { font-weight: bold; background-color: #fff3cd; }
                .summary-table tr:last-child { font-weight: bold; background-color: #d4edda; }
                .ensemble-row { background-color: #d4edda !important; }
                hr { margin: 40px auto; width: 90%; border: 1px solid #ddd; }
            </style>
        </head>
        <body>
            <h1>BRAF Ensemble Results (5 Models)</h1>
        """)

        # 각 모델별 테이블
        for model_data in models:
            model_id = model_data.get("model_id", "?")
            model_table_data = []

            for split_name, split_key in [("Train", "train"), ("Val", "val"), ("Test", "test")]:
                metrics = _get_model_metrics(model_data, split_key)
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
                if split_key in ["train", "val"]:
                    row["Loss"] = _safe_round(metrics.get("loss"))
                model_table_data.append(row)

            model_df = pd.DataFrame(model_table_data)
            html_parts.append(f"<h2>Model {model_id}</h2>")
            html_parts.append(model_df.to_html(index=False, border=1, justify='center'))

        # Summary Table
        html_parts.append("<hr><h2>Test Results Summary (All Models + Ensemble)</h2>")

        summary_data = []
        test_summary = split_summaries.get("test", {})

        for model_data in models:
            test_m = _get_model_metrics(model_data, "test")
            summary_data.append({
                "Model": f"Model {model_data.get('model_id', '?')}",
                "Accuracy": _safe_round(test_m.get("accuracy")),
                "AUC": _safe_round(test_m.get("auc")),
                "Sensitivity": _safe_round(test_m.get("sensitivity")),
                "Specificity": _safe_round(test_m.get("specificity")),
                "Precision": _safe_round(test_m.get("precision", test_m.get("ppv")) if test_m else None),
                "NPV": _safe_round(test_m.get("npv")),
                "F1": _safe_round(test_m.get("f1")),
            })

        # Mean
        summary_data.append({
            "Model": "Mean",
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

        # Std
        summary_data.append({
            "Model": "Std",
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

        # Ensemble
        summary_data.append({
            "Model": "Ensemble",
            "Accuracy": _safe_round(ensemble_metrics.get("accuracy")),
            "AUC": _safe_round(ensemble_metrics.get("auc")),
            "Sensitivity": _safe_round(ensemble_metrics.get("sensitivity")),
            "Specificity": _safe_round(ensemble_metrics.get("specificity")),
            "Precision": _safe_round(ensemble_metrics.get("precision", ensemble_metrics.get("ppv"))),
            "NPV": _safe_round(ensemble_metrics.get("npv")),
            "F1": _safe_round(ensemble_metrics.get("f1")),
        })

        summary_df = pd.DataFrame(summary_data)
        html_parts.append(summary_df.to_html(index=False, border=1, justify='center', classes='summary-table'))
        html_parts.append("</body></html>")

        unified_html = "\n".join(html_parts)
        html_path = Path(model_save_dir) / "ensemble_results_all_models.html"
        with open(html_path, "w") as f:
            f.write(unified_html)

        mlflow.log_artifact(str(html_path), artifact_path="tables")

        # Artifacts 업로드
        mlflow.log_artifact(str(json_path), artifact_path="results")

        viz_dir = Path(model_save_dir) / "visualizations"
        if viz_dir.exists():
            mlflow.log_artifact(str(viz_dir), artifact_path="visualizations")

        ckpt_dir = Path(model_save_dir) / "checkpoints"
        if ckpt_dir.exists():
            mlflow.log_artifact(str(ckpt_dir), artifact_path="checkpoints")
            pt_files = list(ckpt_dir.glob('*.pt'))
            print(f"[✓] Checkpoints uploaded ({len(pt_files)} .pt files)")

            # Registered models에 등록 (REST API 직접 호출로 서버 호환성 확보)
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            run_id = mlflow.active_run().info.run_id
            model_name = f"braf_ensemble_{version}"

            # 모델 설명 생성
            ens_auc = ensemble_metrics.get("auc", "N/A")
            ens_acc = ensemble_metrics.get("accuracy", "N/A")
            ens_f1 = ensemble_metrics.get("f1", "N/A")
            test_auc_mean = split_summaries.get("test", {}).get("auc", {}).get("mean", "N/A")

            model_description = (
                f"BRAF Mutation Ensemble ({len(models)} models)\n"
                f"Backbone: ABMIL + UNI2-H (1536-dim)\n"
                f"Ensemble AUC: {ens_auc}, Acc: {ens_acc}, F1: {ens_f1}\n"
                f"Mean(Test) AUC: {test_auc_mean}\n"
                f"bag_size={bag_size}, lr={lr}, seed={seed}"
            )

            # 모델 이름 생성 (없으면 새로 만들기)
            try:
                client.get_registered_model(model_name)
                client.update_registered_model(model_name, description=model_description)
            except Exception:
                client.create_registered_model(model_name, description=model_description)

            # 모델별 개별 메트릭 매핑 (model_id → test metrics)
            model_metrics_map = {}
            for model_data in models:
                mid = model_data.get("model_id")
                if mid is not None:
                    test_m = _get_model_metrics(model_data, "test")
                    model_metrics_map[int(mid)] = test_m

            for pt_file in sorted(pt_files):
                try:
                    # 파일명에서 model_id 추출 (model_0_auc0.9123.pt → 0)
                    import re
                    mid_match = re.search(r'model_(\d+)', pt_file.name)
                    mid = int(mid_match.group(1)) if mid_match else None
                    ind_metrics = model_metrics_map.get(mid, {}) if mid is not None else {}

                    ind_auc = ind_metrics.get("auc", "N/A")
                    ind_acc = ind_metrics.get("accuracy", "N/A")
                    ind_f1 = ind_metrics.get("f1", "N/A")
                    ind_sens = ind_metrics.get("sensitivity", "N/A")
                    ind_spec = ind_metrics.get("specificity", "N/A")

                    version_desc = (
                        f"Model {mid} | Backbone: UNI2-H (1536-dim)\n"
                        f"Test AUC: {ind_auc}, Acc: {ind_acc}, F1: {ind_f1}\n"
                        f"Sens: {ind_sens}, Spec: {ind_spec}"
                    )

                    source = f"runs:/{run_id}/checkpoints/{pt_file.name}"
                    mv = client.create_model_version(
                        name=model_name, source=source, run_id=run_id,
                        description=version_desc
                    )
                    # 버전별 태그: 개별 모델 메트릭
                    client.set_model_version_tag(model_name, mv.version, "embedding", "UNI2-H (1536-dim)")
                    client.set_model_version_tag(model_name, mv.version, "model_id", str(mid))
                    client.set_model_version_tag(model_name, mv.version, "test_auc", str(ind_auc))
                    client.set_model_version_tag(model_name, mv.version, "test_acc", str(ind_acc))
                    client.set_model_version_tag(model_name, mv.version, "test_f1", str(ind_f1))
                    client.set_model_version_tag(model_name, mv.version, "test_sensitivity", str(ind_sens))
                    client.set_model_version_tag(model_name, mv.version, "test_specificity", str(ind_spec))
                    client.set_model_version_tag(model_name, mv.version, "bag_size", str(bag_size))
                    client.set_model_version_tag(model_name, mv.version, "model_arch", "ABMIL_Gated")
                    print(f"  [✓] Registered: {pt_file.name} (version {mv.version}) | AUC: {ind_auc}")
                except Exception as e:
                    print(f"  [!] Register skipped ({pt_file.name}): {e}")

        print(f"[✓] MLflow upload completed: {run_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload ensemble results to MLflow")
    parser.add_argument("--model_save_dir", type=str, required=True,
                        help="Model save directory (e.g., outputs/braf_ensemble_v0.1.6)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bag_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    save_dir = Path(args.model_save_dir)

    # JSON 자동 탐색
    json_path = save_dir / "ensemble_results.json"
    if not json_path.exists():
        json_candidates = list(save_dir.glob("*results*.json"))
        if json_candidates:
            json_path = json_candidates[0]
        else:
            print(f"[!] No results JSON found in {save_dir}")
            sys.exit(1)

    # 체크포인트에서 config 자동 추출
    ckpt_dir = save_dir / "checkpoints"
    ckpt_path = None
    ckpt_config = {}
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.pt"))
        if ckpts:
            ckpt_path = str(ckpts[0])
            ckpt_config = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("config", {})

    lr = args.lr or ckpt_config.get("lr", 1e-4)
    epochs = args.epochs or 100
    bag_size = args.bag_size or ckpt_config.get("bag_size", 5000)
    seed = args.seed or ckpt_config.get("seed", 42)

    print(f"Model dir  : {save_dir}")
    print(f"JSON       : {json_path}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Config     : lr={lr}, epochs={epochs}, bag_size={bag_size}, seed={seed}")

    upload_ensemble_to_mlflow(
        model_save_dir=str(save_dir),
        json_path=str(json_path),
        model_checkpoint_path=ckpt_path,
        lr=lr,
        epochs=epochs,
        bag_size=bag_size,
        seed=seed,
    )
