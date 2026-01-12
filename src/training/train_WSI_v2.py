# -*- coding: utf-8 -*-
"""
WSI-level training script using full NPY files (no bag sampling).
Outputs:
  - results_cv_summary.json
  - attention_scores/attention_scores_fold{N}.json
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, precision_recall_curve
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
sys.path.append(os.path.join(src_dir, "models"))

from abmil_v2 import ABMILModel, ABMILGatedBaseConfig
from utils.datasets import set_seed
from utils.datasets_wsi import WSIFullDataset, load_json_splits_wsi

evaluation_dir = os.path.join(src_dir, "evaluation")
sys.path.insert(0, evaluation_dir)
from metric import compute_metrics_with_confusion, compute_summary_statistics, print_summary_statistics


def configure_threads(num_threads: int = 1) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass


def evaluate_full_wsi_for_visualization(
    model: nn.Module,
    test_wsi_list: List[str],
    device: torch.device,
    meta_npy_dir: str,
    nonmeta_npy_dir: str,
) -> Dict[str, Dict]:
    model.eval()
    attention_scores_dict = {}

    successful = 0
    failed = 0

    with torch.no_grad():
        for wsi_file in test_wsi_list:
            wsi_name = Path(wsi_file).stem
            meta_path = Path(meta_npy_dir) / f"{wsi_name}.npy"
            nonmeta_path = Path(nonmeta_npy_dir) / f"{wsi_name}.npy"

            if meta_path.exists():
                full_features = np.load(meta_path)
                data_type = "meta"
            elif nonmeta_path.exists():
                full_features = np.load(nonmeta_path)
                data_type = "nonmeta"
            else:
                failed += 1
                continue

            features_tensor = torch.from_numpy(full_features).float().unsqueeze(0).to(device)

            try:
                results_dict = model(
                    h=features_tensor,
                    loss_fn=None,
                    label=None,
                    return_attention=True,
                    return_extra=True,
                )

                attention_weights = results_dict.get("attention")
                if attention_weights is None:
                    failed += 1
                    continue

                if attention_weights.dim() == 3:
                    attn_raw = attention_weights[0, 0, :]
                elif attention_weights.dim() == 2:
                    attn_raw = attention_weights[0, :]
                else:
                    failed += 1
                    continue

                attn_scores = F.softmax(attn_raw, dim=0).cpu().numpy()

                logits = results_dict["logits"]
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                attention_scores_dict[wsi_name] = {
                    "scores": attn_scores.tolist(),
                    "n_patches": len(attn_scores),
                    "predicted_label": int(preds.cpu().numpy()[0]),
                    "pred_prob": float(probs.cpu().numpy()[0]),
                    "data_type": data_type,
                }
                successful += 1
            except Exception:
                failed += 1

    print(f"[OK] Full attention extracted: {successful}/{len(test_wsi_list)}")
    if failed > 0:
        print(f"[!] Failed: {failed}/{len(test_wsi_list)}")

    return attention_scores_dict


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0

    def __call__(self, score: float, model: Optional[nn.Module] = None, epoch: Optional[int] = None) -> bool:
        improved = False
        if self.best_score is None:
            self.best_score = score
            improved = True
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            improved = True
        else:
            self.counter += 1

        if improved and model is not None:
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch is not None:
                self.best_epoch = epoch

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = False,
    return_details: bool = False,
) -> Tuple[Dict[str, float], List[float], List[int], List[int], List[str]]:
    if train:
        model.train()
    else:
        model.eval()

    all_probs, all_labels, all_preds, all_names = [], [], [], []
    total_loss = 0.0

    for features, label, filename in dataloader:
        features = features.to(device)
        label = label.to(device)

        if train:
            optimizer.zero_grad()
            logits, loss = model(h=features, loss_fn=loss_fn, label=label, return_extra=False)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits, loss = model(h=features, loss_fn=loss_fn, label=label, return_extra=False)

        if loss is not None:
            total_loss += float(loss.item())

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(label.detach().cpu().numpy().tolist())
        all_names.extend(filename)

    metrics = compute_metrics_with_confusion(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / max(1, len(dataloader))

    if return_details:
        return metrics, all_probs, all_labels, all_preds, all_names
    return metrics, [], [], [], []


def save_model_checkpoint(
    model: nn.Module,
    fold_idx: int,
    fold_result: Dict,
    save_dir: Path,
    is_best: bool = False,
) -> Path:
    save_dir = save_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "fold": fold_idx + 1,
        "model_state_dict": model.state_dict(),
        "accuracy": fold_result["test_metrics"]["accuracy"],
        "auc": fold_result["test_metrics"]["auc"],
        "optimal_threshold": fold_result["optimal_threshold"],
        "config": {
            "model": "ABMILGatedBase",
        },
    }

    filename = f"{'best_' if is_best else ''}model_fold{fold_idx + 1}_auc{fold_result['test_metrics']['auc']:.4f}.pt"
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)
    print(f"[OK] Model saved: {checkpoint_path}")
    return checkpoint_path


def run_k_fold_cv(cv_splits: dict, args, device: torch.device):
    fold_results = []
    all_predictions, all_true_labels = [], []
    model_paths = []

    for fold_data in cv_splits.get("folds", []):
        fold_num = fold_data["fold"]
        print("\n" + "=" * 80)
        print(f"Fold {fold_num}")
        print("=" * 80)

        train_dataset = WSIFullDataset(fold_data["train_wsis_paths"])
        val_dataset = WSIFullDataset(fold_data["val_wsis_paths"])
        test_dataset = WSIFullDataset(fold_data["test_wsis_paths"])

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        config = ABMILGatedBaseConfig()
        model = ABMILModel(config).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
        history = {
            "train_loss": [],
            "train_auc": [],
            "train_acc": [],
            "val_loss": [],
            "val_auc": [],
            "val_acc": [],
        }
        best_val_metrics = None
        best_train_metrics = None
        best_epoch = 0

        for epoch in range(args.epochs):
            train_metrics, _, _, _, _ = run_one_epoch(
                model, train_loader, device, loss_fn, optimizer=optimizer, train=True
            )
            val_metrics, _, _, _, _ = run_one_epoch(
                model, val_loader, device, loss_fn, optimizer=None, train=False
            )

            history["train_loss"].append(train_metrics["loss"])
            history["train_auc"].append(train_metrics["auc"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_auc"].append(val_metrics["auc"])
            history["val_acc"].append(val_metrics["accuracy"])

            if best_val_metrics is None or val_metrics["auc"] > best_val_metrics["auc"]:
                best_val_metrics = val_metrics
                best_train_metrics = train_metrics
                best_epoch = epoch + 1

            if early_stopping(val_metrics["auc"], model, epoch=epoch + 1):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        early_stopping.restore_best(model)

        test_metrics, test_probs, test_labels, test_preds, test_names = run_one_epoch(
            model, test_loader, device, loss_fn, optimizer=None, train=False, return_details=True
        )

        if len(set(test_labels)) > 1:
            fpr, tpr, thresholds_roc = roc_curve(test_labels, test_probs)
            precision, recall, thresholds_pr = precision_recall_curve(test_labels, test_probs)

            youden_index = tpr - fpr
            optimal_idx = int(np.argmax(youden_index))
            optimal_threshold = float(thresholds_roc[optimal_idx])
            optimal_youden = float(youden_index[optimal_idx])
            optimal_preds = [1 if p >= optimal_threshold else 0 for p in test_probs]
            optimal_metrics = compute_metrics_with_confusion(test_labels, optimal_preds, test_probs)
        else:
            fpr, tpr = [0.0, 1.0], [0.0, 1.0]
            precision, recall = [1.0], [0.0]
            optimal_threshold = 0.5
            optimal_youden = 0.0
            optimal_metrics = test_metrics.copy()

        full_attention_scores = {}
        if args.generate_plots:
            test_wsi_files = [Path(name).stem for name in test_names]
            full_attention_scores = evaluate_full_wsi_for_visualization(
                model,
                test_wsi_files,
                device,
                args.meta_npy_dir,
                args.nonmeta_npy_dir,
            )
            wsi_to_label = {Path(p).stem: int(lbl) for p, lbl in zip(test_names, test_labels)}
            for wsi_name in full_attention_scores:
                if wsi_name in wsi_to_label:
                    full_attention_scores[wsi_name]["true_label"] = wsi_to_label[wsi_name]

        fold_result = {
            "fold": fold_num,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "best_epoch": best_epoch,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "optimal_threshold": optimal_threshold,
            "optimal_youden_index": optimal_youden,
            "optimal_threshold_metrics": optimal_metrics,
            "history": history,
            "test_fpr": list(map(float, fpr)),
            "test_tpr": list(map(float, tpr)),
            "test_precision": list(map(float, precision)),
            "test_recall": list(map(float, recall)),
            "full_attention_scores": full_attention_scores,
        }

        fold_results.append(fold_result)
        all_predictions.extend(test_preds)
        all_true_labels.extend(test_labels)

        if args.save_model:
            checkpoint_path = save_model_checkpoint(
                model, fold_num - 1, fold_result, Path(args.model_save_dir)
            )
            model_paths.append(checkpoint_path)

    return fold_results, all_predictions, all_true_labels, model_paths


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.int16)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj


def save_results_json(
    fold_results: List[Dict],
    summary_stats: Dict,
    save_dir: Path,
) -> Tuple[Path, List[Path]]:
    save_dir.mkdir(parents=True, exist_ok=True)

    cv_summary = {
        "mode": "full_npy",
        "folds": [],
        "summary": summary_stats,
    }

    for fold_result in fold_results:
        fold_summary = {
            "fold": fold_result["fold"],
            "train_metrics": fold_result["best_train_metrics"],
            "val_metrics": fold_result["best_val_metrics"],
            "test_metrics": fold_result["test_metrics"],
            "optimal_threshold": fold_result["optimal_threshold"],
            "optimal_youden_index": fold_result["optimal_youden_index"],
            "optimal_threshold_metrics": fold_result["optimal_threshold_metrics"],
            "training_history": fold_result["history"],
            "roc_curve": {
                "fpr": fold_result["test_fpr"],
                "tpr": fold_result["test_tpr"],
            },
            "pr_curve": {
                "precision": fold_result["test_precision"],
                "recall": fold_result["test_recall"],
            },
        }
        cv_summary["folds"].append(fold_summary)

    cv_summary_path = save_dir / "results_cv_summary.json"
    with open(cv_summary_path, "w") as f:
        json.dump(convert_numpy(cv_summary), f, indent=2)
    print(f"[OK] CV Summary saved: {cv_summary_path}")

    attention_dir = save_dir / "attention_scores"
    attention_dir.mkdir(parents=True, exist_ok=True)

    saved_attention_files = []
    for fold_result in fold_results:
        if fold_result.get("full_attention_scores"):
            fold_num = fold_result["fold"]
            attention_data = {
                "fold": fold_num,
                "n_wsis": len(fold_result["full_attention_scores"]),
                "attention_scores": fold_result["full_attention_scores"],
            }
            attention_path = attention_dir / f"attention_scores_fold{fold_num}.json"
            with open(attention_path, "w") as f:
                json.dump(convert_numpy(attention_data), f, indent=2)
            saved_attention_files.append(attention_path)
            print(f"[OK] Fold {fold_num} attention scores saved: {attention_path}")

    if not saved_attention_files:
        print("[!] No attention scores were saved (--generate_plots not enabled)")

    return cv_summary_path, saved_attention_files


def print_cv_summary(fold_results: List[Dict]) -> Dict:
    summary_stats = compute_summary_statistics(fold_results)
    print_summary_statistics(summary_stats)
    return summary_stats


def run_training(args):
    configure_threads()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_splits = load_json_splits_wsi(args.cv_split_file, args.data_root)
    print(f"Loaded {len(cv_splits.get('folds', []))} folds")

    fold_results, predictions, true_labels, model_paths = run_k_fold_cv(
        cv_splits, args, device
    )

    summary_stats = print_cv_summary(fold_results)
    cv_path, attention_paths = save_results_json(
        fold_results, summary_stats, Path(args.model_save_dir)
    )

    checkpoint_for_mlflow = None
    if args.save_model and model_paths:
        best_idx = int(np.argmax([fr["test_metrics"]["auc"] for fr in fold_results]))
        best_fold_num = fold_results[best_idx]["fold"]
        for path in reversed(model_paths):
            if f"fold{best_fold_num}" in path.name:
                checkpoint_for_mlflow = path
                break
        if checkpoint_for_mlflow is None:
            checkpoint_for_mlflow = model_paths[-1]

    if args.generate_plots:
        viz_dir = Path(args.model_save_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        try:
            from evaluation.metric import generate_all_plots
            from evaluation.visualization_wsi import generate_attention_heatmaps_from_results

            generate_all_plots(fold_results, viz_dir)
            generate_attention_heatmaps_from_results(
                results_json_path=str(cv_path),
                json_meta_dir=args.json_meta_dir,
                json_nonmeta_dir=args.json_nonmeta_dir,
                save_dir=str(viz_dir),
                fold_num="best",
                n_correct=3,
                n_incorrect=3,
            )
        except ImportError as exc:
            print(f"[!] Visualization import failed: {exc}")

    return {
        "model_save_dir": args.model_save_dir,
        "json_path": str(cv_path),
        "model_checkpoint_path": str(checkpoint_for_mlflow) if checkpoint_for_mlflow else None,
        "args": args,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WSI-level training script using full NPY files"
    )

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

    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    parser.add_argument("--model_save_dir", type=str, default="./outputs",
                        help="Directory to save model and results")
    parser.add_argument("--save_model", action="store_true",
                        help="Whether to save model checkpoints")
    parser.add_argument("--save_best_only", action="store_true",
                        help="Only save the best model across folds")
    parser.add_argument("--generate_plots", action="store_true",
                        help="Generate visualization plots + attention heatmaps")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with verbose output")

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
