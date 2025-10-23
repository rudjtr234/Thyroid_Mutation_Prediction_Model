"""
WSI-level training script using full NPY files (no bag sampling)
"""

import os
import sys
import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc,
)
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
sys.path.append(os.path.join(src_dir, 'models'))

from abmil_v2 import ABMILModel, ABMILGatedBaseConfig
from utils.datasets import set_seed
from utils.datasets_wsi import WSIFullDataset, load_json_splits_wsi, _infer_label
from utils.metrics import comprehensive_evaluation

# Visualization import (선택적)
evaluation_dir = os.path.join(src_dir, 'evaluation')
sys.path.insert(0, evaluation_dir)

try:
    from visualization_wsi import visualize_wsi_attention_thumbnail
    VISUALIZATION_WSI_AVAILABLE = True
except ImportError:
    print("[!] Warning: visualization_wsi.py not found. WSI thumbnail visualization will be skipped.")
    VISUALIZATION_WSI_AVAILABLE = False


# =========================
# Metrics & Aggregation Utilities
# =========================

def compute_metrics_with_confusion(y_true, y_pred, y_prob) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "auc": 0.5,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "ppv": 0.0,
            "npv": 0.0,
            "f1": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total > 0 else 0.0
    auc_score = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) if total > 0 else 0.0

    return {
        "accuracy": float(acc),
        "auc": float(auc_score),
        "sensitivity": float(sens),
        "recall": float(sens),
        "specificity": float(spec),
        "precision": float(ppv),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# =========================
# Epoch Execution
# =========================

# ===============================
# run_one_epoch 함수 수정
# ===============================
def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = False,
    return_details: bool = False,
    debug: bool = False,
):
    loss_fn = nn.CrossEntropyLoss()
    model.train() if train else model.eval()

    total_loss = 0.0
    wsi_probs: List[float] = []
    wsi_preds: List[int] = []
    wsi_labels: List[int] = []
    wsi_names: List[str] = []

    with torch.set_grad_enabled(train):
        for batch_idx, (features, label, filename) in enumerate(dataloader):
            features = features.to(device)
            label = label.to(device)

            if isinstance(filename, (list, tuple)):
                filename = filename[0]
            filename_str = str(filename)

            # ==========================================
            # Full NPY 모드: 배치 차원 제거
            # features: (1, N, D) -> (N, D) 필요 없음! 배치 유지
            # label: (1,) -> scalar 필요 없음! 배치 유지
            # ==========================================
            # ABMIL v2는 배치 처리를 지원하므로 squeeze 하지 않음
            
            if debug and batch_idx == 0:
                print(f"[DEBUG] Feature shape: {features.shape}")  # (1, N, 1536)
                print(f"  Label shape: {label.shape}")  # (1,)
                print(f"  Label value: {label.item()}")

            if train:
                optimizer.zero_grad()
                # Training mode: return_extra=False (DDP-safe)
                logits, loss = model(
                    h=features,
                    loss_fn=loss_fn,
                    label=label,
                    return_extra=False
                )
                loss.backward()
                optimizer.step()
            else:
                # Evaluation mode: return_extra=False for consistency
                logits, loss = model(
                    h=features,
                    loss_fn=loss_fn,
                    label=label,
                    return_extra=False
                )

            total_loss += loss.item()
            
            # logits shape: (batch, num_classes) = (1, 2)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            prob = probs.detach().cpu().item()
            pred = preds.detach().cpu().item()
            lbl = label.detach().cpu().item()

            wsi_probs.append(prob)
            wsi_preds.append(pred)
            wsi_labels.append(lbl)
            wsi_names.append(filename_str)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    metrics = compute_metrics_with_confusion(wsi_labels, wsi_preds, wsi_probs)
    metrics["loss"] = float(avg_loss)

    if return_details:
        return metrics, wsi_probs, wsi_labels, wsi_preds, wsi_names
    return metrics



# =========================
# Early Stopping
# =========================

class EarlyStopping:
    def __init__(self, patience: int = 25, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score: float, model: Optional[nn.Module] = None) -> bool:
        improved = False
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            improved = True
        else:
            self.counter += 1

        if improved and model is not None and self.restore_best_weights:
            self.best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# =========================
# Leakage & Label Diagnostics
# =========================

def check_data_leakage(cv_splits: Dict) -> bool:
    print("\n" + "=" * 80)
    print("Checking Data Leakage")
    print("=" * 80)

    all_clear = True
    for fold in cv_splits["folds"]:
        fold_num = fold["fold"]
        train_files = {os.path.basename(p) for p in fold["train_wsis_paths"]}
        val_files = {os.path.basename(p) for p in fold["val_wsis_paths"]}
        test_files = {os.path.basename(p) for p in fold["test_wsis_paths"]}

        print(f"\nFold {fold_num}:")
        print(f"  Train: {len(train_files):3d} WSIs")
        print(f"  Val:   {len(val_files):3d} WSIs")
        print(f"  Test:  {len(test_files):3d} WSIs")

        overlaps = {
            "Train-Val": train_files & val_files,
            "Train-Test": train_files & test_files,
            "Val-Test": val_files & test_files,
        }

        fold_clear = True
        for label, items in overlaps.items():
            if items:
                all_clear = False
                fold_clear = False
                print(f"  ❌ {label} overlap ({len(items)} files): {list(items)[:3]}")
        if fold_clear:
            print("  ✅ No overlap detected")

    print("\n" + "=" * 80)
    if all_clear:
        print("✅ All folds passed leakage check")
    else:
        print("❌ DATA LEAKAGE DETECTED! Please review CV splits")
    print("=" * 80 + "\n")

    return all_clear


def check_label_distribution(cv_splits: Dict):
    print("\n" + "=" * 80)
    print("Label Distribution (WSI-level)")
    print("=" * 80)

    for fold in cv_splits["folds"]:
        fold_num = fold["fold"]
        print(f"\nFold {fold_num}:")
        for split in ["train", "val", "test"]:
            paths = fold[f"{split}_wsis_paths"]
            pos = sum(1 for p in paths if _infer_label(p) == 1)
            neg = len(paths) - pos
            total = len(paths)
            pos_ratio = (pos / total * 100) if total > 0 else 0.0
            print(f"  {split.capitalize():5s}: Pos={pos:3d} ({pos_ratio:5.1f}%), Neg={neg:3d}, Total={total:3d}")

    print("\n" + "=" * 80 + "\n")


# =========================
# Visualization Helpers
# =========================

def plot_roc_curves(fold_results: List[Dict], save_dir: Path):
    plt.figure(figsize=(10, 8))
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold in fold_results:
        fpr = fold.get("test_fpr", [])
        tpr = fold.get("test_tpr", [])
        roc_auc = fold["test_metrics"]["auc"]
        fold_num = fold["fold"]
        if fpr and tpr:
            plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {fold_num} (AUC={roc_auc:.3f})")
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b', linewidth=2,
                 label=f"Mean ROC (AUC={mean_auc:.3f} ± {std_auc:.3f})")
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                         label='± 1 std. dev.')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - WSI Level (Full NPY)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    save_path = save_dir / 'roc_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] ROC curves saved: {save_path}")


def plot_precision_recall_curves(fold_results: List[Dict], save_dir: Path):
    plt.figure(figsize=(10, 8))
    for fold in fold_results:
        precision = fold.get("test_precision", [])
        recall = fold.get("test_recall", [])
        if precision and recall:
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, alpha=0.5, label=f"Fold {fold['fold']} (AUC={pr_auc:.3f})")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - WSI Level (Full NPY)')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)

    save_path = save_dir / 'precision_recall_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Precision-Recall curves saved: {save_path}")


def plot_training_curves(fold_results: List[Dict], save_dir: Path):
    n_folds = len(fold_results)
    fig, axes = plt.subplots(n_folds, 3, figsize=(18, 4 * n_folds))
    if n_folds == 1:
        axes = axes.reshape(1, -1)

    for idx, fold in enumerate(fold_results):
        history = fold['history']
        best_epoch = fold['best_epoch']
        epochs = range(1, len(history['train_loss']) + 1)

        axes[idx, 0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
        axes[idx, 0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
        test_loss = fold['test_metrics'].get('loss')
        if test_loss is not None:
            axes[idx, 0].scatter([best_epoch], [test_loss], color='green', s=100, marker='*',
                                 label=f'Test Loss ({test_loss:.4f})', zorder=5)
        axes[idx, 0].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
        axes[idx, 0].set_title(f"Fold {fold['fold']} - Loss")
        axes[idx, 0].set_xlabel('Epoch')
        axes[idx, 0].set_ylabel('Loss')
        axes[idx, 0].grid(alpha=0.3)
        axes[idx, 0].legend(fontsize=8)

        axes[idx, 1].plot(epochs, history['train_auc'], label='Train AUC', linewidth=2)
        axes[idx, 1].plot(epochs, history['val_auc'], label='Val AUC', linewidth=2)
        test_auc = fold['test_metrics']['auc']
        axes[idx, 1].scatter([best_epoch], [test_auc], color='green', s=100, marker='*',
                             label=f'Test AUC ({test_auc:.3f})', zorder=5)
        axes[idx, 1].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
        axes[idx, 1].set_ylim([0, 1.05])
        axes[idx, 1].set_title(f"Fold {fold['fold']} - AUC")
        axes[idx, 1].set_xlabel('Epoch')
        axes[idx, 1].set_ylabel('AUC')
        axes[idx, 1].grid(alpha=0.3)
        axes[idx, 1].legend(fontsize=8)

        axes[idx, 2].plot(epochs, history['train_acc'], label='Train Acc', linewidth=2)
        axes[idx, 2].plot(epochs, history['val_acc'], label='Val Acc', linewidth=2)
        test_acc = fold['test_metrics']['accuracy']
        axes[idx, 2].scatter([best_epoch], [test_acc], color='green', s=100, marker='*',
                             label=f'Test Acc ({test_acc:.3f})', zorder=5)
        axes[idx, 2].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
        axes[idx, 2].set_ylim([0, 1.05])
        axes[idx, 2].set_title(f"Fold {fold['fold']} - Accuracy")
        axes[idx, 2].set_xlabel('Epoch')
        axes[idx, 2].set_ylabel('Accuracy')
        axes[idx, 2].grid(alpha=0.3)
        axes[idx, 2].legend(fontsize=8)

    plt.tight_layout()
    save_path = save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Training curves saved: {save_path}")


def plot_metric_comparison(fold_results: List[Dict], save_dir: Path):
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'precision', 'npv', 'f1']
    metric_names = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        fold_nums = [f['fold'] for f in fold_results]
        train_vals = [f['best_train_metrics'].get(metric, 0.0) for f in fold_results]
        val_vals = [f['best_val_metrics'].get(metric, 0.0) for f in fold_results]
        test_vals = [f['test_metrics'].get(metric, 0.0) for f in fold_results]

        x = np.arange(len(fold_nums))
        width = 0.25
        bars1 = axes[idx].bar(x - width, train_vals, width, label='Train', color='skyblue', alpha=0.8)
        bars2 = axes[idx].bar(x, val_vals, width, label='Val', color='orange', alpha=0.8)
        bars3 = axes[idx].bar(x + width, test_vals, width, label='Test', color='lightgreen', alpha=0.8)

        for bar, val in zip(bars3, test_vals):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                           f"{val:.3f}", ha='center', va='bottom', fontsize=8)

        axes[idx].axhline(np.mean(test_vals), color='green', linestyle='--', alpha=0.5, linewidth=1)
        axes[idx].set_title(f'{name} Comparison')
        axes[idx].set_xlabel('Fold')
        axes[idx].set_ylabel(name)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(fold_nums)
        axes[idx].set_ylim([0, 1.05])
        axes[idx].grid(alpha=0.3, axis='y')
        axes[idx].legend(fontsize=8)

    fig.delaxes(axes[-1])
    plt.tight_layout()
    save_path = save_dir / 'metric_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Metric comparison saved: {save_path}")


def plot_confusion_matrices(fold_results: List[Dict], save_dir: Path):
    n_folds = len(fold_results)
    cols = min(3, n_folds)
    rows = (n_folds + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n_folds > 1 else [axes]

    for idx, fold in enumerate(fold_results):
        metrics = fold['test_metrics']
        cm = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
        im = axes[idx].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[idx].set_title(f"Fold {fold['fold']} Confusion Matrix")
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(['Pred Neg', 'Pred Pos'])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_yticklabels(['True Neg', 'True Pos'])

        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, int(cm[i, j]), ha='center', va='center',
                               color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    for idx in range(n_folds, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    save_path = save_dir / 'confusion_matrices.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Confusion matrices saved: {save_path}")


# =========================
# Reporting Helpers
# =========================

def print_fold_table(fold_num: int, train_metrics: Dict, val_metrics: Dict, test_metrics: Dict):
    print(f"\n{'=' * 80}")
    print(f"Fold {fold_num} Results")
    print(f"{'=' * 80}")

    print(
        f"| {'Set':8s} | {'Accuracy':8s} | {'AUC':8s} | {'Sensitivity':11s} | "
        f"{'Specificity':11s} | {'Precision':9s} | {'NPV':8s} | {'F1-score':8s} |"
    )
    print("|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 13 + "|" +
          "-" * 13 + "|" + "-" * 11 + "|" + "-" * 10 + "|" + "-" * 10 + "|")

    def fmt_row(name: str, metrics: Dict):
        return (
            f"| {name:8s} | {metrics.get('accuracy', 0.0):8.2f} | "
            f"{metrics.get('auc', 0.0):8.2f} | {metrics.get('sensitivity', 0.0):11.2f} | "
            f"{metrics.get('specificity', 0.0):11.2f} | {metrics.get('precision', 0.0):9.2f} | "
            f"{metrics.get('npv', 0.0):8.2f} | {metrics.get('f1', 0.0):8.2f} |"
        )

    print(fmt_row('Train', train_metrics))
    print(fmt_row('Val', val_metrics))
    print(fmt_row('Test', test_metrics))


# =========================
# Model Saving
# =========================

def save_model_checkpoint(model, fold_idx, fold_result, save_dir, args, is_best=False):
    save_dir = Path(save_dir) / 'checkpoints'
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'fold': fold_idx + 1,
        'model_state_dict': model.state_dict(),
        'accuracy': fold_result['test_metrics']['accuracy'],
        'auc': fold_result['test_metrics']['auc'],
        'config': {
            'lr': args.lr,
            'seed': args.seed,
            'model_version': 'Thyroid_prediction_model_v0.3.0_full_npy',
            'model': 'ABMILGatedBase',
            'mode': 'full_npy',
        },
    }

    filename = f"{'best_' if is_best else ''}model_fold{fold_idx + 1}_auc{fold_result['test_metrics']['auc']:.4f}.pt"
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)
    print(f"[✓] Model saved: {checkpoint_path}")
    return checkpoint_path


# =========================
# JSON Serialization Helper
# =========================

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {str(k): convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# =========================
# Cross-Validation Loop
# =========================

def run_k_fold_cv(cv_splits, args, device):
    print(f"\nRunning {len(cv_splits['folds'])}-Fold CV (Full NPY Mode)")
    print("=" * 60)

    all_fold_results = []
    all_predictions, all_true_labels = [], []
    saved_model_paths = []
    config = ABMILGatedBaseConfig()

    for fold_data in cv_splits['folds']:
        fold_idx = fold_data['fold'] - 1
        print(f"\nFold {fold_data['fold']}/{len(cv_splits['folds'])}")

        train_dataset = WSIFullDataset(fold_data['train_wsis_paths'])
        val_dataset = WSIFullDataset(fold_data['val_wsis_paths'])
        test_dataset = WSIFullDataset(fold_data['test_wsis_paths'])

        print(f"Dataset sizes -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        model = ABMILModel(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)

        history = {
            'train_loss': [], 'train_auc': [], 'train_acc': [],
            'val_loss': [], 'val_auc': [], 'val_acc': []
        }
        best_train_metrics = {}
        best_val_metrics = {}
        best_epoch = 0

        for epoch in range(args.epochs):
            train_metrics = run_one_epoch(model, train_loader, device, optimizer=optimizer, train=True, debug=(args.debug and epoch==0))
            val_metrics = run_one_epoch(model, val_loader, device, train=False, debug=(args.debug and epoch==0))

            scheduler.step(1 - val_metrics.get('auc', 0.0))

            history['train_loss'].append(train_metrics['loss'])
            history['train_auc'].append(train_metrics['auc'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_auc'].append(val_metrics['auc'])
            history['val_acc'].append(val_metrics['accuracy'])

            if args.debug or (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{args.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | Train AUC: {train_metrics['auc']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | Val AUC: {val_metrics['auc']:.4f}"
                )

            if val_metrics['auc'] > best_val_metrics.get('auc', -float('inf')):
                best_val_metrics = val_metrics.copy()
                best_train_metrics = train_metrics.copy()
                best_epoch = epoch + 1

            if early_stopping(val_metrics['auc'], model):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        early_stopping.restore_best(model)
        if best_epoch == 0:
            best_epoch = len(history['train_loss'])

        test_metrics, test_probs, test_labels, test_preds, test_names = run_one_epoch(
            model, test_loader, device, train=False, return_details=True, debug=args.debug
        )

        if len(set(test_labels)) > 1:
            fpr, tpr, _ = roc_curve(test_labels, test_probs)
            precision, recall, _ = precision_recall_curve(test_labels, test_probs)
            fpr, tpr, precision, recall = fpr.tolist(), tpr.tolist(), precision.tolist(), recall.tolist()
        else:
            fpr, tpr = [0.0, 1.0], [0.0, 1.0]
            precision, recall = [1.0], [0.0]

        fold_result = {
            'fold': fold_data['fold'],
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'best_epoch': best_epoch,
            'best_train_metrics': best_train_metrics,
            'best_val_metrics': best_val_metrics,
            'test_metrics': test_metrics,
            'history': history,
            'test_fpr': fpr,
            'test_tpr': tpr,
            'test_precision': precision,
            'test_recall': recall,
        }

        all_fold_results.append(fold_result)
        all_predictions.extend(test_probs)
        all_true_labels.extend(test_labels)

        print(
            f"Fold {fold_data['fold']} Test -> "
            f"AUC: {test_metrics['auc']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, "
            f"Sensitivity: {test_metrics['sensitivity']:.4f}, Specificity: {test_metrics['specificity']:.4f}"
        )
        print_fold_table(fold_data['fold'], best_train_metrics, best_val_metrics, test_metrics)

# ===============================
# WSI Thumbnail Visualization 부분 수정
# ===============================
# run_k_fold_cv 함수 내부의 시각화 부분:

        # ================================
        # ✨ WSI Thumbnail Visualization ✨
        # ================================
        if fold_data['fold'] == 1 and VISUALIZATION_WSI_AVAILABLE:
            print(f"\n{'─'*80}")
            print(f"Generating WSI Attention Thumbnails")
            print(f"{'─'*80}")
            
            # 시각화할 WSI 선택 (test set의 BRAF+ 샘플)
            target_wsi_names = ['TC_04_3712', 'TC_04_4612', 'TC_04_5975']
            
            viz_dir = Path(args.model_save_dir) / "wsi_thumbnails"
            
            # 모델을 evaluation 모드로 설정
            model.eval()
            
            # Attention scores 수집
            attention_dict = {}
            with torch.no_grad():
                for features, label, filename in test_loader:
                    wsi_name = filename[0] if isinstance(filename, (list, tuple)) else filename
                    wsi_name = Path(wsi_name).stem
                    
                    if wsi_name not in target_wsi_names:
                        continue
                    
                    features = features.to(device)
                    
                    # ✨ return_extra=True로 attention 가져오기
                    outputs = model(
                        h=features,
                        loss_fn=None,
                        label=None,
                        return_attention=True,
                        return_extra=True
                    )
                    
                    # ✨ 새로운 구조: outputs['attention']
                    if outputs['attention'] is not None:
                        # attention shape: (1, 1, N) -> (N,)
                        attn_scores = outputs['attention'].squeeze().cpu().numpy()
                        attention_dict[wsi_name] = attn_scores
            
            try:
                visualize_wsi_attention_thumbnail(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    patch_base_dir=args.patch_dir,
                    save_dir=viz_dir,
                    fold_num=fold_data['fold'],
                    wsi_names=target_wsi_names,
                    max_thumbnail_size=2048,
                    grid_cols=None,
                    overlay_alpha=0.5,
                    show_colorbar=True,
                    precomputed_attention=attention_dict  # ✨ 미리 계산한 attention 전달
                )
            except Exception as e:
                print(f"[!] Warning: Failed to generate WSI thumbnails: {e}")
                import traceback
                traceback.print_exc()

                
        if args.save_model:
            if args.save_best_only:
                prev_best = max((r['test_metrics']['auc'] for r in all_fold_results[:-1]), default=-float('inf'))
                if test_metrics['auc'] > prev_best:
                    path = save_model_checkpoint(model, fold_idx, fold_result, args.model_save_dir, args, is_best=True)
                    saved_model_paths.append(path)
            else:
                path = save_model_checkpoint(model, fold_idx, fold_result, args.model_save_dir, args)
                saved_model_paths.append(path)

    return all_fold_results, all_predictions, all_true_labels, saved_model_paths


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description='WSI-level training script using full NPY files')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing WSI embeddings')
    parser.add_argument('--cv_split_file', type=str, required=True,
                       help='Path to CV split JSON file')
    parser.add_argument('--patch_dir', type=str, 
                       default="/data/143/member/kwk/dl/thyroid/image/slide-v1-240412/patch",
                       help='Patch image directory for WSI thumbnail visualization')
    
    # Model save arguments
    parser.add_argument('--model_save_dir', type=str,
                       default='/home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model/outputs/Thyroid_prediction_model_v0.3.0_full_npy',
                       help='Directory to save model checkpoints and results')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=25,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Save options
    parser.add_argument('--save_model', action='store_true',
                       help='Save model checkpoints')
    parser.add_argument('--save_best_only', action='store_true',
                       help='Save only the best model across folds')
    
    # Debug option
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    
    args = parser.parse_args()

    if args.debug:
        print("[DEBUG] Arguments:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

            
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)

    cv_splits = load_json_splits_wsi(args.cv_split_file, args.data_root)
    leakage_ok = check_data_leakage(cv_splits)
    check_label_distribution(cv_splits)

    if not leakage_ok:
        response = input("Data leakage detected. Continue training? (yes/no): ")
        if response.strip().lower() != 'yes':
            print("Training aborted due to leakage.")
            return

    fold_results, predictions, true_labels, model_paths = run_k_fold_cv(cv_splits, args, device)

    print("\n" + "=" * 80)
    print("Generating Visualization Plots")
    print("=" * 80 + "\n")
    viz_dir = Path(args.model_save_dir) / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curves(fold_results, viz_dir)
    plot_precision_recall_curves(fold_results, viz_dir)
    plot_training_curves(fold_results, viz_dir)
    plot_metric_comparison(fold_results, viz_dir)
    plot_confusion_matrices(fold_results, viz_dir)

    summary_stats = {}
    metrics_order = ['accuracy', 'auc', 'sensitivity', 'specificity', 'precision', 'npv', 'f1']
    metric_names = {
        'accuracy': 'Accuracy',
        'auc': 'AUC',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'precision': 'Precision',
        'npv': 'NPV',
        'f1': 'F1-score',
    }
    for set_name, metric_key in [('train', 'best_train_metrics'), ('val', 'best_val_metrics'), ('test', 'test_metrics')]:
        summary_stats[set_name] = {}
        for metric in metrics_order:
            values = [fold[metric_key].get(metric, 0.0) for fold in fold_results]
            summary_stats[set_name][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values],
            }

    print("\n" + "=" * 80)
    print("WSI Results (Mean ± Std Across Folds) - Full NPY Mode")
    print("=" * 80 + "\n")
    header = f"| {'Set':8s} |" + ''.join([f" {metric_names[metric]:17s} |" for metric in metrics_order])
    print(header)
    print("|" + "-" * 10 + "|" + "-" * 19 * len(metrics_order) + "|")
    for set_name, label in [('train', 'Train'), ('val', 'Val'), ('test', 'Test')]:
        row = f"| {label:8s} |"
        for metric in metrics_order:
            stats = summary_stats[set_name][metric]
            row += f" {stats['mean']:.3f} ± {stats['std']:.3f} |"
        print(row)

    total_tp = sum(f['test_metrics']['tp'] for f in fold_results)
    total_tn = sum(f['test_metrics']['tn'] for f in fold_results)
    total_fp = sum(f['test_metrics']['fp'] for f in fold_results)
    total_fn = sum(f['test_metrics']['fn'] for f in fold_results)
    print("\nTotal Confusion Matrix (Test):")
    print(f"  TP: {total_tp:4d} | FN: {total_fn:4d}")
    print(f"  FP: {total_fp:4d} | TN: {total_tn:4d}")

    results = {
        "summary_statistics": summary_stats,
        "folds": fold_results,
        "mode": "full_npy",
    }

    if len(set(true_labels)) > 1:
        eval_results = comprehensive_evaluation(true_labels, predictions)
        results['final_aggregated'] = eval_results
        print("\n" + "=" * 80)
        print("Final Aggregated Results (All Folds Combined)")
        print("=" * 80)
        for key, value in eval_results.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                print(f"{key:20s}: {float(value):.4f}")

    results_path = Path(args.model_save_dir) / 'results_wsi_level_full_npy.json'
    with open(results_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\n[✓] Results saved: {results_path}")

    if args.save_model and model_paths:
        print(f"[✓] Saved {len(model_paths)} model checkpoints")

    print(f"[✓] All visualizations saved in: {viz_dir}")
    print("[✓] Training completed!")


if __name__ == '__main__':
    main()
