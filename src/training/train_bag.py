"""
active_code - with attention scores extraction

CUDA_VISIBLE_DEVICES=4 python train_bag.py \
  --data_root /data/143/member/jks/Thyroid_Mutation_dataset/embeddings \
  --model_save_dir /home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model_v2/outputs/Thyroid_prediction_model_v0.5.0 \
  --cv_split_file /home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model_v2/src/utils/cv_splits/cv_splits/cv_splits_balanced_k5_seed42_v0.3.0.json \
  --epochs 100 \
  --lr 1e-5 \
  --bag_size 500 \
  --seed 42 \
  --save_model \
  --save_best_only \
  --debug

"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
import warnings
import sys
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)
sys.path.append(os.path.join(src_dir, 'models'))

# Models & Utils import
from abmil import ABMILModel, ABMILGatedBaseConfig
from utils.datasets import ThyroidWSIDataset, set_seed
from utils.metrics import comprehensive_evaluation
from torch.utils.data import DataLoader


# =========================
# EarlyStopping
# =========================
class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model=None):
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

        if improved and model is not None and self.restore_best_weights:
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# =========================
# 지표 계산 함수
# =========================
def compute_metrics_with_confusion(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": acc,
        "auc": auc,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


# =========================
# Visualization Functions
# =========================
def plot_roc_curves(fold_results, save_dir):
    """각 fold별 ROC curve와 평균 ROC curve 그리기"""
    plt.figure(figsize=(10, 8))
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for fold_result in fold_results:
        fold_num = fold_result['fold']
        fpr = fold_result['test_fpr']
        tpr = fold_result['test_tpr']
        roc_auc = fold_result['test_metrics']['auc']
        
        # 각 fold의 ROC curve
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold_num} (AUC = {roc_auc:.3f})')
        
        # 평균 계산을 위해 interpolation
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
    
    # 평균 ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b', linewidth=2,
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    
    # 표준편차 영역
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                     label='± 1 std. dev.')
    
    # 대각선 (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Cross Validation', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    
    save_path = Path(save_dir) / 'roc_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] ROC curves saved: {save_path}")


def plot_precision_recall_curves(fold_results, save_dir):
    """각 fold별 Precision-Recall curve 그리기"""
    plt.figure(figsize=(10, 8))
    
    for fold_result in fold_results:
        fold_num = fold_result['fold']
        precision = fold_result['test_precision']
        recall = fold_result['test_recall']
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, alpha=0.5, linewidth=2,
                label=f'Fold {fold_num} (AUC = {pr_auc:.3f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Cross Validation', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=9)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    save_path = Path(save_dir) / 'precision_recall_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Precision-Recall curves saved: {save_path}")


def plot_training_curves(fold_results, save_dir):
    """각 fold별 학습 곡선 (Loss, AUC, Accuracy) 그리기 - Train/Val/Test 포함"""
    n_folds = len(fold_results)
    fig, axes = plt.subplots(n_folds, 3, figsize=(18, 4*n_folds))
    
    if n_folds == 1:
        axes = axes.reshape(1, -1)
    
    for idx, fold_result in enumerate(fold_results):
        fold_num = fold_result['fold']
        history = fold_result['history']
        best_epoch = fold_result['best_epoch']
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        axes[idx, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[idx, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[idx, 0].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
        axes[idx, 0].set_xlabel('Epoch', fontsize=10)
        axes[idx, 0].set_ylabel('Loss', fontsize=10)
        axes[idx, 0].set_title(f'Fold {fold_num} - Loss', fontsize=11, fontweight='bold')
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 0].grid(alpha=0.3)
        
        # AUC
        axes[idx, 1].plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
        axes[idx, 1].plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
        test_auc = fold_result['test_metrics']['auc']
        axes[idx, 1].scatter([best_epoch], [test_auc], 
                           color='green', s=100, marker='*', label=f'Test AUC ({test_auc:.3f})', zorder=5)
        axes[idx, 1].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
        axes[idx, 1].set_xlabel('Epoch', fontsize=10)
        axes[idx, 1].set_ylabel('AUC', fontsize=10)
        axes[idx, 1].set_title(f'Fold {fold_num} - AUC', fontsize=11, fontweight='bold')
        axes[idx, 1].legend(fontsize=8)
        axes[idx, 1].grid(alpha=0.3)
        axes[idx, 1].set_ylim([0, 1.05])
        
        # Accuracy
        axes[idx, 2].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[idx, 2].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        test_acc = fold_result['test_metrics']['accuracy']
        axes[idx, 2].scatter([best_epoch], [test_acc], 
                           color='green', s=100, marker='*', label=f'Test Acc ({test_acc:.3f})', zorder=5)
        axes[idx, 2].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
        axes[idx, 2].set_xlabel('Epoch', fontsize=10)
        axes[idx, 2].set_ylabel('Accuracy', fontsize=10)
        axes[idx, 2].set_title(f'Fold {fold_num} - Accuracy', fontsize=11, fontweight='bold')
        axes[idx, 2].legend(fontsize=8)
        axes[idx, 2].grid(alpha=0.3)
        axes[idx, 2].set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Training curves saved: {save_path}")


def plot_metric_comparison(fold_results, save_dir):
    """각 fold별 최종 train/val/test metrics 비교 bar plot"""
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']
    metric_names = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        fold_nums = [r['fold'] for r in fold_results]
        
        # Train, Val, Test 값 추출
        train_values = [r['best_train_metrics'][metric] for r in fold_results]
        val_values = [r['best_val_metrics'][metric] for r in fold_results]
        test_values = [r['test_metrics'][metric] for r in fold_results]
        
        x = np.arange(len(fold_nums))
        width = 0.25
        
        # Bar plot
        bars1 = axes[idx].bar(x - width, train_values, width, label='Train', color='skyblue', alpha=0.8)
        bars2 = axes[idx].bar(x, val_values, width, label='Val', color='orange', alpha=0.8)
        bars3 = axes[idx].bar(x + width, test_values, width, label='Test', color='lightgreen', alpha=0.8)
        
        # 평균선
        axes[idx].axhline(y=np.mean(train_values), color='blue', linestyle='--', alpha=0.5, linewidth=1)
        axes[idx].axhline(y=np.mean(val_values), color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[idx].axhline(y=np.mean(test_values), color='green', linestyle='--', alpha=0.5, linewidth=1,
                         label=f'Test Mean: {np.mean(test_values):.3f}')
        
        axes[idx].set_xlabel('Fold', fontsize=10)
        axes[idx].set_ylabel(name, fontsize=10)
        axes[idx].set_title(f'{name} Comparison', fontsize=11, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(fold_nums)
        axes[idx].set_ylim([0, 1.05])
        axes[idx].legend(fontsize=8)
        axes[idx].grid(alpha=0.3, axis='y')
        
        # 각 bar 위에 값 표시 (test만)
        for bar, val in zip(bars3, test_values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 마지막 subplot 제거
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'metric_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Metric comparison saved: {save_path}")


def plot_confusion_matrices(fold_results, save_dir):
    """각 fold별 confusion matrix 시각화"""
    n_folds = len(fold_results)
    cols = min(3, n_folds)
    rows = (n_folds + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_folds == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, fold_result in enumerate(fold_results):
        fold_num = fold_result['fold']
        metrics = fold_result['test_metrics']
        
        cm = np.array([[metrics['tn'], metrics['fp']],
                       [metrics['fn'], metrics['tp']]])
        
        im = axes[idx].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[idx].set_title(f'Fold {fold_num} Confusion Matrix')
        
        # 색상 바
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # 축 레이블
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Predicted Neg', 'Predicted Pos'])
        axes[idx].set_yticklabels(['Actual Neg', 'Actual Pos'])
        
        # 값 표시
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black",
                             fontsize=14, fontweight='bold')
    
    # 사용하지 않는 subplot 제거
    for idx in range(n_folds, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'confusion_matrices.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Confusion matrices saved: {save_path}")


# =========================
# Train / Val / Test 함수
# =========================
def run_one_epoch(model, dataloader, device, optimizer=None, train=False):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    loss_fn = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(train):
        for features, label, filename in dataloader:
            features = features.to(device)
            label = label.to(device)

            if train:
                optimizer.zero_grad()
                results_dict, _ = model(h=features, loss_fn=loss_fn, label=label)
                loss = results_dict['loss']
                logits = results_dict['logits']
                loss.backward()
                optimizer.step()
            else:
                results_dict, _ = model(h=features, loss_fn=loss_fn, label=label)
                loss = results_dict['loss']
                logits = results_dict['logits']

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().detach().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

    if len(set(all_labels)) > 1:
        metrics = compute_metrics_with_confusion(all_labels, all_preds, all_probs)
    else:
        metrics = {
            "accuracy": 0.0, "auc": 0.5,
            "sensitivity": 0.0, "specificity": 0.0,
            "ppv": 0.0, "npv": 0.0,
            "f1": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0
        }

    metrics["loss"] = avg_loss
    return metrics


# ✅ 수정된 evaluate_model 함수 - attention scores도 반환
def evaluate_model_with_attention(model, dataloader, device):
    """
    모델 평가 + attention scores 추출
    """
    model.eval()
    all_probs, all_labels, all_preds, all_filenames = [], [], [], []
    attention_scores_dict = {}
    
    with torch.no_grad():
        for features, label, filename in dataloader:
            features = features.to(device)
            label = label.to(device)
            
            # ✅ 수정: return_attention=True 추가!
            results_dict, attention_weights = model(
                h=features, 
                loss_fn=None, 
                label=None,
                return_attention=True  # ← 이게 중요!
            )
            
            logits = results_dict['logits']
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_filenames.extend(filename)
            
            # Attention scores 저장
            wsi_name = filename[0] if isinstance(filename, (list, tuple)) else filename
            wsi_name = wsi_name.replace('.pt', '')
            
            # None 체크
            if attention_weights is None:
                print(f"  ⚠️ Warning: attention_weights is None for {wsi_name}")
                continue
            
            if isinstance(attention_weights, dict):
                if 'A' not in attention_weights:
                    print(f"  ⚠️ Warning: 'A' key not found for {wsi_name}")
                    print(f"      Available keys: {list(attention_weights.keys())}")
                    continue
                
                if attention_weights['A'] is None:
                    print(f"  ⚠️ Warning: attention_weights['A'] is None for {wsi_name}")
                    continue
                
                attn_scores = attention_weights['A'].cpu().numpy().flatten()
                
            elif isinstance(attention_weights, torch.Tensor):
                attn_scores = attention_weights.cpu().numpy().flatten()
            else:
                print(f"  ⚠️ Warning: Unknown attention_weights type: {type(attention_weights)}")
                continue
            
            attention_scores_dict[wsi_name] = {
                'scores': attn_scores.tolist(),
                'n_patches': len(attn_scores),
                'true_label': int(label.cpu().numpy()[0]),
                'predicted_label': int(preds[-1]),
                'pred_prob': float(probs[-1].cpu().numpy())
            }
    
    return all_probs, all_labels, all_preds, all_filenames, attention_scores_dict

# 기존 evaluate_model도 유지 (하위 호환성)
def evaluate_model(model, dataloader, device):
    all_probs, all_labels, all_preds, _, _ = evaluate_model_with_attention(model, dataloader, device)
    return all_probs, all_labels, all_preds


# =========================
# Print Fold Table
# =========================
def print_fold_table(fold_num, train_metrics, val_metrics, test_metrics):
    """각 fold별 결과를 표 형태로 출력"""
    print(f"\n{'='*80}")
    print(f"Fold {fold_num} Results")
    print(f"{'='*80}")
    
    print(f"| {'Set':8s} | {'Accuracy':8s} | {'AUC':8s} | {'Sensitivity':11s} | "
          f"{'Specificity':11s} | {'Precision':9s} | {'NPV':8s} | {'F1-score':8s} |")
    print("|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*13 + "|" + 
          "-"*13 + "|" + "-"*11 + "|" + "-"*10 + "|" + "-"*10 + "|")
    
    print(f"| {'Train':8s} | {train_metrics['accuracy']:8.2f} | "
          f"{train_metrics['auc']:8.2f} | {train_metrics['sensitivity']:11.2f} | "
          f"{train_metrics['specificity']:11.2f} | {train_metrics['ppv']:9.2f} | "
          f"{train_metrics['npv']:8.2f} | {train_metrics['f1']:8.2f} |")
    
    print(f"| {'Val':8s} | {val_metrics['accuracy']:8.2f} | "
          f"{val_metrics['auc']:8.2f} | {val_metrics['sensitivity']:11.2f} | "
          f"{val_metrics['specificity']:11.2f} | {val_metrics['ppv']:9.2f} | "
          f"{val_metrics['npv']:8.2f} | {val_metrics['f1']:8.2f} |")
    
    print(f"| {'Test':8s} | {test_metrics['accuracy']:8.2f} | "
          f"{test_metrics['auc']:8.2f} | {test_metrics['sensitivity']:11.2f} | "
          f"{test_metrics['specificity']:11.2f} | {test_metrics['ppv']:9.2f} | "
          f"{test_metrics['npv']:8.2f} | {test_metrics['f1']:8.2f} |")


# =========================
# Model Save / Load CV Splits
# =========================
def save_model_checkpoint(model, fold_idx, fold_result, save_dir, args, is_best=False):
    save_dir = Path(save_dir) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'fold': fold_idx + 1,
        'model_state_dict': model.state_dict(),
        'accuracy': fold_result['test_metrics']['accuracy'],
        'auc': fold_result['test_metrics']['auc'],
        'config': {
            'lr': args.lr,
            'bag_size': args.bag_size,
            'seed': args.seed,
            'model': 'ABMILGatedBase'
        }
    }

    filename = f"{'best_' if is_best else ''}model_fold{fold_idx+1}_auc{fold_result['test_metrics']['auc']:.4f}.pt"
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)

    print(f"[✓] Model saved: {checkpoint_path}")
    return checkpoint_path


def load_cv_splits_with_paths(cv_split_file, data_root, debug=False):
    with open(cv_split_file, 'r') as f:
        cv_splits = json.load(f)
    for fold_data in cv_splits['folds']:
        for split_name in ['train_wsis', 'val_wsis', 'test_wsis']:
            file_paths = []
            for filename in fold_data[split_name]:
                meta_path = os.path.join(data_root, 'meta', filename)
                nonmeta_path = os.path.join(data_root, 'nonmeta', filename)
                if os.path.exists(meta_path):
                    file_paths.append(meta_path)
                elif os.path.exists(nonmeta_path):
                    file_paths.append(nonmeta_path)
                else:
                    print(f"Warning: {filename} not found")
            fold_data[f'{split_name}_paths'] = file_paths
    return cv_splits


def check_data_leakage(cv_splits):
    """CV split에서 data leakage 체크"""
    print("\n" + "="*80)
    print("Checking Data Leakage")
    print("="*80)
    
    all_folds_ok = True
    
    for fold_data in cv_splits['folds']:
        fold_num = fold_data['fold']
        
        train_files = set([os.path.basename(p) for p in fold_data['train_wsis_paths']])
        val_files = set([os.path.basename(p) for p in fold_data['val_wsis_paths']])
        test_files = set([os.path.basename(p) for p in fold_data['test_wsis_paths']])
        
        print(f"\nFold {fold_num}:")
        print(f"  Train: {len(train_files):3d} files")
        print(f"  Val:   {len(val_files):3d} files")
        print(f"  Test:  {len(test_files):3d} files")
        print(f"  Total: {len(train_files) + len(val_files) + len(test_files):3d} files")
        
        train_val_overlap = train_files & val_files
        train_test_overlap = train_files & test_files
        val_test_overlap = val_files & test_files
        
        has_overlap = False
        
        if train_val_overlap:
            print(f"  ❌ Train-Val overlap: {len(train_val_overlap)} files")
            print(f"      Examples: {list(train_val_overlap)[:3]}")
            has_overlap = True
            all_folds_ok = False
        
        if train_test_overlap:
            print(f"  ❌ Train-Test overlap: {len(train_test_overlap)} files")
            print(f"      Examples: {list(train_test_overlap)[:3]}")
            has_overlap = True
            all_folds_ok = False
        
        if val_test_overlap:
            print(f"  ❌ Val-Test overlap: {len(val_test_overlap)} files")
            print(f"      Examples: {list(val_test_overlap)[:3]}")
            has_overlap = True
            all_folds_ok = False
        
        if not has_overlap:
            print(f"  ✅ No overlap detected")
    
    print("\n" + "="*80)
    if all_folds_ok:
        print("✅ All folds passed leakage check")
    else:
        print("❌ DATA LEAKAGE DETECTED! CV splits need to be regenerated!")
        print("   Training results cannot be trusted.")
    print("="*80 + "\n")
    
    return all_folds_ok


def check_label_distribution(cv_splits, data_root, bag_size):
    """각 fold의 label 분포 확인 (JSON 데이터 직접 사용)"""
    print("\n" + "="*80)
    print("Label Distribution Analysis")
    print("="*80)
    
    for fold_data in cv_splits['folds']:
        fold_num = fold_data['fold']
        
        print(f"\nFold {fold_num}:")
        
        # JSON에 이미 pos/neg count가 있음!
        print(f"  Train: Pos={fold_data['train_pos_count']:3d} ({fold_data['train_pos_count']/fold_data['train_count']*100:5.1f}%), "
              f"Neg={fold_data['train_neg_count']:3d} ({fold_data['train_neg_count']/fold_data['train_count']*100:5.1f}%), "
              f"Total={fold_data['train_count']:3d}")
        print(f"  Val  : Pos={fold_data['val_pos_count']:3d} ({fold_data['val_pos_count']/fold_data['val_count']*100:5.1f}%), "
              f"Neg={fold_data['val_neg_count']:3d} ({fold_data['val_neg_count']/fold_data['val_count']*100:5.1f}%), "
              f"Total={fold_data['val_count']:3d}")
        print(f"  Test : Pos={fold_data['test_pos_count']:3d} ({fold_data['test_pos_count']/fold_data['test_count']*100:5.1f}%), "
              f"Neg={fold_data['test_neg_count']:3d} ({fold_data['test_neg_count']/fold_data['test_count']*100:5.1f}%), "
              f"Total={fold_data['test_count']:3d}")
    
    print("\n" + "="*80 + "\n")


# =========================
# Run K-Fold CV (✅ 수정됨)
# =========================
def run_k_fold_cv(cv_splits, args, device):
    all_fold_results = []
    all_predictions, all_true_labels = [], []
    saved_model_paths = []
    config = ABMILGatedBaseConfig()

    for fold_data in cv_splits['folds']:
        fold_idx = fold_data['fold'] - 1
        print(f"\n{'='*80}")
        print(f"Fold {fold_data['fold']}/{len(cv_splits['folds'])}")
        print(f"{'='*80}")

        train_dataset = ThyroidWSIDataset(fold_data['train_wsis_paths'], bag_size=args.bag_size, use_variance=False)
        val_dataset = ThyroidWSIDataset(fold_data['val_wsis_paths'], bag_size=args.bag_size, use_variance=False)
        test_dataset = ThyroidWSIDataset(fold_data['test_wsis_paths'], bag_size=args.bag_size, use_variance=False)

        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=1, 
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=1, 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        model = ABMILModel(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        early_stopping = EarlyStopping(patience=8, min_delta=0.001)

        best_train_metrics, best_val_metrics = {}, {}
        best_epoch = 0
        
        # History 기록용
        history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }

        print(f"\n{'─'*80}")
        print(f"Starting training for Fold {fold_data['fold']}")
        print(f"{'─'*80}\n")

        for epoch in range(args.epochs):
            train_metrics = run_one_epoch(model, train_loader, device, optimizer, train=True)
            val_metrics = run_one_epoch(model, val_loader, device, train=False)
            scheduler.step(val_metrics["loss"])
            
            # History 저장
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_auc'].append(train_metrics['auc'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_auc'].append(val_metrics['auc'])
            
            print(f"Epoch [{epoch+1:3d}/{args.epochs}] | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train AUC: {train_metrics['auc']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            if val_metrics["auc"] > best_val_metrics.get("auc", 0):
                best_val_metrics = val_metrics
                best_train_metrics = train_metrics
                best_epoch = epoch + 1
                print(f"  ↑ Best model updated (Val AUC: {val_metrics['auc']:.4f})")

            if early_stopping(val_metrics["auc"], model):
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                print(f"  Best epoch was {best_epoch} with Val AUC: {best_val_metrics['auc']:.4f}")
                early_stopping.restore_best(model)
                break

        # ✅ Test evaluation with attention scores
        print(f"\n{'─'*80}")
        print(f"Testing Fold {fold_data['fold']} on best model (Epoch {best_epoch})")
        print(f"{'─'*80}")
        
        # Attention scores 포함하여 평가
        test_probs, test_labels, test_preds, test_filenames, attention_scores_dict = \
            evaluate_model_with_attention(model, test_loader, device)
        
        test_metrics = compute_metrics_with_confusion(test_labels, test_preds, test_probs)
        
        # ROC curve와 Precision-Recall curve 데이터 계산
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        precision, recall, _ = precision_recall_curve(test_labels, test_probs)

        print(f"\n📊 Test Results:")
        print(f"  AUC:         {test_metrics['auc']:.4f}")
        print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {test_metrics['specificity']:.4f}")
        print(f"  PPV:         {test_metrics['ppv']:.4f}")
        print(f"  NPV:         {test_metrics['npv']:.4f}")
        print(f"  F1 Score:    {test_metrics['f1']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {test_metrics['tp']:3d}  |  FN: {test_metrics['fn']:3d}")
        print(f"    FP: {test_metrics['fp']:3d}  |  TN: {test_metrics['tn']:3d}")
        print(f"\n  Attention Scores: Extracted for {len(attention_scores_dict)} WSIs")

        # Fold별 표 출력
        print_fold_table(fold_data['fold'], best_train_metrics, best_val_metrics, test_metrics)

        # ✅ fold_result에 attention_scores_dict 추가
        fold_result = {
            "fold": fold_data['fold'],
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "best_epoch": best_epoch,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "history": history,
            "test_fpr": fpr.tolist(),
            "test_tpr": tpr.tolist(),
            "test_precision": precision.tolist(),
            "test_recall": recall.tolist(),
            "test_attention_scores": attention_scores_dict  # ✅ 추가!
        }

        all_fold_results.append(fold_result)
        all_predictions.extend(test_probs)
        all_true_labels.extend(test_labels)

        if args.save_model:
            if args.save_best_only:
                if fold_idx == 0 or test_metrics['auc'] > max(r['test_metrics']['auc'] for r in all_fold_results[:-1]):
                    path = save_model_checkpoint(model, fold_idx, fold_result, args.model_save_dir, args, is_best=True)
                    saved_model_paths.append(path)
                    print(f"✓ Best model saved")
            else:
                path = save_model_checkpoint(model, fold_idx, fold_result, args.model_save_dir, args)
                saved_model_paths.append(path)
                print(f"✓ Model checkpoint saved")

    # ==========================================
    # 전체 Summary 계산
    # ==========================================
    print(f"\n\n{'='*80}")
    print(f"WSI Instance Results (Mean ± Std Across All Folds)")
    print(f"{'='*80}\n")
    
    metrics_order = ['accuracy', 'auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1']
    metric_names = {
        'accuracy': 'Accuracy',
        'auc': 'AUC',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'ppv': 'PPV',
        'npv': 'NPV',
        'f1': 'F1-score'
    }
    
    # Summary 통계 계산 (JSON 저장용)
    summary_stats = {}
    for set_name, metric_key in [('train', 'best_train_metrics'), 
                                   ('val', 'best_val_metrics'), 
                                   ('test', 'test_metrics')]:
        summary_stats[set_name] = {}
        for metric in metrics_order:
            values = [r[metric_key][metric] for r in all_fold_results]
            summary_stats[set_name][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': [float(v) for v in values]
            }
    
    # Summary 표 출력
    header = f"| {'Set':8s} |"
    for metric in metrics_order:
        header += f" {metric_names[metric]:17s} |"
    print(header)
    print("|" + "-"*10 + "|" + "-"*19*len(metrics_order) + "|")
    
    for set_name, set_label in [('train', 'Train'), ('val', 'Val'), ('test', 'Test')]:
        row = f"| **{set_label}** |"
        for metric in metrics_order:
            mean_val = summary_stats[set_name][metric]['mean']
            std_val = summary_stats[set_name][metric]['std']
            row += f" **{mean_val:.3f} ± {std_val:.3f}** |"
        print(row)
    
    # Total Confusion Matrix
    print(f"\n{'─'*80}")
    print(f"Total Confusion Matrix (Test):")
    print(f"{'─'*80}")
    total_tp = sum(r['test_metrics']['tp'] for r in all_fold_results)
    total_tn = sum(r['test_metrics']['tn'] for r in all_fold_results)
    total_fp = sum(r['test_metrics']['fp'] for r in all_fold_results)
    total_fn = sum(r['test_metrics']['fn'] for r in all_fold_results)
    
    print(f"    TP: {total_tp:4d}  |  FN: {total_fn:4d}")
    print(f"    FP: {total_fp:4d}  |  TN: {total_tn:4d}")

    return all_fold_results, all_predictions, all_true_labels, saved_model_paths, summary_stats


# =========================
# Utility
# =========================
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {str(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_save_dir', type=str, required=True)
    parser.add_argument('--cv_split_file', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bag_size', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_best_only', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not Path(args.cv_split_file).exists():
        raise ValueError(f"CV split file not found: {args.cv_split_file}")
    
    cv_splits = load_cv_splits_with_paths(args.cv_split_file, args.data_root, debug=args.debug)

    leakage_check_passed = check_data_leakage(cv_splits)
    check_label_distribution(cv_splits, args.data_root, args.bag_size)
    
    if not leakage_check_passed:
        print("\n⚠️  WARNING: Data leakage detected!")
        response = input("Continue training anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Training aborted.")
            return
    
    # Cross-validation 실행
    fold_results, predictions, true_labels, model_paths, summary_stats = run_k_fold_cv(cv_splits, args, device)

    # ==========================================
    # Visualization 생성
    # ==========================================
    print(f"\n{'='*80}")
    print(f"Generating Visualization Plots")
    print(f"{'='*80}\n")
    
    viz_dir = Path(args.model_save_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    plot_roc_curves(fold_results, viz_dir)
    plot_precision_recall_curves(fold_results, viz_dir)
    plot_training_curves(fold_results, viz_dir)
    plot_metric_comparison(fold_results, viz_dir)
    plot_confusion_matrices(fold_results, viz_dir)

    # ==========================================
    # JSON 결과 구성
    # ==========================================
    results = {
        "summary_statistics": summary_stats,
        "folds": fold_results,  # ✅ 이제 test_attention_scores 포함!
    }
    
    # Final aggregated results
    if len(set(true_labels)) > 1:
        eval_results = comprehensive_evaluation(true_labels, predictions)
        results["final_aggregated"] = eval_results
        
        print(f"\n{'='*80}")
        print(f"Final Aggregated Results (All Folds Combined)")
        print(f"{'='*80}\n")
        
        for key, value in eval_results.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                print(f"{key:20s}: {value:.4f}")

    # JSON 저장
    results_path = Path(args.model_save_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\n[✓] Results saved: {results_path}")
    print(f"    ✅ Attention scores included for all test WSIs")

    if args.save_model and model_paths:
        print(f"[✓] Saved {len(model_paths)} model checkpoints")

    print(f"\n[✓] All visualizations saved in: {viz_dir}")
    print("\n[✓] Training completed!")


if __name__ == "__main__":
    main()