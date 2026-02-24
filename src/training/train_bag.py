# -*- coding: utf-8 -*-
"""
K-Fold Cross-Validation Training Script with Multi-Threshold Evaluation

threshold 별 지표 출력 버전 (v0.5.0)
threshold_comparison_summary.json # 전체 비교 요약


"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import warnings
import sys
from multiprocessing import cpu_count

# GPU 설정 강제 (코드 최상단에서 설정)
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    print(f"⚠️  CUDA_VISIBLE_DEVICES not set, forcing GPU 2")
else:
    print(f"✓ CUDA_VISIBLE_DEVICES already set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

warnings.filterwarnings('ignore')

# =========================
# 경로 설정
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)
sys.path.append(os.path.join(src_dir, 'models'))

# evaluation 디렉토리 경로 추가
evaluation_dir = os.path.join(src_dir, 'evaluation')
sys.path.insert(0, evaluation_dir)

# =========================
# Models & Utils import
# =========================
from abmil import ABMILModel, ABMILGatedBaseConfig
from utils.datasets import ThyroidWSIDataset, set_seed
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

# =========================
# Evaluation 모듈 import
# =========================
from metric import (
    compute_metrics_with_confusion,
    compute_roc_curve_data,
    compute_precision_recall_curve_data,
    compute_summary_statistics,
    print_fold_table,
    print_summary_statistics,
    print_confusion_matrix_summary,
    generate_all_plots
)


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
        self.best_epoch = 0

    def __call__(self, score, model=None, epoch=None):
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

    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# =========================
# Model Checkpoint
# =========================
def save_model_checkpoint(model, fold_idx, fold_result, save_dir, args, is_best=False):
    save_dir = Path(save_dir) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'fold': fold_idx + 1,
        'model_state_dict': model.state_dict(),
        'accuracy': fold_result['test_metrics_optimal']['accuracy'],
        'auc': fold_result['test_metrics_optimal']['auc'],
        'optimal_threshold': fold_result['optimal_threshold'],
        'config': {
            'lr': args.lr,
            'bag_size': args.bag_size,
            'seed': args.seed,
            'model': 'ABMILGatedBase'
        }
    }

    filename = f"{'best_' if is_best else ''}model_fold{fold_idx+1}_auc{fold_result['test_metrics_optimal']['auc']:.4f}.pt"
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)

    print(f"[✓] Model saved: {checkpoint_path}")
    return checkpoint_path


# =========================
# CV Split Loading
# =========================
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
    """CV split data leakage 체크"""
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
            has_overlap = True
            all_folds_ok = False
        
        if train_test_overlap:
            print(f"  ❌ Train-Test overlap: {len(train_test_overlap)} files")
            has_overlap = True
            all_folds_ok = False
        
        if val_test_overlap:
            print(f"  ❌ Val-Test overlap: {len(val_test_overlap)} files")
            has_overlap = True
            all_folds_ok = False
        
        if not has_overlap:
            print(f"  ✅ No overlap detected")
    
    print("\n" + "="*80)
    if all_folds_ok:
        print("✅ All folds passed leakage check")
    else:
        print("❌ DATA LEAKAGE DETECTED!")
    print("="*80 + "\n")
    
    return all_folds_ok


def check_label_distribution(cv_splits, data_root, bag_size):
    """각 fold의 label 분포 확인"""
    print("\n" + "="*80)
    print("Label Distribution Analysis")
    print("="*80)
    
    for fold_data in cv_splits['folds']:
        fold_num = fold_data['fold']
        print(f"\nFold {fold_num}:")
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
# Training Loop (threshold=0.5 고정)
# =========================
def run_one_epoch(model, dataloader, device, optimizer=None, train=False, threshold=0.5):
    """Training/Validation with fixed threshold=0.5"""
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
            preds = (probs >= threshold).long()

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


# ============================================================
# Optimal Threshold Selection from Validation Set
# ============================================================
def find_optimal_threshold_on_validation(model, dataloader, device, thresholds, metric='f1'):
    """
    Validation set으로 최적 threshold 찾기

    Args:
        model: 학습된 모델
        dataloader: validation dataloader
        device: torch device
        thresholds: 탐색할 threshold 목록
        metric: 최적화할 metric ('f1', 'youden', 'accuracy')

    Returns:
        best_threshold: 최적 threshold
        best_score: 최적 score
        all_results: 모든 threshold 결과
    """
    model.eval()
    all_probs, all_labels = [], []

    # 1단계: 확률값 수집
    with torch.no_grad():
        for features, label, filename in dataloader:
            features = features.to(device)
            label = label.to(device)

            results_dict = model(h=features, loss_fn=None, label=None,
                                return_attention=False, return_extra=True)

            logits = results_dict['logits']
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # 2단계: 각 threshold로 평가
    threshold_results = {}
    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        preds = (np.array(all_probs) >= threshold).astype(int)

        if len(set(all_labels)) > 1:
            metrics = compute_metrics_with_confusion(all_labels, preds, all_probs)

            # Metric 선택
            if metric == 'f1':
                score = metrics['f1']
            elif metric == 'youden':
                # Youden's Index = Sensitivity + Specificity - 1
                score = metrics['sensitivity'] + metrics['specificity'] - 1
            elif metric == 'accuracy':
                score = metrics['accuracy']
            elif metric == 'balanced_f1':
                # F1과 NPV의 조화평균 (둘 다 높은 threshold 선택)
                f1 = metrics['f1']
                npv = metrics['npv']
                if f1 + npv > 0:
                    score = 2 * (f1 * npv) / (f1 + npv)  # Harmonic mean
                else:
                    score = 0.0
            elif metric == 'gmean':
                # G-mean = sqrt(Sensitivity × Specificity)
                # 불균형 데이터에서 양/음성 균형 맞추기
                score = np.sqrt(metrics['sensitivity'] * metrics['specificity'])
            elif metric == 'balanced_acc':
                # Balanced Accuracy = (Sensitivity + Specificity) / 2
                score = (metrics['sensitivity'] + metrics['specificity']) / 2
            elif metric == 'f1_npv_weighted':
                # F1과 NPV의 가중평균 (NPV에 더 높은 가중치)
                # NPV가 낮은 경우를 개선하기 위해 NPV에 60% 가중치
                score = 0.4 * metrics['f1'] + 0.6 * metrics['npv']
            else:
                score = metrics['f1']  # default

            threshold_results[threshold] = {
                'score': score,
                'metrics': metrics
            }

            if score > best_score:
                best_score = score
                best_threshold = threshold

    print(f"\n📊 Validation Threshold Search (optimizing {metric.upper()}):")
    print(f"{'─'*100}")
    print(f"{'Thresh':<8} {'Score':<10} {'F1':<8} {'NPV':<8} {'Sens':<8} {'Spec':<8} {'PPV':<8} {'Acc':<8} {'AUC':<8}")
    print(f"{'─'*100}")

    for threshold in thresholds:
        result = threshold_results[threshold]
        m = result['metrics']
        marker = " 👈 BEST" if threshold == best_threshold else ""
        print(f"{threshold:<8.2f} {result['score']:<10.4f} {m['f1']:<8.4f} {m['npv']:<8.4f} "
              f"{m['sensitivity']:<8.4f} {m['specificity']:<8.4f} {m['ppv']:<8.4f} "
              f"{m['accuracy']:<8.4f} {m['auc']:<8.4f}{marker}")

    print(f"{'─'*100}")
    print(f"✅ Best Threshold: {best_threshold:.2f} ({metric.upper()}={best_score:.4f})\n")

    return best_threshold, best_score, threshold_results


# ============================================================
# Single Threshold Test Evaluation
# ============================================================
def evaluate_model_with_single_threshold(model, dataloader, device, threshold):
    """
    Test evaluation with a single threshold
    Returns: metrics, probs, labels, filenames
    """
    model.eval()
    all_probs, all_labels, all_filenames = [], [], []

    # 확률값 수집
    with torch.no_grad():
        for features, label, filename in dataloader:
            features = features.to(device)
            label = label.to(device)

            results_dict = model(h=features, loss_fn=None, label=None,
                                return_attention=False, return_extra=True)

            logits = results_dict['logits']
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_filenames.extend(filename)

    # Threshold로 예측
    preds = (np.array(all_probs) >= threshold).astype(int)

    if len(set(all_labels)) > 1:
        metrics = compute_metrics_with_confusion(all_labels, preds, all_probs)
    else:
        metrics = {
            "accuracy": 0.0, "auc": 0.5,
            "sensitivity": 0.0, "specificity": 0.0,
            "ppv": 0.0, "npv": 0.0,
            "f1": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0
        }

    return metrics, all_probs, all_labels, all_filenames


# ============================================================
# evaluate_model_with_attention
# ============================================================
def evaluate_model_with_attention(model, dataloader, device, threshold=0.5):
    """Single threshold evaluation with attention scores"""
    model.eval()
    all_probs, all_labels, all_preds, all_filenames = [], [], [], []
    attention_scores_dict = {}
    
    with torch.no_grad():
        for features, label, filename in dataloader:
            features = features.to(device)
            label = label.to(device)
            
            # ✅ FIX: 반환값 1개로 수정
            results_dict = model(h=features, loss_fn=None, label=None, 
                               return_attention=True, return_extra=True)
            
            logits = results_dict['logits']
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_filenames.extend(filename)
            
            # Attention scores 저장
            if 'attention' in results_dict and results_dict['attention'] is not None:
                attention_weights = results_dict['attention']
                if attention_weights.dim() == 3:
                    attn_raw = attention_weights[0, 0, :]
                    attn_scores = F.softmax(attn_raw, dim=0).cpu().numpy()
                elif attention_weights.dim() == 2:
                    attn_raw = attention_weights[0, :]
                    attn_scores = F.softmax(attn_raw, dim=0).cpu().numpy()
                else:
                    continue
                
                wsi_name = Path(filename[0]).stem
                attention_scores_dict[wsi_name] = {
                    'scores': attn_scores.tolist(),
                    'n_patches': len(attn_scores),
                    'predicted_label': int(preds.cpu().numpy()[0]),
                    'pred_prob': float(probs.cpu().numpy()[0]),
                }
    
    return all_probs, all_labels, all_preds, all_filenames, attention_scores_dict


# ============================================================
# Full WSI Attention
# ============================================================
def evaluate_full_wsi_for_visualization(
    model: nn.Module,
    test_wsi_list: List[str],
    device: torch.device,
    meta_npy_dir: str,
    nonmeta_npy_dir: str,
):
    """Extract full WSI attention scores from ORIGINAL embeddings for visualization."""
    model.eval()
    attention_scores_dict = {}
    
    print(f"\n{'─'*80}")
    print(f"Extracting Full WSI Attention Scores (from ORIGINAL FULL embeddings)")
    print(f"{'─'*80}")
    print(f"📂 Meta NPY dir: {meta_npy_dir}")
    print(f"📂 Nonmeta NPY dir: {nonmeta_npy_dir}")
    
    successful = 0
    failed = 0
    
    with torch.no_grad():
        for wsi_file in test_wsi_list:
            wsi_name = Path(wsi_file).stem
            
            meta_path = Path(meta_npy_dir) / f"{wsi_name}.npy"
            nonmeta_path = Path(nonmeta_npy_dir) / f"{wsi_name}.npy"
            
            if meta_path.exists():
                full_features = np.load(meta_path)
                data_type = 'meta'
                npy_path = meta_path
            elif nonmeta_path.exists():
                full_features = np.load(nonmeta_path)
                data_type = 'nonmeta'
                npy_path = nonmeta_path
            else:
                print(f"⚠️ NPY not found: {wsi_name}")
                failed += 1
                continue
            
            n_patches = full_features.shape[0]
            if n_patches < 5000:
                print(f"⚠️ WARNING: {wsi_name} has only {n_patches} patches!")
            
            features_tensor = torch.from_numpy(full_features).float().unsqueeze(0).to(device)
            
            try:
                results_dict = model(
                    h=features_tensor,
                    loss_fn=None,
                    label=None,
                    return_attention=True,
                    return_extra=True
                )
                
                if 'attention' in results_dict and results_dict['attention'] is not None:
                    attention_weights = results_dict['attention']
                    
                    if attention_weights.dim() == 3:
                        attn_raw = attention_weights[0, 0, :]
                        attn_scores = F.softmax(attn_raw, dim=0).cpu().numpy()
                    elif attention_weights.dim() == 2:
                        attn_raw = attention_weights[0, :]
                        attn_scores = F.softmax(attn_raw, dim=0).cpu().numpy()
                    else:
                        print(f"⚠️ Unexpected attention shape for {wsi_name}")
                        failed += 1
                        continue
                else:
                    print(f"⚠️ No attention in results for {wsi_name}")
                    failed += 1
                    continue
                
                logits = results_dict['logits']
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                attention_scores_dict[wsi_name] = {
                    'scores': attn_scores.tolist(),
                    'n_patches': len(attn_scores),
                    'predicted_label': int(preds.cpu().numpy()[0]),
                    'pred_prob': float(probs.cpu().numpy()[0]),
                    'data_type': data_type,
                }
                
                print(f"✓ {wsi_name} ({data_type}): {len(attn_scores)} patches")
                successful += 1
                
            except Exception as e:
                print(f"✗ Failed to extract attention for {wsi_name}: {e}")
                failed += 1
                continue
    
    print(f"{'─'*80}")
    print(f"✓ Successfully extracted: {successful}/{len(test_wsi_list)} WSIs")
    if failed > 0:
        print(f"✗ Failed: {failed}/{len(test_wsi_list)} WSIs")
    print(f"{'─'*80}\n")
    
    return attention_scores_dict


# =========================
# K-Fold CV with Multi-Threshold
# =========================
def run_k_fold_cv(cv_splits, args, device):
    all_fold_results = []
    all_predictions, all_true_labels = [], []
    saved_model_paths = []
    config = ABMILGatedBaseConfig()

    for fold_data in cv_splits['folds']:
        fold_idx = fold_data['fold'] - 1

        if hasattr(args, 'test_fold') and args.test_fold is not None and fold_data['fold'] != args.test_fold:
            continue

        print(f"\n{'='*80}")
        print(f"Fold {fold_data['fold']}/{len(cv_splits['folds'])}")
        print(f"Optimal threshold will be selected from validation set")
        if hasattr(args, 'test_fold') and args.test_fold is not None:
            print(f"⚠️  TEST MODE: Running only Fold {args.test_fold}")
        print(f"{'='*80}")

        # DataLoader
        train_dataset = ThyroidWSIDataset(fold_data['train_wsis_paths'], bag_size=args.bag_size, use_variance=False)
        val_dataset = ThyroidWSIDataset(fold_data['val_wsis_paths'], bag_size=args.bag_size, use_variance=False)
        test_dataset = ThyroidWSIDataset(fold_data['test_wsis_paths'], bag_size=None, use_variance=False)  # ✅ Test는 전체 npy 사용

        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
        print(f"⚠️  Test set uses FULL embeddings (no bag_size limit)")

        # Calculate optimal num_workers (40% of CPUs for training)
        n_cpus = cpu_count()
        train_workers = max(8, int(n_cpus * 0.4))  # 40% for training, minimum 8
        val_workers = max(4, int(n_cpus * 0.2))    # 20% for val/test, minimum 4

        print(f"DataLoader workers: train={train_workers}, val/test={val_workers} (Total CPUs: {n_cpus})")

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=train_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=val_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=val_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

        # Model
        model = ABMILModel(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        early_stopping = EarlyStopping(patience=8, min_delta=0.001)

        best_train_metrics, best_val_metrics = {}, {}
        history = {'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_f1': [],
                   'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': []}

        print(f"\n{'─'*80}")
        print(f"Starting training for Fold {fold_data['fold']}")
        print(f"{'─'*80}\n")

        # Training (threshold=0.5 고정)
        for epoch in range(args.epochs):
            train_metrics = run_one_epoch(model, train_loader, device, optimizer, train=True, threshold=0.5)
            val_metrics = run_one_epoch(model, val_loader, device, train=False, threshold=0.5)
            scheduler.step(val_metrics["loss"])
            
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['train_auc'].append(train_metrics['auc'])
            history['train_f1'].append(train_metrics['f1'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_auc'].append(val_metrics['auc'])
            history['val_f1'].append(val_metrics['f1'])
            
            print(f"Epoch [{epoch+1:3d}/{args.epochs}] | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train AUC: {train_metrics['auc']:.4f} | "
                  f"Train F1: {train_metrics['f1']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            if val_metrics["auc"] > best_val_metrics.get("auc", 0):
                best_val_metrics = val_metrics
                best_train_metrics = train_metrics
                print(f"  ↑ Best model updated (Val AUC: {val_metrics['auc']:.4f}, Val F1: {val_metrics['f1']:.4f})")

            if early_stopping(val_metrics["auc"], model, epoch+1):
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                print(f"  Best epoch was {early_stopping.best_epoch} with Val AUC: {best_val_metrics['auc']:.4f}")
                early_stopping.restore_best(model)
                break

        # ============================================================
        # 1️⃣ Validation Set으로 최적 Threshold 찾기
        # ============================================================
        print(f"\n{'='*80}")
        print(f"Step 1: Finding Optimal Threshold on Validation Set")
        print(f"{'='*80}")

        # Threshold 후보 (더 세밀하게 탐색)
        threshold_candidates = np.arange(0.05, 0.95, 0.05).tolist()

        optimal_threshold, optimal_score, _ = \
            find_optimal_threshold_on_validation(
                model, val_loader, device,
                thresholds=threshold_candidates,
                metric='youden'  # Youden's Index로 최적화 (sens + spec - 1)
            )

        # ============================================================
        # 2️⃣ Test Set 평가 (최적 threshold만)
        # ============================================================
        print(f"\n{'='*80}")
        print(f"Step 2: Testing with Optimal Threshold ({optimal_threshold:.2f})")
        print(f"{'='*80}")

        test_metrics_optimal, test_probs, test_labels, test_filenames = \
            evaluate_model_with_single_threshold(model, test_loader, device, optimal_threshold)

        # ROC/PR curve (threshold 무관)
        fpr, tpr, _ = compute_roc_curve_data(test_labels, test_probs)
        precision, recall, _ = compute_precision_recall_curve_data(test_labels, test_probs)

        # 결과 출력
        print(f"\n🎯 Test Results with Optimal Threshold ({optimal_threshold:.2f}):")
        print(f"{'─'*80}")
        print(f"{'Metric':<15} {'Value':<10}")
        print(f"{'─'*80}")
        print(f"{'AUC':<15} {test_metrics_optimal['auc']:<10.4f}")
        print(f"{'Accuracy':<15} {test_metrics_optimal['accuracy']:<10.4f}")
        print(f"{'F1-Score':<15} {test_metrics_optimal['f1']:<10.4f}")
        print(f"{'Sensitivity':<15} {test_metrics_optimal['sensitivity']:<10.4f}")
        print(f"{'Specificity':<15} {test_metrics_optimal['specificity']:<10.4f}")
        print(f"{'PPV':<15} {test_metrics_optimal['ppv']:<10.4f}")
        print(f"{'NPV':<15} {test_metrics_optimal['npv']:<10.4f}")
        print(f"{'─'*80}")
        print(f"{'TP':<15} {test_metrics_optimal['tp']:<10}")
        print(f"{'TN':<15} {test_metrics_optimal['tn']:<10}")
        print(f"{'FP':<15} {test_metrics_optimal['fp']:<10}")
        print(f"{'FN':<15} {test_metrics_optimal['fn']:<10}")
        print(f"{'─'*80}\n")

        # ============================================================
        # 3️⃣ Full WSI Attention Extraction (from test_loader full embeddings)
        print(f"\n{'─'*80}")
        print(f"Extracting FULL WSI Attention Scores")
        print(f"{'─'*80}")
        
        meta_npy_dir = "/path/to/meta/npy"
        nonmeta_npy_dir = "/path/to/nonmeta/npy"
        
        test_wsi_files = [os.path.basename(p) for p in fold_data['test_wsis_paths']]
        
        full_attention_scores = evaluate_full_wsi_for_visualization(
            model, test_wsi_files, device, meta_npy_dir, nonmeta_npy_dir
        )
        
        print(f"✅ Extracted FULL attention scores for {len(full_attention_scores)} WSIs")
        
        # True labels 매핑
        wsi_to_label = {}
        for wsi_path, label in zip(fold_data['test_wsis_paths'], test_labels):
            wsi_name = Path(wsi_path).stem
            wsi_to_label[wsi_name] = int(label)
        
        for wsi_name in full_attention_scores:
            if wsi_name in wsi_to_label:
                full_attention_scores[wsi_name]['true_label'] = wsi_to_label[wsi_name]

        # Fold 결과 출력
        print_fold_table(fold_data['fold'], best_train_metrics, best_val_metrics, test_metrics_optimal)

        # Fold 결과 저장
        fold_result = {
            "fold": fold_data['fold'],
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "best_epoch": early_stopping.best_epoch,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
            "test_metrics_optimal": test_metrics_optimal,
            "optimal_threshold": optimal_threshold,
            "optimal_val_f1": optimal_score,
            "history": history,
            "test_fpr": fpr.tolist(),
            "test_tpr": tpr.tolist(),
            "test_precision": precision.tolist(),
            "test_recall": recall.tolist(),
            "full_attention_scores": full_attention_scores  # ✅ Test set은 전체 npy 사용, attention은 full_attention_scores에 저장
        }

        all_fold_results.append(fold_result)
        all_predictions.extend(test_probs)
        all_true_labels.extend(test_labels)

        # Model saving
        if args.save_model:
            if args.save_best_only:
                if fold_idx == 0 or test_metrics_optimal['auc'] > max(r['test_metrics_optimal']['auc'] for r in all_fold_results[:-1]):
                    path = save_model_checkpoint(model, fold_idx, fold_result, args.model_save_dir, args, is_best=True)
                    saved_model_paths.append(path)
                    print(f"✓ Best model saved")
            else:
                path = save_model_checkpoint(model, fold_idx, fold_result, args.model_save_dir, args)
                saved_model_paths.append(path)
                print(f"✓ Model checkpoint saved")

    # Summary 계산 (Optimal threshold만)
    summary_stats_optimal = compute_summary_statistics(
        [{**r, 'test_metrics': r['test_metrics_optimal']} for r in all_fold_results]
    )

    return all_fold_results, all_predictions, all_true_labels, saved_model_paths, summary_stats_optimal


def convert_numpy(obj):
    """NumPy 타입을 Python 기본 타입으로 변환"""
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
    

def get_stat_value(stats_dict, primary_key, fallback_key='', default=0.0):
    """안전하게 통계 값을 가져오는 헬퍼 함수
    
    Args:
        stats_dict: compute_summary_statistics()가 반환한 딕셔너리
        primary_key: 우선 시도할 키 (예: 'mean_auc')
        fallback_key: primary_key가 없을 때 시도할 키 (예: 'mean_test_auc')
        default: 둘 다 없을 때 반환할 기본값
    """
    if primary_key in stats_dict:
        return stats_dict[primary_key]
    if fallback_key and fallback_key in stats_dict:
        return stats_dict[fallback_key]
    return default


# =========================
# Training Pipeline
# =========================
def run_training(args):
    """
    Run the complete training pipeline.

    Args:
        args: Namespace object with training configuration

    Returns:
        dict: Training results including paths and metadata
    """
    set_seed(args.seed)

    # GPU 설정 - CUDA_VISIBLE_DEVICES에 따라 첫 번째 GPU 사용
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES로 재매핑된 첫 번째 GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    if not Path(args.cv_split_file).exists():
        raise ValueError(f"CV split file not found: {args.cv_split_file}")

    # Data validation
    cv_splits = load_cv_splits_with_paths(args.cv_split_file, args.data_root, debug=args.debug)
    leakage_check_passed = check_data_leakage(cv_splits)
    check_label_distribution(cv_splits, args.data_root, args.bag_size)
    
    if not leakage_check_passed:
        print("\n⚠️  WARNING: Data leakage detected!")
        response = input("Continue training anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Training aborted.")
            return
        

    # Training
    fold_results, predictions, true_labels, model_paths, summary_stats_optimal = \
        run_k_fold_cv(cv_splits, args, device)

    # Summary 출력 (Optimal Threshold)
    print(f"\n{'='*80}")
    print(f"Summary Statistics - OPTIMAL THRESHOLD (selected from Validation)")
    print(f"{'='*80}\n")
    print_summary_statistics(summary_stats_optimal)
    print_confusion_matrix_summary([{**r, 'test_metrics': r['test_metrics_optimal']} for r in fold_results])

    # Fold별 optimal threshold 출력
    print(f"\n{'='*80}")
    print(f"Optimal Thresholds per Fold")
    print(f"{'='*80}")
    print(f"{'Fold':<8} {'Optimal Thresh':<16} {'Val F1':<12} {'Test F1':<12} {'Test AUC':<12}")
    print(f"{'─'*80}")
    for r in fold_results:
        print(f"{r['fold']:<8} {r['optimal_threshold']:<16.2f} "
              f"{r['optimal_val_f1']:<12.4f} "
              f"{r['test_metrics_optimal']['f1']:<12.4f} "
              f"{r['test_metrics_optimal']['auc']:<12.4f}")
    mean_optimal_thresh = np.mean([r['optimal_threshold'] for r in fold_results])
    print(f"{'─'*80}")
    print(f"Mean Optimal Threshold: {mean_optimal_thresh:.2f}")
    print(f"{'='*80}\n")

    # Results 저장
    print(f"\n{'='*80}")
    print(f"Saving Results")
    print(f"{'='*80}")

    # 1. Optimal Threshold Summary (각 fold별 최적 threshold)
    cv_summary_optimal = {
        "threshold": "optimal (per-fold from validation)",
        "mean_optimal_threshold": float(np.mean([r['optimal_threshold'] for r in fold_results])),
        "summary_statistics": summary_stats_optimal,
        "folds": []
    }

    for fold_result in fold_results:
        fold_summary = {
            "fold": fold_result['fold'],
            "train_size": fold_result['train_size'],
            "val_size": fold_result['val_size'],
            "test_size": fold_result['test_size'],
            "best_epoch": fold_result['best_epoch'],
            "optimal_threshold": fold_result['optimal_threshold'],
            "optimal_val_f1": fold_result['optimal_val_f1'],
            "best_train_metrics": fold_result['best_train_metrics'],
            "best_val_metrics": fold_result['best_val_metrics'],
            "test_metrics": fold_result['test_metrics_optimal'],
            "history": fold_result['history'],
            "test_fpr": fold_result['test_fpr'],
            "test_tpr": fold_result['test_tpr'],
            "test_precision": fold_result['test_precision'],
            "test_recall": fold_result['test_recall']
        }
        cv_summary_optimal["folds"].append(fold_summary)

    cv_summary_optimal_path = Path(args.model_save_dir) / "results_cv_summary_optimal.json"
    with open(cv_summary_optimal_path, "w") as f:
        json.dump(convert_numpy(cv_summary_optimal), f, indent=2)

    print(f"[✓] Optimal CV Summary saved: {cv_summary_optimal_path}")

    # 2. Attention Scores 저장
    attention_dir = Path(args.model_save_dir) / "attention_scores"
    attention_dir.mkdir(parents=True, exist_ok=True)

    for fold_result in fold_results:
        fold_num = fold_result['fold']
        attention_data = {
            "fold": fold_num,
            "attention_scores": fold_result['full_attention_scores']
        }
        attention_path = attention_dir / f"attention_scores_fold{fold_num}.json"
        with open(attention_path, "w") as f:
            json.dump(convert_numpy(attention_data), f, indent=2)
        print(f"[✓] Fold {fold_num} attention scores saved: {attention_path}")

    print(f"\n{'='*80}")
    print(f"Summary of Saved Files:")
    print(f"  - 1 optimal threshold summary (per-fold) ⭐")
    print(f"  - {len(fold_results)} attention score files")
    print(f"{'='*80}")
    print(f"\n💡 Final Results: 'results_cv_summary_optimal.json'")
    print(f"   Mean optimal threshold: {cv_summary_optimal['mean_optimal_threshold']:.2f}")
    print(f"{'='*80}")

    # Visualization
    if args.generate_plots:
        viz_dir = Path(args.model_save_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Generating Visualization Plots")
        print(f"{'='*80}\n")
        
        # ✅ FIX: generate_all_plots를 위한 올바른 데이터 구조 (Optimal threshold 사용)
        fold_results_for_plotting = []
        for r in fold_results:
            plot_result = r.copy()
            plot_result['test_metrics'] = r['test_metrics_optimal']
            fold_results_for_plotting.append(plot_result)

        generate_all_plots(fold_results_for_plotting, viz_dir)
        print(f"✓ Plots generated using optimal thresholds")
        
        # Attention Heatmaps
        try:
            from evaluation.visualization import generate_attention_heatmaps_from_results
            
            print(f"\n{'='*80}")
            print(f"Generating Attention Heatmaps (Best Fold)")
            print(f"{'='*80}\n")
            
            json_meta_dir = os.getenv('JSON_META_DIR', '/path/to/json/meta')
            json_nonmeta_dir = os.getenv('JSON_NONMETA_DIR', '/path/to/json/nonmeta')
            
            generate_attention_heatmaps_from_results(
                results_dir=str(args.model_save_dir),
                json_meta_dir=json_meta_dir,
                json_nonmeta_dir=json_nonmeta_dir,
                save_dir=str(viz_dir),
                fold_num='best',
                n_positive=3,
                n_negative=3,
                interpolation='gaussian',
                dpi=200
            )
            
            print(f"✓ Attention heatmaps saved to: {viz_dir}")
            
        except ImportError as e:
            print(f"[!] Warning: Could not import visualization module: {e}")
        except Exception as e:
            print(f"[!] Error during heatmap generation: {e}")

    if args.save_model and model_paths:
        print(f"[✓] Saved {len(model_paths)} model checkpoints")

    print(f"\n[✓] Training completed!")
    print(f"   Best results saved in: results_cv_summary_optimal.json")
    print(f"   Mean optimal threshold: {cv_summary_optimal['mean_optimal_threshold']:.2f}")

    # Return training results for potential MLflow upload
    return {
        'model_save_dir': args.model_save_dir,
        'json_path': str(cv_summary_optimal_path),  # Ensure string path
        'cv_summary_path': str(cv_summary_optimal_path),  # Explicit key for MLflow
        'model_checkpoint_path': str(model_paths[0]) if model_paths else None,
        'args': args
    }
