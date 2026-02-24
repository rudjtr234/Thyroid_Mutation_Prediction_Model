# -*- coding: utf-8 -*-
"""
Ensemble Training Script for BRAF Mutation Classification

5개의 독립적인 모델을 학습하고, 동일한 Test Set으로 앙상블 평가
- 각 모델: 서로 다른 Positive 700장 + 공통 Negative 700장
- Test Set: 고정 (모든 모델이 공유)
- MLflow: 단일 run에 5개 모델 결과 + 평균

Version: ensemble_v1.0
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

# GPU 설정 강제
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
training_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(training_dir)
sys.path.insert(0, src_dir)

evaluation_dir = os.path.join(src_dir, 'evaluation')
sys.path.insert(0, evaluation_dir)

# =========================
# Models & Utils import
# =========================
from models.abmil import ABMILModel, ABMILGatedBaseConfig
from utils.datasets import set_seed
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset

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
# Ensemble Dataset (JSON 기반)
# =========================
class EnsembleDataset(Dataset):
    """
    앙상블 학습용 Dataset
    - JSON에서 positive/negative 파일 리스트를 읽어서 구성
    """
    def __init__(self, positive_files: List[str], negative_files: List[str],
                 pos_dir: str, neg_dir: str, bag_size: int = 2000):
        self.wsi_list = []
        self.bag_size = bag_size

        # Positive files (label=1)
        for filename in positive_files:
            filepath = os.path.join(pos_dir, filename)
            if os.path.exists(filepath):
                self.wsi_list.append({
                    'filepath': filepath,
                    'label': 1,
                    'filename': filename
                })
            else:
                print(f"Warning: Positive file not found: {filepath}")

        # Negative files (label=0)
        for filename in negative_files:
            filepath = os.path.join(neg_dir, filename)
            if os.path.exists(filepath):
                self.wsi_list.append({
                    'filepath': filepath,
                    'label': 0,
                    'filename': filename
                })
            else:
                print(f"Warning: Negative file not found: {filepath}")

        labels = [wsi['label'] for wsi in self.wsi_list]
        print(f"Dataset: {len(self.wsi_list)} WSIs (Pos: {sum(labels)}, Neg: {len(labels) - sum(labels)})")

    def _select_tiles(self, features):
        if self.bag_size is None:
            return features
        if len(features) <= self.bag_size:
            return features
        indices = np.random.choice(len(features), self.bag_size, replace=False)
        return features[indices]

    def __len__(self):
        return len(self.wsi_list)

    def __getitem__(self, idx):
        wsi = self.wsi_list[idx]
        features = np.load(wsi['filepath'])
        features = self._select_tiles(features)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(wsi['label'], dtype=torch.long)
        return features, label, wsi['filename']


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
def save_model_checkpoint(model, model_idx, model_result, save_dir, args, is_best=False):
    save_dir = Path(save_dir) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_id': model_idx,
        'model_state_dict': model.state_dict(),
        'val_auc': model_result['best_val_metrics']['auc'],
        'config': {
            'lr': args.lr,
            'bag_size': args.bag_size,
            'seed': args.seed,
            'model': 'ABMILGatedBase'
        }
    }

    filename = f"{'best_' if is_best else ''}model_{model_idx}_auc{model_result['best_val_metrics']['auc']:.4f}.pt"
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)

    print(f"[✓] Model saved: {checkpoint_path}")
    return checkpoint_path


# =========================
# Load Ensemble Split JSON
# =========================
def load_ensemble_splits(ensemble_json_path: str, test_json_path: str):
    """
    앙상블 학습용 JSON 파일 로드

    Returns:
        ensemble_data: 5개 모델의 train/val 데이터
        test_data: 공통 테스트 데이터
    """
    with open(ensemble_json_path, 'r') as f:
        ensemble_data = json.load(f)

    with open(test_json_path, 'r') as f:
        test_data = json.load(f)

    print(f"\n{'='*80}")
    print(f"Loaded Ensemble Configuration")
    print(f"{'='*80}")
    print(f"Number of models: {len(ensemble_data['models'])}")
    print(f"Shared negative count: {ensemble_data['shared_negative']['count']}")
    print(f"Test set: Pos={len(test_data['positive'])}, Neg={len(test_data['negative'])}")
    print(f"{'='*80}\n")

    return ensemble_data, test_data


# =========================
# Training Loop
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
                logits = results_dict['logits']
                loss = results_dict['loss']
                loss.backward()
                optimizer.step()
            else:
                results_dict, _ = model(h=features, loss_fn=loss_fn, label=label)
                logits = results_dict['logits']
                loss = results_dict['loss']

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


# =========================
# Test Evaluation (개별 모델)
# =========================
def evaluate_model_on_test(model, test_loader, device, threshold=0.5):
    """Test set 평가 - 확률값 반환"""
    model.eval()
    all_probs, all_labels, all_filenames = [], [], []

    with torch.no_grad():
        for features, label, filename in test_loader:
            features = features.to(device)
            label = label.to(device)

            results_dict, _ = model(h=features, loss_fn=None, label=None)
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


# =========================
# Ensemble Prediction (앙상블 평균)
# =========================
def ensemble_predict(all_model_probs: List[List[float]], labels: List[int], threshold=0.5):
    """
    앙상블 예측 (확률 평균)

    Args:
        all_model_probs: 각 모델의 예측 확률 리스트 [model1_probs, model2_probs, ...]
        labels: 실제 레이블
        threshold: 분류 임계값

    Returns:
        metrics: 앙상블 성능 지표
        ensemble_probs: 앙상블 확률 (평균)
    """
    # 확률 평균
    all_model_probs = np.array(all_model_probs)  # (n_models, n_samples)
    ensemble_probs = np.mean(all_model_probs, axis=0)  # (n_samples,)

    # Threshold로 예측
    ensemble_preds = (ensemble_probs >= threshold).astype(int)

    if len(set(labels)) > 1:
        metrics = compute_metrics_with_confusion(labels, ensemble_preds, ensemble_probs.tolist())
    else:
        metrics = {
            "accuracy": 0.0, "auc": 0.5,
            "sensitivity": 0.0, "specificity": 0.0,
            "ppv": 0.0, "npv": 0.0,
            "f1": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0
        }

    return metrics, ensemble_probs.tolist()


# =========================
# Full WSI Attention Extraction
# =========================
def _extract_full_wsi_attention(model, test_filenames, test_labels, device,
                                 meta_npy_dir, nonmeta_npy_dir):
    """
    Extract full WSI attention scores from original embeddings.
    Reuses the same logic as train_bag.py's evaluate_full_wsi_for_visualization.
    """
    model.eval()
    attention_scores_dict = {}

    # Build unique WSI list from filenames
    wsi_names = list(set(Path(f).stem for f in test_filenames))

    # Build label mapping
    label_map = {}
    for f, lbl in zip(test_filenames, test_labels):
        label_map[Path(f).stem] = int(lbl)

    with torch.no_grad():
        for wsi_name in wsi_names:
            meta_path = Path(meta_npy_dir) / f"{wsi_name}.npy"
            nonmeta_path = Path(nonmeta_npy_dir) / f"{wsi_name}.npy"

            if meta_path.exists():
                full_features = np.load(meta_path)
            elif nonmeta_path.exists():
                full_features = np.load(nonmeta_path)
            else:
                continue

            features_tensor = torch.from_numpy(full_features).float().unsqueeze(0).to(device)

            try:
                results_dict = model(
                    h=features_tensor, loss_fn=None, label=None,
                    return_attention=True, return_extra=True
                )

                attention_weights = results_dict['attention']
                if attention_weights is None:
                    continue

                if attention_weights.dim() == 3:
                    attn_raw = attention_weights[0, 0, :]
                elif attention_weights.dim() == 2:
                    attn_raw = attention_weights[0, :]
                else:
                    continue

                attn_scores = F.softmax(attn_raw, dim=0).cpu().numpy()
                logits = results_dict['logits']
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                attention_scores_dict[wsi_name] = {
                    'scores': attn_scores.tolist(),
                    'n_patches': len(attn_scores),
                    'predicted_label': int(preds.cpu().numpy()[0]),
                    'pred_prob': float(probs.cpu().numpy()[0]),
                    'true_label': label_map.get(wsi_name),
                }
            except Exception as e:
                print(f"    Failed attention extraction for {wsi_name}: {e}")
                continue

    return attention_scores_dict


# =========================
# Main Training Pipeline
# =========================
def run_ensemble_training(args):
    """5개 앙상블 모델 학습 파이프라인"""
    set_seed(args.seed)

    # GPU 설정
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load ensemble configuration
    ensemble_data, test_data = load_ensemble_splits(args.ensemble_json, args.test_json)

    # Test Dataset 생성 (공통)
    # test_data['positive']와 test_data['negative']가 dict인 경우 처리
    if isinstance(test_data['positive'], dict):
        test_pos_files = test_data['positive']['files']
        test_pos_dir = test_data['positive']['dir']
    else:
        test_pos_files = test_data['positive']
        test_pos_dir = test_data.get('positive_dir', args.data_root + '/meta/npy')

    if isinstance(test_data['negative'], dict):
        test_neg_files = test_data['negative']['files']
        test_neg_dir = test_data['negative']['dir']
    else:
        test_neg_files = test_data['negative']
        test_neg_dir = test_data.get('negative_dir', args.data_root + '/non-meta/npy')

    test_dataset = EnsembleDataset(
        positive_files=test_pos_files,
        negative_files=test_neg_files,
        pos_dir=test_pos_dir,
        neg_dir=test_neg_dir,
        bag_size=None  # Test는 전체 사용
    )

    n_cpus = cpu_count()
    val_workers = 4
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                            num_workers=val_workers, pin_memory=True)

    print(f"\n{'='*80}")
    print(f"Test Set: {len(test_dataset)} samples (shared across all models)")
    print(f"{'='*80}\n")

    # 결과 저장용
    all_model_results = []
    all_test_probs = []  # 앙상블용
    test_labels = None
    saved_model_paths = []
    config = ABMILGatedBaseConfig()

    # 5개 모델 학습
    for model_info in ensemble_data['models']:
        model_id = model_info['model_id']

        print(f"\n{'='*80}")
        print(f"Training Model {model_id}/{len(ensemble_data['models'])}")
        print(f"{'='*80}")

        # Train/Val Dataset 생성
        train_pos_files = model_info['train']['positive']['files']
        train_neg_files = model_info['train']['negative']['files']
        train_pos_dir = model_info['train']['positive']['dir']
        train_neg_dir = model_info['train']['negative']['dir']

        val_pos_files = model_info['val']['positive']['files']
        val_neg_files = model_info['val']['negative']['files']
        val_pos_dir = model_info['val']['positive']['dir']
        val_neg_dir = model_info['val']['negative']['dir']

        train_dataset = EnsembleDataset(
            positive_files=train_pos_files,
            negative_files=train_neg_files,
            pos_dir=train_pos_dir,
            neg_dir=train_neg_dir,
            bag_size=args.bag_size
        )

        val_dataset = EnsembleDataset(
            positive_files=val_pos_files,
            negative_files=val_neg_files,
            pos_dir=val_pos_dir,
            neg_dir=val_neg_dir,
            bag_size=None  # Val은 전체 패치 사용 (안정적인 평가)
        )

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # DataLoader
        train_workers = 4
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                                 num_workers=train_workers, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                               num_workers=val_workers, pin_memory=True,
                               persistent_workers=True, prefetch_factor=2)

        # Model
        model = ABMILModel(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        early_stopping = EarlyStopping(patience=8, min_delta=0.001)

        best_train_metrics, best_val_metrics = {}, {}
        history = {'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_f1': [],
                   'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': []}

        # Training loop
        for epoch in range(args.epochs):
            train_metrics = run_one_epoch(model, train_loader, device, optimizer, train=True)
            val_metrics = run_one_epoch(model, val_loader, device, train=False)
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
                  f"Train AUC: {train_metrics['auc']:.4f} | "
                  f"Train F1: {train_metrics['f1']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            if val_metrics["auc"] > best_val_metrics.get("auc", 0):
                best_val_metrics = val_metrics
                best_train_metrics = train_metrics
                print(f"  ↑ Best model updated (Val AUC: {val_metrics['auc']:.4f})")

            if early_stopping(val_metrics["auc"], model, epoch+1):
                print(f"\n⚠ Early stopping at epoch {epoch+1}")
                early_stopping.restore_best(model)
                break

        # Test evaluation
        print(f"\n{'─'*80}")
        print(f"Evaluating Model {model_id} on Test Set")
        print(f"{'─'*80}")

        test_metrics, test_probs, labels, test_filenames = evaluate_model_on_test(
            model, test_loader, device, threshold=0.5
        )

        if test_labels is None:
            test_labels = labels

        all_test_probs.append(test_probs)

        print(f"Test AUC: {test_metrics['auc']:.4f}, Test F1: {test_metrics['f1']:.4f}")

        # Full WSI attention extraction (for heatmap visualization)
        full_attention_scores = {}
        if args.generate_plots:
            print(f"\n  Extracting full WSI attention for Model {model_id}...")
            full_attention_scores = _extract_full_wsi_attention(
                model, test_filenames, labels, device,
                meta_npy_dir=test_pos_dir,
                nonmeta_npy_dir=test_neg_dir,
            )
            print(f"  Extracted attention for {len(full_attention_scores)} WSIs")

        # ROC/PR curve
        fpr, tpr, _ = compute_roc_curve_data(labels, test_probs)
        precision, recall, _ = compute_precision_recall_curve_data(labels, test_probs)

        # 모델 결과 저장
        model_result = {
            "model_id": model_id,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "best_epoch": early_stopping.best_epoch,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "history": history,
            "test_fpr": fpr.tolist(),
            "test_tpr": tpr.tolist(),
            "test_precision": precision.tolist(),
            "test_recall": recall.tolist(),
            "test_probs": test_probs,
            "test_filenames": test_filenames,
            "full_attention_scores": full_attention_scores,
        }
        all_model_results.append(model_result)

        # Model checkpoint
        if args.save_model:
            path = save_model_checkpoint(model, model_id, model_result, args.model_save_dir, args)
            saved_model_paths.append(path)

    # =========================
    # Ensemble Evaluation
    # =========================
    print(f"\n{'='*80}")
    print(f"Ensemble Evaluation (Average of 5 Models)")
    print(f"{'='*80}")

    ensemble_metrics, ensemble_probs = ensemble_predict(all_test_probs, test_labels, threshold=0.5)

    print(f"\n🎯 Ensemble Test Results:")
    print(f"{'─'*60}")
    print(f"{'Metric':<15} {'Value':<10}")
    print(f"{'─'*60}")
    print(f"{'AUC':<15} {ensemble_metrics['auc']:<10.4f}")
    print(f"{'Accuracy':<15} {ensemble_metrics['accuracy']:<10.4f}")
    print(f"{'F1-Score':<15} {ensemble_metrics['f1']:<10.4f}")
    print(f"{'Sensitivity':<15} {ensemble_metrics['sensitivity']:<10.4f}")
    print(f"{'Specificity':<15} {ensemble_metrics['specificity']:<10.4f}")
    print(f"{'PPV':<15} {ensemble_metrics['ppv']:<10.4f}")
    print(f"{'NPV':<15} {ensemble_metrics['npv']:<10.4f}")
    print(f"{'─'*60}\n")

    # =========================
    # Summary Statistics (5개 모델 평균)
    # =========================
    print(f"\n{'='*80}")
    print(f"Individual Model Summary")
    print(f"{'='*80}")
    print(f"{'Model':<8} {'Train AUC':<12} {'Val AUC':<12} {'Test AUC':<12} {'Test F1':<12}")
    print(f"{'─'*60}")

    test_aucs = []
    test_f1s = []
    for r in all_model_results:
        test_aucs.append(r['test_metrics']['auc'])
        test_f1s.append(r['test_metrics']['f1'])
        print(f"Model {r['model_id']:<3} "
              f"{r['best_train_metrics'].get('auc', 0):<12.4f} "
              f"{r['best_val_metrics'].get('auc', 0):<12.4f} "
              f"{r['test_metrics']['auc']:<12.4f} "
              f"{r['test_metrics']['f1']:<12.4f}")

    print(f"{'─'*60}")
    print(f"{'Mean':<8} {'':<12} {'':<12} {np.mean(test_aucs):<12.4f} {np.mean(test_f1s):<12.4f}")
    print(f"{'Std':<8} {'':<12} {'':<12} {np.std(test_aucs):<12.4f} {np.std(test_f1s):<12.4f}")
    print(f"{'─'*60}")
    print(f"{'Ensemble':<8} {'':<12} {'':<12} {ensemble_metrics['auc']:<12.4f} {ensemble_metrics['f1']:<12.4f}")
    print(f"{'='*80}\n")

    # =========================
    # Save Results
    # =========================
    print(f"\n{'='*80}")
    print(f"Saving Results")
    print(f"{'='*80}")

    # Summary statistics
    summary_stats = {
        "mean_test_auc": float(np.mean(test_aucs)),
        "std_test_auc": float(np.std(test_aucs)),
        "mean_test_f1": float(np.mean(test_f1s)),
        "std_test_f1": float(np.std(test_f1s)),
        "ensemble_auc": ensemble_metrics['auc'],
        "ensemble_f1": ensemble_metrics['f1'],
    }

    # Convert numpy types
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

    # Main results JSON
    results = {
        "description": "BRAF Ensemble (5 models) Results",
        "threshold": 0.5,
        "summary_statistics": summary_stats,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_probs": ensemble_probs,
        "models": []
    }

    for model_result in all_model_results:
        # test_probs와 test_filenames는 별도 저장하지 않음 (용량 문제)
        model_summary = {
            "model_id": model_result['model_id'],
            "train_size": model_result['train_size'],
            "val_size": model_result['val_size'],
            "test_size": model_result['test_size'],
            "best_epoch": model_result['best_epoch'],
            "best_train_metrics": model_result['best_train_metrics'],
            "best_val_metrics": model_result['best_val_metrics'],
            "test_metrics": model_result['test_metrics'],
            "history": model_result['history'],
            "test_fpr": model_result['test_fpr'],
            "test_tpr": model_result['test_tpr'],
            "test_precision": model_result['test_precision'],
            "test_recall": model_result['test_recall'],
        }
        results["models"].append(model_summary)

    results_path = Path(args.model_save_dir) / "ensemble_results.json"
    with open(results_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"[✓] Results saved: {results_path}")

    # Visualization
    if args.generate_plots:
        viz_dir = Path(args.model_save_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Fold 형식으로 변환 (기존 generate_all_plots 재사용)
        fold_results_for_plotting = []
        for r in all_model_results:
            plot_result = {
                'fold': r['model_id'],
                'test_metrics': r['test_metrics'],
                'test_fpr': r['test_fpr'],
                'test_tpr': r['test_tpr'],
                'test_precision': r['test_precision'],
                'test_recall': r['test_recall'],
                'history': r['history'],
                'best_epoch': r['best_epoch'],
                'best_train_metrics': r['best_train_metrics'],
                'best_val_metrics': r['best_val_metrics'],
            }
            fold_results_for_plotting.append(plot_result)

        generate_all_plots(fold_results_for_plotting, viz_dir)
        print(f"[✓] Plots saved: {viz_dir}")

        # Attention Heatmaps (individual best + ensemble average)
        try:
            from evaluation.visualization import generate_ensemble_attention_heatmaps

            # Collect attention scores from all models
            all_model_attention = [r['full_attention_scores'] for r in all_model_results]

            # Find best model by val AUC
            best_model_idx = max(
                range(len(all_model_results)),
                key=lambda i: all_model_results[i]['best_val_metrics'].get('auc', 0)
            )

            # Use first model's test_filenames (all models share the same test set)
            test_fnames = all_model_results[0]['test_filenames']

            # Save attention scores JSON
            attention_dir = Path(args.model_save_dir) / "attention_scores"
            attention_dir.mkdir(parents=True, exist_ok=True)

            for r in all_model_results:
                mid = r['model_id']
                attn_data = {"model_id": mid, "attention_scores": r['full_attention_scores']}
                attn_path = attention_dir / f"attention_scores_model{mid}.json"
                with open(attn_path, "w") as f:
                    json.dump(convert_numpy(attn_data), f, indent=2)
                print(f"[✓] Model {mid} attention scores saved: {attn_path}")

            json_meta_dir = getattr(args, 'json_meta_dir', os.getenv('JSON_META_DIR', '/path/to/json/meta'))
            json_nonmeta_dir = getattr(args, 'json_nonmeta_dir', os.getenv('JSON_NONMETA_DIR', '/path/to/json/nonmeta'))
            svs_base_dir = getattr(args, 'svs_base_dir', os.getenv('SVS_BASE_DIR', '/path/to/svs_base_dir'))

            generate_ensemble_attention_heatmaps(
                all_model_attention_scores=all_model_attention,
                ensemble_probs=ensemble_probs,
                test_labels=test_labels,
                test_filenames=test_fnames,
                json_meta_dir=json_meta_dir,
                json_nonmeta_dir=json_nonmeta_dir,
                save_dir=str(viz_dir),
                n_positive=5,
                n_negative=5,
                svs_base_dir=svs_base_dir,
                thumbnail_max_side=2048,
                best_model_idx=best_model_idx,
            )
            print(f"[✓] Attention heatmaps saved to: {viz_dir}")

        except Exception as e:
            print(f"[!] Warning: Could not generate attention heatmaps: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[✓] Ensemble training completed!")
    print(f"   Results: {results_path}")
    print(f"   Ensemble AUC: {ensemble_metrics['auc']:.4f}")
    print(f"   Ensemble F1: {ensemble_metrics['f1']:.4f}")

    return {
        'model_save_dir': args.model_save_dir,
        'json_path': str(results_path),
        'model_checkpoint_path': str(saved_model_paths[0]) if saved_model_paths else None,
        'ensemble_metrics': ensemble_metrics,
        'args': args
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train 5 Ensemble Models for BRAF Mutation')
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
    parser.add_argument('--json_meta_dir', type=str,
                        default='/path/to/json/meta',
                        help='JSON metadata directory for meta cases')
    parser.add_argument('--json_nonmeta_dir', type=str,
                        default='/path/to/json/nonmeta',
                        help='JSON metadata directory for nonmeta cases')
    parser.add_argument('--svs_base_dir', type=str,
                        default='/path/to/svs_base_dir',
                        help='SVS base directory for heatmap overlay')

    args = parser.parse_args()
    run_ensemble_training(args)
