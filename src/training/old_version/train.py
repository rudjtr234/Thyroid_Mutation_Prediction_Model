"""
ABMIL Training with Fixed CV Splits - 200 WSI Version
- Train:Val:Test = 8:1:1 (160:20:20)
- 5-Fold CV with fixed splits
- WSI-level splitting (no data leakage)
"""

import os
import glob
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                           recall_score, f1_score, confusion_matrix,
                           classification_report)
import warnings
import sys
import random
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings('ignore')

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)
sys.path.append(os.path.join(src_dir, 'models'))

from abmil import ABMILModel, ABMILGatedBaseConfig


class EarlyStopping:
    def __init__(self, patience=25, min_delta=0.001, restore_best_weights=True):
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
            self.best_weights = model.state_dict().copy()

        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ThyroidBagDataset(Dataset):
    def __init__(self, data_root, bag_size=2000):
        self.bags = []
        self.bag_size = bag_size

        print(f"Looking for files in: {data_root}")
        
        braf_dir = os.path.join(data_root, "meta")
        non_braf_dir = os.path.join(data_root, "nonmeta")
        
        if not os.path.exists(braf_dir):
            raise ValueError(f"BRAF+ directory not found: {braf_dir}")
        if not os.path.exists(non_braf_dir):
            raise ValueError(f"BRAF- directory not found: {non_braf_dir}")
        
        braf_files = glob.glob(os.path.join(braf_dir, "*.npy"))
        print(f"Found BRAF+ files: {len(braf_files)}")
        
        non_braf_files = glob.glob(os.path.join(non_braf_dir, "*.npy"))
        print(f"Found BRAF- files: {len(non_braf_files)}")

        def make_bags(features, label, filename, prefix):
            indices = np.random.permutation(len(features))
            features = features[indices]
            
            num_bags = max(1, int(np.ceil(len(features) / self.bag_size)))
            for i in range(num_bags):
                start = i * self.bag_size
                end = min((i + 1) * self.bag_size, len(features))
                bag_feats = features[start:end]
                
                self.bags.append({
                    'features': bag_feats,
                    'label': label,
                    'filename': filename,
                    'bag_id': f"{prefix}_{filename}_bag{i+1}"
                })

        for braf_file in braf_files:
            try:
                features = np.load(braf_file)
                filename = os.path.basename(braf_file)
                make_bags(features, label=1, filename=filename, prefix="BRAF_POS")
            except Exception as e:
                print(f"Warning: Could not load {braf_file}: {e}")
                continue
            
        for non_braf_file in non_braf_files:
            try:
                features = np.load(non_braf_file)
                filename = os.path.basename(non_braf_file)
                make_bags(features, label=0, filename=filename, prefix="BRAF_NEG")
            except Exception as e:
                print(f"Warning: Could not load {non_braf_file}: {e}")
                continue

        print(f"\nFinal Dataset: {len(self.bags)} bags from {len(braf_files + non_braf_files)} WSIs")
        
        labels = [bag['label'] for bag in self.bags]
        braf_pos_count = sum(labels)
        braf_neg_count = len(labels) - braf_pos_count
        print(f"Class distribution: BRAF+ = {braf_pos_count} bags, BRAF- = {braf_neg_count} bags")
        
        unique_wsis = len(set([bag['filename'] for bag in self.bags]))
        print(f"Total unique WSIs: {unique_wsis}")
        
        if len(self.bags) == 0:
            raise ValueError("No valid data files found!")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        features = torch.tensor(bag['features'], dtype=torch.float32)
        label = torch.tensor(bag['label'], dtype=torch.long)
        return features, label, bag['bag_id'], bag['filename']

    def get_wsi_list(self):
        return list(set([bag['filename'] for bag in self.bags]))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_cv_splits(dataset, k_folds=5, seed=42):
    """
    8:1:1 비율로 Train:Val:Test 분할하는 K-Fold CV splits 생성
    
    200개 WSI (100 BRAF+, 100 BRAF-) 기준:
    - 각 fold: Train 160개 (80+80), Val 20개 (10+10), Test 20개 (10+10)
    """
    unique_wsis = dataset.get_wsi_list()
    
    # 클래스별로 분할
    braf_pos_wsis = []
    braf_neg_wsis = []
    
    for wsi in unique_wsis:
        wsi_bags = [bag for bag in dataset.bags if bag['filename'] == wsi]
        if wsi_bags[0]['label'] == 1:
            braf_pos_wsis.append(wsi)
        else:
            braf_neg_wsis.append(wsi)
    
    # 시드 고정하고 섞기
    np.random.seed(seed)
    np.random.shuffle(braf_pos_wsis)
    np.random.shuffle(braf_neg_wsis)
    
    # 각 클래스를 k개 fold로 분할
    pos_folds = np.array_split(braf_pos_wsis, k_folds)
    neg_folds = np.array_split(braf_neg_wsis, k_folds)
    
    cv_splits = {
        'seed': seed,
        'k_folds': k_folds,
        'total_wsis': len(unique_wsis),
        'braf_pos_count': len(braf_pos_wsis),
        'braf_neg_count': len(braf_neg_wsis),
        'split_ratio': '8:1:1 (train:val:test)',
        'folds': []
    }
    
    for fold_idx in range(k_folds):
        # Test: 현재 fold (1개 fold = 20개 WSI)
        test_pos = list(pos_folds[fold_idx])
        test_neg = list(neg_folds[fold_idx])
        test_wsis = test_pos + test_neg
        
        # Val: 다음 fold (1개 fold = 20개 WSI)
        val_idx = (fold_idx + 1) % k_folds
        val_pos = list(pos_folds[val_idx])
        val_neg = list(neg_folds[val_idx])
        val_wsis = val_pos + val_neg
        
        # Train: 나머지 folds (3개 folds = 160개 WSI)
        train_wsis = []
        for i in range(k_folds):
            if i != fold_idx and i != val_idx:
                train_wsis.extend(pos_folds[i])
                train_wsis.extend(neg_folds[i])
        
        fold_data = {
            'fold': fold_idx + 1,
            'train_wsis': train_wsis,
            'train_count': len(train_wsis),
            'train_braf_pos': len([w for w in train_wsis if w in braf_pos_wsis]),
            'train_braf_neg': len([w for w in train_wsis if w in braf_neg_wsis]),
            'val_wsis': val_wsis,
            'val_count': len(val_wsis),
            'val_braf_pos': len([w for w in val_wsis if w in braf_pos_wsis]),
            'val_braf_neg': len([w for w in val_wsis if w in braf_neg_wsis]),
            'test_wsis': test_wsis,
            'test_count': len(test_wsis),
            'test_braf_pos': len([w for w in test_wsis if w in braf_pos_wsis]),
            'test_braf_neg': len([w for w in test_wsis if w in braf_neg_wsis])
        }
        
        cv_splits['folds'].append(fold_data)
    
    return cv_splits


def save_cv_splits(cv_splits, save_dir):
    """CV splits JSON 저장"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"cv_splits_k{cv_splits['k_folds']}_seed{cv_splits['seed']}.json"
    save_path = save_dir / filename
    
    with open(save_path, 'w') as f:
        json.dump(cv_splits, f, indent=2)
    
    print(f"\n[✓] CV splits saved: {save_path}")
    
    # 요약 출력
    print(f"\nCV Splits Summary:")
    print(f"  Total WSIs: {cv_splits['total_wsis']}")
    print(f"  BRAF+: {cv_splits['braf_pos_count']}, BRAF-: {cv_splits['braf_neg_count']}")
    print(f"  K-folds: {cv_splits['k_folds']}")
    print(f"  Seed: {cv_splits['seed']}")
    print(f"  Split ratio: {cv_splits['split_ratio']}")
    
    for fold_data in cv_splits['folds']:
        print(f"\n  Fold {fold_data['fold']}:")
        print(f"    Train: {fold_data['train_count']} WSIs ({fold_data['train_braf_pos']}+, {fold_data['train_braf_neg']}-)")
        print(f"    Val:   {fold_data['val_count']} WSIs ({fold_data['val_braf_pos']}+, {fold_data['val_braf_neg']}-)")
        print(f"    Test:  {fold_data['test_count']} WSIs ({fold_data['test_braf_pos']}+, {fold_data['test_braf_neg']}-)")
    
    return save_path


def load_cv_splits(split_file):
    """저장된 CV splits 로드"""
    with open(split_file, 'r') as f:
        cv_splits = json.load(f)
    
    print(f"[✓] Loaded CV splits from: {split_file}")
    print(f"    K-folds: {cv_splits['k_folds']}, Seed: {cv_splits['seed']}")
    
    return cv_splits


def train_one_epoch(model, train_bags, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    loss_fn = nn.CrossEntropyLoss()

    for features, label, bag_id, _ in train_bags:
        features = features.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        optimizer.zero_grad()
        results_dict, _ = model(h=features, loss_fn=loss_fn, label=label)
        loss = results_dict['loss']
        logits = results_dict['logits']

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(train_bags) if len(train_bags) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    return avg_loss, accuracy


def validate_one_epoch(model, val_bags, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, label, bag_id, _ in val_bags:
            features = features.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)

            results_dict, _ = model(h=features, loss_fn=loss_fn, label=label)
            loss = results_dict['loss']
            logits = results_dict['logits']

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(val_bags) if len(val_bags) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    return avg_loss, accuracy


def evaluate_model(model, test_bags, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    predictions = []

    with torch.no_grad():
        for features, label, bag_id, filename in test_bags:
            features = features.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)

            results_dict, log_dict = model(h=features, loss_fn=loss_fn, label=label, return_attention=True)
            logits = results_dict['logits']
            probs = torch.softmax(logits, dim=1)
            
            attention = log_dict.get('attention', None)
            if attention is not None:
                attention_weight = attention[0].mean().item()
            else:
                attention_weight = 1.0

            predictions.append({
                'bag_id': bag_id,
                'filename': filename,
                'true_label': label.item(),
                'braf_prob': probs[0, 1].item(),
                'attention_weight': attention_weight
            })

    # WSI 단위로 aggregation
    wsi_results = {}
    for p in predictions:
        if p['filename'] not in wsi_results:
            wsi_results[p['filename']] = {
                'probs': [], 
                'attention_weights': [],
                'true_label': p['true_label']
            }
        wsi_results[p['filename']]['probs'].append(p['braf_prob'])
        wsi_results[p['filename']]['attention_weights'].append(p['attention_weight'])

    aggregated = []
    for fname, vals in wsi_results.items():
        probs = np.array(vals['probs'])
        attn_weights = np.array(vals['attention_weights'])
        
        attn_weights_sum = attn_weights.sum()
        if attn_weights_sum > 0:
            attn_weights_norm = attn_weights / attn_weights_sum
            weighted_prob = np.sum(probs * attn_weights_norm)
        else:
            weighted_prob = np.mean(probs)
        
        simple_avg_prob = np.mean(probs)
        pred_label = 1 if weighted_prob >= 0.5 else 0
        
        aggregated.append({
            'filename': fname,
            'true_label': vals['true_label'],
            'weighted_prob': weighted_prob,
            'simple_avg_prob': simple_avg_prob,
            'avg_prob': weighted_prob,
            'pred_label': pred_label,
            'num_bags': len(probs)
        })

    return aggregated


def comprehensive_evaluation(true_labels, predictions, threshold=0.5):
    pred_labels = (np.array(predictions) >= threshold).astype(int)
    
    results = {
        'auc': roc_auc_score(true_labels, predictions) if len(set(true_labels)) > 1 else 0.5,
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels, zero_division=0),
        'recall': recall_score(true_labels, pred_labels, zero_division=0),
        'f1_score': f1_score(true_labels, pred_labels, zero_division=0),
        'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist(),
        'classification_report': classification_report(true_labels, pred_labels, zero_division=0)
    }
    return results


def save_model_checkpoint(model, fold_idx, fold_result, save_dir, args, is_best=False):
    """폴드별 모델 체크포인트 저장"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'fold': fold_idx + 1,
        'model_state_dict': model.state_dict(),
        'accuracy': fold_result['accuracy'],
        'auc': fold_result['auc'],
        'config': {
            'k_folds': args.k_folds,
            'lr': args.lr,
            'bag_size': args.bag_size,
            'seed': args.seed,
            'model': 'ABMILGatedBase'
        }
    }
    
    # 파일명
    if is_best:
        filename = f"best_model_fold{fold_idx+1}_auc{fold_result['auc']:.4f}.pt"
    else:
        filename = f"model_fold{fold_idx+1}_auc{fold_result['auc']:.4f}.pt"
    
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)
    
    # 파일 크기 확인
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"[✓] Model saved: {checkpoint_path} ({file_size_mb:.2f} MB)")
    
    return checkpoint_path


def run_k_fold_cv(dataset, cv_splits, args, device):
    """K-Fold Cross-Validation with model saving"""
    print(f"\nRunning {cv_splits['k_folds']}-Fold CV (8:1:1 split)")
    print("=" * 60)
    
    all_fold_results = []
    all_predictions = []
    all_predictions_simple = []
    all_true_labels = []
    saved_model_paths = []
    
    # 모델 저장 디렉토리
    model_save_dir = Path(args.model_save_dir)
    
    config = ABMILGatedBaseConfig()

    for fold_data in cv_splits['folds']:
        fold_idx = fold_data['fold'] - 1
        print(f"\nFold {fold_data['fold']}/{cv_splits['k_folds']}")
        
        train_wsis = fold_data['train_wsis']
        val_wsis = fold_data['val_wsis']
        test_wsis = fold_data['test_wsis']
        
        print(f"Train: {len(train_wsis)} WSIs, Val: {len(val_wsis)} WSIs, Test: {len(test_wsis)} WSIs")

        # Bag 데이터 준비
        train_bags = [dataset[i] for i in range(len(dataset)) 
                     if dataset.bags[i]['filename'] in train_wsis]
        val_bags = [dataset[i] for i in range(len(dataset)) 
                   if dataset.bags[i]['filename'] in val_wsis]
        test_bags = [dataset[i] for i in range(len(dataset)) 
                    if dataset.bags[i]['filename'] in test_wsis]
        
        print(f"Train bags: {len(train_bags)}, Val bags: {len(val_bags)}, Test bags: {len(test_bags)}")

        # 모델 초기화
        model = ABMILModel(config).to(device)
        
        if fold_idx == 0:
            print("Model architecture:")
            print(model)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total params: {total_params:,}")

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        early_stopping = EarlyStopping(patience=30, min_delta=0.001)

        print("Starting training...")
        
        # 훈련 루프
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_bags, optimizer, device)
            val_loss, val_acc = validate_one_epoch(model, val_bags, device)
            
            scheduler.step(val_loss)
            
            if early_stopping(val_acc, model):
                print(f"Early stopping at epoch {epoch+1}")
                early_stopping.restore_best(model)
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # 테스트 평가
        test_results = evaluate_model(model, test_bags, device)
        
        fold_predictions = [r['weighted_prob'] for r in test_results]
        fold_predictions_simple = [r['simple_avg_prob'] for r in test_results]
        fold_true_labels = [r['true_label'] for r in test_results]
        fold_pred_labels = [r['pred_label'] for r in test_results]
        
        fold_accuracy = accuracy_score(fold_true_labels, fold_pred_labels)
        fold_auc = roc_auc_score(fold_true_labels, fold_predictions) if len(set(fold_true_labels)) > 1 else 0.5
        
        fold_pred_labels_simple = [1 if p >= 0.5 else 0 for p in fold_predictions_simple]
        fold_accuracy_simple = accuracy_score(fold_true_labels, fold_pred_labels_simple)
        fold_auc_simple = roc_auc_score(fold_true_labels, fold_predictions_simple) if len(set(fold_true_labels)) > 1 else 0.5
        
        print(f"Fold {fold_data['fold']} Results:")
        print(f"  Attention-weighted: Accuracy={fold_accuracy:.4f}, AUC={fold_auc:.4f}")
        print(f"  Simple average:     Accuracy={fold_accuracy_simple:.4f}, AUC={fold_auc_simple:.4f}")
        
        all_predictions.extend(fold_predictions)
        all_predictions_simple.extend(fold_predictions_simple)
        all_true_labels.extend(fold_true_labels)
        
        fold_result = {
            'fold': fold_data['fold'],
            'accuracy': fold_accuracy,
            'auc': fold_auc,
            'accuracy_simple': fold_accuracy_simple,
            'auc_simple': fold_auc_simple,
            'test_wsis': test_wsis,
            'test_results': test_results
        }
        all_fold_results.append(fold_result)
        
        # 모델 저장
        if args.save_model:
            model_path = save_model_checkpoint(model, fold_idx, fold_result, model_save_dir, args)
            saved_model_paths.append(str(model_path))

    # 최고 성능 모델 별도 저장
    if args.save_model and args.save_best_only:
        best_fold_idx = max(range(len(all_fold_results)), key=lambda i: all_fold_results[i]['auc'])
        best_fold = all_fold_results[best_fold_idx]
        print(f"\n[Best Model] Fold {best_fold['fold']}, AUC: {best_fold['auc']:.4f}")

    return all_fold_results, all_predictions, all_predictions_simple, all_true_labels, saved_model_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                       default='/data/143/member/jks/Thyroid_Mutation_dataset/embeddings',
                       help='Data directory path')
    parser.add_argument('--model_save_dir', type=str,
                       default='/home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model/outputs/Thyroid_prediction_model_v0.1.0',
                       help='Model checkpoint save directory')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--bag_size', type=int, default=2000, help='Patches per bag')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cv_split_file', type=str, default=None,
                       help='Existing CV split file to load')
    parser.add_argument('--create_splits_only', action='store_true',
                       help='Only create and save CV splits without training')
    parser.add_argument('--save_model', action='store_true',
                       help='Save model checkpoints for each fold')
    parser.add_argument('--save_best_only', action='store_true',
                       help='Save only the best performing fold model')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model save directory: {args.model_save_dir}")

    # 데이터셋 로드
    print("Loading dataset...")
    dataset = ThyroidBagDataset(args.data_root, bag_size=args.bag_size)
    
    if len(dataset) < 10:
        raise ValueError("Need at least 10 bags!")

    # CV splits 처리
    cv_splits_dir = Path(args.data_root).parent / "cv_splits"
    
    if args.cv_split_file and Path(args.cv_split_file).exists():
        cv_splits = load_cv_splits(args.cv_split_file)
    else:
        print("\nCreating new CV splits...")
        cv_splits = create_cv_splits(dataset, k_folds=args.k_folds, seed=args.seed)
        split_file = save_cv_splits(cv_splits, cv_splits_dir)
        
        if args.create_splits_only:
            print("\n[✓] CV splits created. Exiting without training.")
            return

    # K-Fold CV 실행
    fold_results, predictions, predictions_simple, true_labels, model_paths = \
        run_k_fold_cv(dataset, cv_splits, args, device)

    # 결과 출력
    print("\n" + "=" * 60)
    print("Cross-Validation Results Summary")
    print("=" * 60)
    
    if len(set(true_labels)) > 1:
        eval_results = comprehensive_evaluation(true_labels, predictions)
        eval_results_simple = comprehensive_evaluation(true_labels, predictions_simple)
        
        print(f"\n[Attention-Weighted Aggregation]")
        print(f"   AUC: {eval_results['auc']:.4f}")
        print(f"   Accuracy: {eval_results['accuracy']:.4f}")
        print(f"   Precision: {eval_results['precision']:.4f}")
        print(f"   Recall: {eval_results['recall']:.4f}")
        print(f"   F1-Score: {eval_results['f1_score']:.4f}")
        
        print(f"\n[Simple Average Aggregation]")
        print(f"   AUC: {eval_results_simple['auc']:.4f}")
        print(f"   Accuracy: {eval_results_simple['accuracy']:.4f}")
        print(f"   Precision: {eval_results_simple['precision']:.4f}")
        print(f"   Recall: {eval_results_simple['recall']:.4f}")
        print(f"   F1-Score: {eval_results_simple['f1_score']:.4f}")
        
        print(f"\nPer-Fold Performance:")
        fold_accs = [f['accuracy'] for f in fold_results]
        fold_aucs = [f['auc'] for f in fold_results]
        
        for i, (acc, auc) in enumerate(zip(fold_accs, fold_aucs)):
            print(f"   Fold {i+1}: Accuracy={acc:.4f}, AUC={auc:.4f}")
        
        print(f"\nMean ± Std (Attention-Weighted):")
        print(f"   Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
        print(f"   AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    
    # 모델 저장 경로 출력
    if args.save_model and model_paths:
        print(f"\n[✓] Saved {len(model_paths)} model checkpoints:")
        for path in model_paths:
            print(f"    {path}")

    # 결과 저장
    save_results(fold_results, predictions, predictions_simple, true_labels, cv_splits, args)
    
    print("\n[✓] Training completed!")


if __name__ == "__main__":
    main()