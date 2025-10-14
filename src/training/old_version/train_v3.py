"""
ABMIL Training with Improved Regularization
- Weight decay 조정 가능
- Gradient clipping 추가
- Early stopping을 AUC 기준으로 변경
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import warnings
import sys
import json

warnings.filterwarnings('ignore')

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)
sys.path.append(os.path.join(src_dir, 'models'))

from abmil import ABMILModel, ABMILGatedBaseConfig
from utils.datasets import ThyroidBagDataset, set_seed
from utils.cv_splits import create_cv_splits, save_cv_splits, load_cv_splits
from utils.metrics import comprehensive_evaluation


# ----------------------
# EarlyStopping (AUC 기준)
# ----------------------
class EarlyStopping:
    def __init__(self, patience=25, min_delta=0.001, restore_best_weights=True, mode='max'):
        """
        mode: 'max' for AUC/Accuracy, 'min' for Loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model=None):
        improved = False
        
        if self.best_score is None:
            self.best_score = score
            improved = True
        else:
            if self.mode == 'max':
                if score > self.best_score + self.min_delta:
                    self.best_score = score
                    self.counter = 0
                    improved = True
                else:
                    self.counter += 1
            else:  # mode == 'min'
                if score < self.best_score - self.min_delta:
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


# ----------------------
# Train / Val / Test
# ----------------------
def train_one_epoch(model, train_bags, optimizer, device, max_grad_norm=1.0):
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
        
        # Gradient clipping 추가
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
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
    all_preds, all_labels, all_probs = [], [], []
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, label, bag_id, _ in val_bags:
            features = features.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)

            results_dict, _ = model(h=features, loss_fn=loss_fn, label=label)
            loss = results_dict['loss']
            logits = results_dict['logits']

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(val_bags) if len(val_bags) > 0 else 0

    if len(set(all_labels)) > 1:
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    else:
        accuracy, auc, precision, recall, f1 = 0.0, 0.5, 0.0, 0.0, 0.0

    return avg_loss, accuracy, auc, precision, recall, f1


def evaluate_model(model, test_bags, device):
    model.eval()
    predictions = []
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for features, label, bag_id, filename in test_bags:
            features = features.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)

            results_dict, log_dict = model(h=features, loss_fn=loss_fn, label=label, return_attention=True)
            logits = results_dict['logits']
            probs = torch.softmax(logits, dim=1)

            attention = log_dict.get('attention', None)
            attention_weight = attention[0].mean().item() if attention is not None else 1.0

            predictions.append({
                'bag_id': bag_id,
                'filename': filename,
                'true_label': label.item(),
                'braf_prob': probs[0, 1].item(),
                'attention_weight': attention_weight
            })

    # WSI 단위 aggregation
    wsi_results = {}
    for p in predictions:
        if p['filename'] not in wsi_results:
            wsi_results[p['filename']] = {'probs': [], 'attention_weights': [], 'true_label': p['true_label']}
        wsi_results[p['filename']]['probs'].append(p['braf_prob'])
        wsi_results[p['filename']]['attention_weights'].append(p['attention_weight'])

    aggregated = []
    for fname, vals in wsi_results.items():
        probs = np.array(vals['probs'])
        attn_weights = np.array(vals['attention_weights'])
        attn_weights_sum = attn_weights.sum()
        weighted_prob = np.sum(probs * (attn_weights / attn_weights_sum)) if attn_weights_sum > 0 else np.mean(probs)

        pred_label = 1 if weighted_prob >= 0.5 else 0
        aggregated.append({
            'filename': fname,
            'true_label': vals['true_label'],
            'weighted_prob': weighted_prob,
            'simple_avg_prob': np.mean(probs),
            'pred_label': pred_label,
            'num_bags': len(probs)
        })

    return aggregated


# ----------------------
# Save Model
# ----------------------
def save_model_checkpoint(model, fold_idx, fold_result, save_dir, args, is_best=False):
    save_dir = Path(save_dir) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'fold': fold_idx + 1,
        'model_state_dict': model.state_dict(),
        'accuracy': fold_result['accuracy'],
        'auc': fold_result['auc'],
        'config': {
            'k_folds': args.k_folds,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'bag_size': args.bag_size,
            'seed': args.seed,
            'model': 'ABMILGatedBase'
        }
    }

    filename = f"{'best_' if is_best else ''}model_fold{fold_idx+1}_auc{fold_result['auc']:.4f}.pt"
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)

    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"[✓] Model saved: {checkpoint_path.name} ({file_size_mb:.2f} MB)")
    return checkpoint_path


# ----------------------
# K-Fold CV
# ----------------------
def run_k_fold_cv(dataset, cv_splits, args, device):
    print(f"\nRunning {cv_splits['k_folds']}-Fold CV")
    print(f"Hyperparameters: lr={args.lr}, weight_decay={args.weight_decay}, max_grad_norm={args.max_grad_norm}")
    print("=" * 80)

    all_fold_results, all_predictions, all_predictions_simple, all_true_labels, saved_model_paths = [], [], [], [], []
    model_save_dir = Path(args.model_save_dir)
    config = ABMILGatedBaseConfig()

    def count_labels(bags):
        labels = [b[1].item() if torch.is_tensor(b[1]) else b[1] for b in bags]
        return dict(zip(*np.unique(labels, return_counts=True)))

    for fold_data in cv_splits['folds']:
        fold_idx = fold_data['fold'] - 1
        print(f"\n{'='*80}")
        print(f"Fold {fold_data['fold']}/{cv_splits['k_folds']}")
        print(f"{'='*80}")

        train_wsis, val_wsis, test_wsis = fold_data['train_wsis'], fold_data['val_wsis'], fold_data['test_wsis']
        train_bags = [dataset[i] for i in range(len(dataset)) if dataset.bags[i]['filename'] in train_wsis]
        val_bags   = [dataset[i] for i in range(len(dataset)) if dataset.bags[i]['filename'] in val_wsis]
        test_bags  = [dataset[i] for i in range(len(dataset)) if dataset.bags[i]['filename'] in test_wsis]

        print(f"Train: {len(train_bags)} bags, {count_labels(train_bags)}")
        print(f"Val:   {len(val_bags)} bags, {count_labels(val_bags)}")
        print(f"Test:  {len(test_bags)} bags, {count_labels(test_bags)}")

        # 모델 초기화
        model = ABMILModel(config).to(device)
        if fold_idx == 0:
            print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        model.apply(init_weights)
        
        # Optimizer 수정
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay  # 조정 가능
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=15, factor=0.5
        )
        
        # Early stopping을 AUC 기준으로 변경
        early_stopping = EarlyStopping(patience=30, min_delta=0.001, mode='max')

        # 학습 루프
        log_file = Path(args.model_save_dir) / f"train_log_fold{fold_idx+1}.txt"
        with open(log_file, "w") as f_log:
            best_val_auc = 0.0
            
            for epoch in range(args.epochs):
                train_loss, train_acc = train_one_epoch(
                    model, train_bags, optimizer, device, 
                    max_grad_norm=args.max_grad_norm
                )
                val_loss, val_acc, val_auc, val_prec, val_rec, val_f1 = validate_one_epoch(model, val_bags, device)
                
                scheduler.step(val_loss)
                
                # Best model 추적
                if val_auc > best_val_auc:
                    best_val_auc = val_auc

                log_str = (f"Epoch {epoch+1:3d}/{args.epochs} | "
                           f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                           f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                           f"AUC: {val_auc:.4f}, F1: {val_f1:.4f} | Best AUC: {best_val_auc:.4f}")
                
                if (epoch + 1) % 10 == 0 or epoch < 5:
                    print(log_str)
                f_log.write(log_str + "\n")

                # Early stopping (AUC 기준)
                if early_stopping(val_auc, model):
                    stop_str = f"Early stopping at epoch {epoch+1} (Best AUC: {best_val_auc:.4f})"
                    print(stop_str)
                    f_log.write(stop_str + "\n")
                    early_stopping.restore_best(model)
                    break

        # Test 평가
        test_results = evaluate_model(model, test_bags, device)
        fold_predictions = [r['weighted_prob'] for r in test_results]
        fold_predictions_simple = [r['simple_avg_prob'] for r in test_results]
        fold_true_labels = [r['true_label'] for r in test_results]
        fold_pred_labels = [r['pred_label'] for r in test_results]

        fold_accuracy = accuracy_score(fold_true_labels, fold_pred_labels)
        fold_auc = roc_auc_score(fold_true_labels, fold_predictions) if len(set(fold_true_labels)) > 1 else 0.5

        print(f"\n[Fold {fold_data['fold']} Test Results]")
        print(f"  Accuracy: {fold_accuracy:.4f}")
        print(f"  AUC: {fold_auc:.4f}")

        fold_result = {
            'fold': fold_data['fold'],
            'accuracy': fold_accuracy,
            'auc': fold_auc,
            'best_val_auc': best_val_auc,
            'test_wsis': test_wsis,
            'test_results': test_results
        }
        all_fold_results.append(fold_result)
        all_predictions.extend(fold_predictions)
        all_predictions_simple.extend(fold_predictions_simple)
        all_true_labels.extend(fold_true_labels)

        if args.save_model:
            saved_model_paths.append(str(save_model_checkpoint(model, fold_idx, fold_result, model_save_dir, args)))

    return all_fold_results, all_predictions, all_predictions_simple, all_true_labels, saved_model_paths


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_save_dir', type=str, required=True)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)  # 추가
    parser.add_argument('--max_grad_norm', type=float, default=1.0)  # 추가
    parser.add_argument('--bag_size', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv_split_file', type=str, default=None)
    parser.add_argument('--create_splits_only', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_best_only', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = ThyroidBagDataset(args.data_root, bag_size=args.bag_size)
    if len(dataset) < 10:
        raise ValueError("Need at least 10 bags!")

    cv_splits_dir = Path(args.model_save_dir) / "cv_splits"
    if args.cv_split_file and Path(args.cv_split_file).exists():
        cv_splits = load_cv_splits(args.cv_split_file)
    else:
        print("\nCreating new CV splits...")
        cv_splits = create_cv_splits(dataset, k_folds=args.k_folds, seed=args.seed)
        save_cv_splits(cv_splits, cv_splits_dir)
        if args.create_splits_only:
            print("\n[✓] CV splits created. Exiting.")
            return

    # K-Fold 학습
    fold_results, predictions, predictions_simple, true_labels, model_paths = \
        run_k_fold_cv(dataset, cv_splits, args, device)

    # 최종 결과
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    if len(set(true_labels)) > 1:
        eval_results = comprehensive_evaluation(true_labels, predictions)
        print(f"Overall AUC: {eval_results['auc']:.4f}")
        print(f"Overall Accuracy: {eval_results['accuracy']:.4f}")
        print(f"Overall F1: {eval_results['f1_score']:.4f}")
        
        fold_aucs = [f['auc'] for f in fold_results]
        print(f"\nPer-fold AUC: {fold_aucs}")
        print(f"Mean ± Std: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # 결과 저장
    results = {
        "hyperparameters": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm
        },
        "final": eval_results if len(set(true_labels)) > 1 else {},
        "folds": fold_results
    }

    results_path = Path(args.model_save_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Results saved: {results_path}")

    print("\n[✓] Training completed!")


if __name__ == "__main__":
    main()
