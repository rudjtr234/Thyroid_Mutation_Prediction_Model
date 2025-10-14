"""

# 기존 train.py (cv_split 새로 생성)
CUDA_VISIBLE_DEVICES=3 \
python train.py \
  --data_root /data/143/member/jks/Thyroid_Mutation_dataset/embeddings \
  --model_save_dir /home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model/outputs/Thyroid_prediction_model_v0.1.0 \
  --k_folds 5 \
  --epochs 100 \
  --lr 1e-5 \
  --bag_size 2000 \
  --seed 42 \
  --save_model \
  --save_best_only

------------------------------------------------------------------------------------------------------------------------

# train_v2.py (이미 생성된 cv_split.json 사용)
CUDA_VISIBLE_DEVICES=3 \
python train_v2.py \
  --data_root /data/143/member/jks/Thyroid_Mutation_dataset/embeddings \
  --model_save_dir /home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model/outputs/Thyroid_prediction_model_v0.1.0 \
  --k_folds 5 \
  --epochs 100 \
  --lr 1e-5 \
  --bag_size 2000 \
  --seed 42 \
  --save_model \
  --save_best_only \
  --cv_split_file /home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model/outputs/Thyroid_prediction_model_v0.1.0/cv_splits/cv_splits_k5_seed42.json

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

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abmil import ABMILModel, ABMILGatedBaseConfig
from utils.datasets import ThyroidBagDataset, set_seed
from utils.cv_splits import create_cv_splits, save_cv_splits, load_cv_splits
from utils.metrics import comprehensive_evaluation


# ----------------------
# EarlyStopping
# ----------------------
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


# ----------------------
# Train / Val
# ----------------------
def train_one_epoch(model, train_bags, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    loss_fn = nn.CrossEntropyLoss()
    
    for i, (features, label, bag_id, _) in enumerate(train_bags):
        features = features.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        logits, loss = model(h=features, loss_fn=loss_fn, label=label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        
        if i < 5:
            print(f"[DEBUG][Train] BagID={bag_id}, Label={label.item()}, "
                  f"Pred={preds.item()}, "
                  f"Logits={logits.detach().cpu().numpy()}")
    
    avg_loss = total_loss / len(train_bags) if len(train_bags) > 0 else 0
    accuracy = accuracy_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    return avg_loss, accuracy


def validate_one_epoch(model, val_bags, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, (features, label, bag_id, _) in enumerate(val_bags):
            features = features.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            
            logits, loss = model(h=features, loss_fn=loss_fn, label=label)
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            if i < 5:
                print(f"[DEBUG][Val] BagID={bag_id}, Label={label.item()}, "
                      f"Pred={preds.item()}, "
                      f"Prob(1)={probs.item():.4f}, "
                      f"Logits={logits.detach().cpu().numpy()}")
    
    avg_loss = total_loss / len(val_bags) if len(val_bags) > 0 else 0
    
    if len(set(all_labels)) > 1:
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
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

    # --- WSI 단위 aggregation ---
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
            'bag_size': args.bag_size,
            'seed': args.seed,
            'model': 'ABMILGatedBase'
        }
    }

    filename = f"{'best_' if is_best else ''}model_fold{fold_idx+1}_auc{fold_result['auc']:.4f}.pt"
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)

    print(f"[✓] Model saved: {checkpoint_path} ({checkpoint_path.stat().st_size/1024/1024:.2f} MB)")
    return checkpoint_path


# ----------------------
# K-Fold CV (Debug 포함)
# ----------------------
def run_k_fold_cv(dataset, cv_splits, args, device):
    print(f"\nRunning {cv_splits['k_folds']}-Fold CV (8:1:1 split)")
    print("=" * 60)

    all_fold_results, all_predictions, all_predictions_simple, all_true_labels, saved_model_paths = [], [], [], [], []
    model_save_dir = Path(args.model_save_dir)
    config = ABMILGatedBaseConfig()

    # ✅ label 분포 카운터 함수
    def count_labels(bags):
        labels = [b[1].item() if torch.is_tensor(b[1]) else b[1] for b in bags]
        return dict(zip(*np.unique(labels, return_counts=True)))

    for fold_data in cv_splits['folds']:
        fold_idx = fold_data['fold'] - 1
        print(f"\nFold {fold_data['fold']}/{cv_splits['k_folds']}")

        train_wsis, val_wsis, test_wsis = fold_data['train_wsis'], fold_data['val_wsis'], fold_data['test_wsis']
        train_bags = [dataset[i] for i in range(len(dataset)) if dataset.bags[i]['filename'] in train_wsis]
        val_bags   = [dataset[i] for i in range(len(dataset)) if dataset.bags[i]['filename'] in val_wsis]
        test_bags  = [dataset[i] for i in range(len(dataset)) if dataset.bags[i]['filename'] in test_wsis]

        # ✅ Debug: 데이터 분포 출력
        print(f"  Train size={len(train_bags)}, label dist={count_labels(train_bags)}")
        print(f"  Val   size={len(val_bags)}, label dist={count_labels(val_bags)}")
        print(f"  Test  size={len(test_bags)}, label dist={count_labels(test_bags)}")
        if len(val_bags) > 0:
            print(f"  [Sample Val Bag] filename={val_bags[0][3]}, label={val_bags[0][1].item()}")

        # --- 모델 초기화 ---
        model = ABMILModel(config).to(device)
        if fold_idx == 0:
            print("Model architecture:")
            print(model)
            print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        early_stopping = EarlyStopping(patience=30, min_delta=0.001)

        # ✅ 로그 파일 준비
        log_file = Path(args.model_save_dir) / f"train_log_fold{fold_idx+1}.txt"
        with open(log_file, "w") as f_log:

            for epoch in range(args.epochs):
                train_loss, train_acc = train_one_epoch(model, train_bags, optimizer, device)
                val_loss, val_acc, val_auc, val_prec, val_rec, val_f1 = validate_one_epoch(model, val_bags, device)
                scheduler.step(val_loss)

                log_str = (f"[Fold {fold_idx+1}] Epoch {epoch+1}/{args.epochs} "
                           f"- Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
                           f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                           f"AUC: {val_auc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
                print(log_str)
                f_log.write(log_str + "\n")

                if early_stopping(val_acc, model):
                    stop_str = f"Early stopping at epoch {epoch+1}"
                    print(stop_str)
                    f_log.write(stop_str + "\n")
                    early_stopping.restore_best(model)
                    break

        # --- Test ---
        test_results = evaluate_model(model, test_bags, device)
        fold_predictions = [r['weighted_prob'] for r in test_results]
        fold_predictions_simple = [r['simple_avg_prob'] for r in test_results]
        fold_true_labels = [r['true_label'] for r in test_results]
        fold_pred_labels = [r['pred_label'] for r in test_results]

        fold_accuracy = accuracy_score(fold_true_labels, fold_pred_labels)
        fold_auc = roc_auc_score(fold_true_labels, fold_predictions) if len(set(fold_true_labels)) > 1 else 0.5

        fold_result = {
            'fold': fold_data['fold'],
            'accuracy': fold_accuracy,
            'auc': fold_auc,
            'test_wsis': test_wsis,
            'test_results': test_results,
            'debug': {   # ✅ debug 정보도 저장
                'train_label_dist': count_labels(train_bags),
                'val_label_dist': count_labels(val_bags),
                'test_label_dist': count_labels(test_bags)
            }
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
    parser.add_argument('--data_root', type=str, required=True, help='Data directory path')
    parser.add_argument('--model_save_dir', type=str, required=True, help='Output directory path')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bag_size', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv_split_file', type=str, default=None)
    parser.add_argument('--create_splits_only', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_best_only', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ThyroidBagDataset(args.data_root, bag_size=args.bag_size)
    if len(dataset) < 10:
        raise ValueError("Need at least 10 bags!")

    # CV splits 저장 위치를 outputs/ 밑으로 통일
    cv_splits_dir = Path(args.model_save_dir) / "cv_splits"
    if args.cv_split_file and Path(args.cv_split_file).exists():
        cv_splits = load_cv_splits(args.cv_split_file)
    else:
        print("\nCreating new CV splits...")
        cv_splits = create_cv_splits(dataset, k_folds=args.k_folds, seed=args.seed)
        save_cv_splits(cv_splits, cv_splits_dir)
        if args.create_splits_only:
            print("\n[✓] CV splits created. Exiting without training.")
            return

    # K-Fold 학습 실행
    fold_results, predictions, predictions_simple, true_labels, model_paths = \
        run_k_fold_cv(dataset, cv_splits, args, device)

    # 최종 결과 출력 및 저장
    results = {}
    if len(set(true_labels)) > 1:
        eval_results = comprehensive_evaluation(true_labels, predictions)
        results["final"] = eval_results
        print(f"\nFinal Results: AUC={eval_results['auc']:.4f}, Acc={eval_results['accuracy']:.4f}")

    results["folds"] = fold_results

    # results.json 저장
    results_path = Path(args.model_save_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Results saved: {results_path}")

    if args.save_model and model_paths:
        print(f"\n[✓] Saved {len(model_paths)} model checkpoints.")

    print("\n[✓] Training completed!")


if __name__ == "__main__":
    main()
