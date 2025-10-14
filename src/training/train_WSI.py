"""
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from collections import defaultdict
import warnings
import sys
import json

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)
sys.path.append(os.path.join(src_dir, 'models'))

from abmil import ABMILModel, ABMILGatedBaseConfig
from utils.datasets import ThyroidWSIDataset, set_seed
from utils.metrics import comprehensive_evaluation
from torch.utils.data import DataLoader


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
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def aggregate_wsi_predictions(wsi_dict, aggregation='mean', debug=False):
    """WSI 단위로 bag 예측을 aggregation"""
    wsi_probs = []
    wsi_labels = []
    wsi_preds = []
    wsi_names = []
    
    for wsi_name, data in wsi_dict.items():
        bag_probs = np.array(data['probs'])
        
        # Aggregation 방법
        if aggregation == 'mean':
            wsi_prob = np.mean(bag_probs)
        elif aggregation == 'max':
            wsi_prob = np.max(bag_probs)
        elif aggregation == 'median':
            wsi_prob = np.median(bag_probs)
        else:
            wsi_prob = np.mean(bag_probs)
        
        wsi_pred = 1 if wsi_prob > 0.5 else 0
        
        wsi_probs.append(wsi_prob)
        wsi_labels.append(data['label'])
        wsi_preds.append(wsi_pred)
        wsi_names.append(wsi_name)
        
        if debug and len(wsi_probs) <= 3:
            print(f"[DEBUG] WSI: {wsi_name}")
            print(f"  Bag probs: {bag_probs}")
            print(f"  Aggregated prob: {wsi_prob:.4f}")
            print(f"  Label: {data['label']}, Pred: {wsi_pred}")
    
    return wsi_probs, wsi_labels, wsi_preds, wsi_names


def train_one_epoch(model, dataloader, optimizer, device, debug=False):
    """Bag 단위 학습 (변경 없음)"""
    model.train()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    loss_fn = nn.CrossEntropyLoss()

    for batch_idx, (features, label, filename) in enumerate(dataloader):
        features = features.to(device)
        label = label.to(device)

        if debug and batch_idx == 0:
            print(f"[DEBUG] Train - Features shape: {features.shape}")
            print(f"[DEBUG] Train - Label: {label.item()}, Filename: {filename}")

        optimizer.zero_grad()
        results_dict, _ = model(h=features, loss_fn=loss_fn, label=label)
        loss = results_dict['loss']
        logits = results_dict['logits']

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
        
        all_probs.extend(probs.cpu().detach().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        if debug and batch_idx == 0:
            print(f"[DEBUG] Train - Loss: {loss.item():.4f}")
            print(f"[DEBUG] Train - Probs: {probs.cpu().detach().numpy()}")
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    
    if len(set(all_labels)) > 1:
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    else:
        accuracy, auc, precision, recall, f1 = 0.0, 0.5, 0.0, 0.0, 0.0
    
    return avg_loss, accuracy, auc, precision, recall, f1


def validate_one_epoch_wsi(model, dataloader, device, aggregation='mean', debug=False):
    """WSI 단위 검증"""
    model.eval()
    wsi_dict = defaultdict(lambda: {'probs': [], 'label': None})
    
    with torch.no_grad():
        for batch_idx, (features, label, filename) in enumerate(dataloader):
            features = features.to(device)
            label = label.to(device)
            
            # filename에서 WSI 이름 추출 (확장자 제거)
            if isinstance(filename, (list, tuple)):
                filename = filename[0]
            wsi_name = os.path.splitext(os.path.basename(filename))[0]
            
            results_dict, _ = model(h=features, loss_fn=None, label=None)
            logits = results_dict['logits']
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            wsi_dict[wsi_name]['probs'].append(probs.item())
            wsi_dict[wsi_name]['label'] = label.item()
            
            if debug and batch_idx == 0:
                print(f"[DEBUG] Val - WSI: {wsi_name}, Prob: {probs.item():.4f}")
    
    # WSI 단위로 aggregation
    wsi_probs, wsi_labels, wsi_preds, _ = aggregate_wsi_predictions(
        wsi_dict, aggregation=aggregation, debug=debug
    )
    
    if len(set(wsi_labels)) > 1:
        accuracy = accuracy_score(wsi_labels, wsi_preds)
        auc = roc_auc_score(wsi_labels, wsi_probs)
        precision = precision_score(wsi_labels, wsi_preds, zero_division=0)
        recall = recall_score(wsi_labels, wsi_preds, zero_division=0)
        f1 = f1_score(wsi_labels, wsi_preds, zero_division=0)
    else:
        accuracy, auc, precision, recall, f1 = 0.0, 0.5, 0.0, 0.0, 0.0
    
    return accuracy, auc, precision, recall, f1


def evaluate_model_wsi(model, dataloader, device, aggregation='mean', debug=False):
    """WSI 단위 테스트"""
    model.eval()
    wsi_dict = defaultdict(lambda: {'probs': [], 'label': None})
    
    with torch.no_grad():
        for batch_idx, (features, label, filename) in enumerate(dataloader):
            features = features.to(device)
            label = label.to(device)
            
            if isinstance(filename, (list, tuple)):
                filename = filename[0]
            wsi_name = os.path.splitext(os.path.basename(filename))[0]
            
            results_dict, _ = model(h=features, loss_fn=None, label=None)
            logits = results_dict['logits']
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            wsi_dict[wsi_name]['probs'].append(probs.item())
            wsi_dict[wsi_name]['label'] = label.item()
            
            if debug and batch_idx < 3:
                print(f"[DEBUG] Test - WSI: {wsi_name}, Prob: {probs.item():.4f}")
    
    wsi_probs, wsi_labels, wsi_preds, wsi_names = aggregate_wsi_predictions(
        wsi_dict, aggregation=aggregation, debug=debug
    )
    
    return wsi_probs, wsi_labels, wsi_preds, wsi_names


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
    
    if debug:
        print(f"[DEBUG] Loaded CV splits: {len(cv_splits['folds'])} folds")
    
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
            
            if debug and fold_data['fold'] == 1:
                print(f"[DEBUG] Fold 1 {split_name}: {len(file_paths)} files")
    
    return cv_splits


def run_k_fold_cv(cv_splits, args, device):
    print(f"\nRunning {len(cv_splits['folds'])}-Fold CV (WSI-level Evaluation)")
    print("=" * 60)

    all_fold_results = []
    all_predictions, all_true_labels = [], []
    saved_model_paths = []
    config = ABMILGatedBaseConfig()

    for fold_data in cv_splits['folds']:
        fold_idx = fold_data['fold'] - 1
        print(f"\nFold {fold_data['fold']}/{len(cv_splits['folds'])}")

        # Dataset 생성
        train_dataset = ThyroidWSIDataset(fold_data['train_wsis_paths'], 
                                          bag_size=args.bag_size, 
                                          use_variance=False)
        val_dataset = ThyroidWSIDataset(fold_data['val_wsis_paths'], 
                                        bag_size=args.bag_size, 
                                        use_variance=False)
        test_dataset = ThyroidWSIDataset(fold_data['test_wsis_paths'], 
                                         bag_size=args.bag_size, 
                                         use_variance=False)

        if args.debug:
            print(f"[DEBUG] Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 모델 초기화
        model = ABMILModel(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        early_stopping = EarlyStopping(patience=30, min_delta=0.001)

        # Train/Val
        best_val_metrics = {}
        best_train_metrics = {}
        
        for epoch in range(args.epochs):
            # Bag 단위 학습
            train_loss, train_acc, train_auc, train_prec, train_rec, train_f1 = \
                train_one_epoch(model, train_loader, optimizer, device, 
                              debug=args.debug and epoch == 0)
            
            # WSI 단위 검증
            val_acc, val_auc, val_prec, val_rec, val_f1 = \
                validate_one_epoch_wsi(model, val_loader, device, 
                                      aggregation='mean',
                                      debug=args.debug and epoch == 0)
            
            scheduler.step(1 - val_auc)  # AUC 기준

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train AUC={train_auc:.4f}, Val AUC(WSI)={val_auc:.4f}")

            if val_auc > best_val_metrics.get("auc", 0):
                best_val_metrics = {
                    "accuracy": val_acc,
                    "auc": val_auc,
                    "precision": val_prec,
                    "recall": val_rec,
                    "f1": val_f1
                }
                best_train_metrics = {
                    "loss": train_loss,
                    "accuracy": train_acc,
                    "auc": train_auc,
                    "precision": train_prec,
                    "recall": train_rec,
                    "f1": train_f1
                }

            if early_stopping(val_auc, model):
                print(f"Early stopping at epoch {epoch}")
                early_stopping.restore_best(model)
                break

        # WSI 단위 테스트
        test_probs, test_labels, test_preds, test_wsi_names = \
            evaluate_model_wsi(model, test_loader, device, 
                             aggregation='mean',
                             debug=args.debug)

        test_metrics = {
            "accuracy": accuracy_score(test_labels, test_preds),
            "auc": roc_auc_score(test_labels, test_probs) if len(set(test_labels)) > 1 else 0.5,
            "precision": precision_score(test_labels, test_preds, zero_division=0),
            "recall": recall_score(test_labels, test_preds, zero_division=0),
            "f1": f1_score(test_labels, test_preds, zero_division=0)
        }

        fold_result = {
            "fold": fold_data['fold'],
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics
        }

        all_fold_results.append(fold_result)
        all_predictions.extend(test_probs)
        all_true_labels.extend(test_labels)

        print(f"Fold {fold_data['fold']} Test (WSI-level): AUC={test_metrics['auc']:.4f}, Acc={test_metrics['accuracy']:.4f}")

        if args.debug:
            print(f"[DEBUG] Fold {fold_data['fold']} - WSI-level Results:")
            print(f"  Train (Bag): AUC={best_train_metrics['auc']:.4f}")
            print(f"  Val (WSI): AUC={best_val_metrics['auc']:.4f}")
            print(f"  Test (WSI): AUC={test_metrics['auc']:.4f}")

        # 모델 저장
        if args.save_model:
            if args.save_best_only:
                if fold_idx == 0 or test_metrics['auc'] > max(r['test_metrics']['auc'] for r in all_fold_results[:-1]):
                    path = save_model_checkpoint(model, fold_idx, fold_result, args.model_save_dir, args, is_best=True)
                    saved_model_paths.append(path)
            else:
                path = save_model_checkpoint(model, fold_idx, fold_result, args.model_save_dir, args)
                saved_model_paths.append(path)

    return all_fold_results, all_predictions, all_true_labels, saved_model_paths


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

    if args.debug:
        print("[DEBUG] WSI-level Evaluation Mode")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cv_splits = load_cv_splits_with_paths(args.cv_split_file, args.data_root, debug=args.debug)
    
    fold_results, predictions, true_labels, model_paths = \
        run_k_fold_cv(cv_splits, args, device)

    # 최종 결과
    results = {}
    if len(set(true_labels)) > 1:
        eval_results = comprehensive_evaluation(true_labels, predictions)
        results["final"] = eval_results
        print(f"\n=== Final Results (WSI-level) ===")
        print(f"AUC: {eval_results['auc']:.4f}")
        print(f"Accuracy: {eval_results['accuracy']:.4f}")

    results["folds"] = fold_results

    # 결과 저장
    results_path = Path(args.model_save_dir) / "results_wsi_level.json"
    with open(results_path, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"\n[✓] Results saved: {results_path}")

    if args.save_model and model_paths:
        print(f"[✓] Saved {len(model_paths)} model checkpoints")

    print("\n[✓] Training completed!")


if __name__ == "__main__":
    main()
