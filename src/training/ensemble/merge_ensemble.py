# -*- coding: utf-8 -*-
"""
Merge ensemble checkpoints by simple weight averaging and evaluate on test set.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =========================
# Path setup
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(training_dir)
sys.path.insert(0, src_dir)

evaluation_dir = os.path.join(src_dir, "evaluation")
sys.path.insert(0, evaluation_dir)


# =========================
# Imports
# =========================
from models.abmil import ABMILModel, ABMILGatedBaseConfig
from utils.datasets import set_seed
from metric import compute_metrics_with_confusion


# =========================
# Dataset (JSON 기반)
# =========================
class EnsembleDataset(Dataset):
    def __init__(self, positive_files: List[str], negative_files: List[str],
                 pos_dir: str, neg_dir: str, bag_size: int = None):
        self.wsi_list = []
        self.bag_size = bag_size

        for filename in positive_files:
            filepath = os.path.join(pos_dir, filename)
            if os.path.exists(filepath):
                self.wsi_list.append({
                    "filepath": filepath,
                    "label": 1,
                    "filename": filename
                })
            else:
                print(f"Warning: Positive file not found: {filepath}")

        for filename in negative_files:
            filepath = os.path.join(neg_dir, filename)
            if os.path.exists(filepath):
                self.wsi_list.append({
                    "filepath": filepath,
                    "label": 0,
                    "filename": filename
                })
            else:
                print(f"Warning: Negative file not found: {filepath}")

        labels = [wsi["label"] for wsi in self.wsi_list]
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
        features = np.load(wsi["filepath"])
        features = self._select_tiles(features)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(wsi["label"], dtype=torch.long)
        return features, label, wsi["filename"]


def load_test_split(test_json_path: str, data_root: str = None):
    with open(test_json_path, "r") as f:
        test_data = json.load(f)

    if isinstance(test_data["positive"], dict):
        test_pos_files = test_data["positive"]["files"]
        test_pos_dir = test_data["positive"]["dir"]
    else:
        test_pos_files = test_data["positive"]
        if data_root is None:
            raise ValueError("data_root is required when test_json has no positive.dir")
        test_pos_dir = test_data.get("positive_dir", data_root + "/meta/npy")

    if isinstance(test_data["negative"], dict):
        test_neg_files = test_data["negative"]["files"]
        test_neg_dir = test_data["negative"]["dir"]
    else:
        test_neg_files = test_data["negative"]
        if data_root is None:
            raise ValueError("data_root is required when test_json has no negative.dir")
        test_neg_dir = test_data.get("negative_dir", data_root + "/non-meta/npy")

    return test_pos_files, test_neg_files, test_pos_dir, test_neg_dir


def average_state_dicts(state_dicts: List[dict]) -> dict:
    if not state_dicts:
        raise ValueError("No state_dicts provided for averaging")

    keys = state_dicts[0].keys()
    for idx, sd in enumerate(state_dicts[1:], start=1):
        if sd.keys() != keys:
            missing = keys - sd.keys()
            extra = sd.keys() - keys
            raise ValueError(
                f"Checkpoint {idx} key mismatch. Missing: {sorted(missing)} Extra: {sorted(extra)}"
            )

    avg_state = {}
    for key in keys:
        base_tensor = state_dicts[0][key]
        if not torch.is_floating_point(base_tensor):
            avg_state[key] = base_tensor
            continue

        total = base_tensor.float().clone()
        for sd in state_dicts[1:]:
            total += sd[key].float()
        avg_state[key] = (total / len(state_dicts)).to(dtype=base_tensor.dtype)

    return avg_state


def load_checkpoint_state(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def evaluate_model_on_test(model, test_loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for features, label, _ in test_loader:
            features = features.to(device)
            label = label.to(device)

            results_dict, _ = model(h=features, loss_fn=None, label=None)
            logits = results_dict["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

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

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Merge ensemble checkpoints and evaluate on test set")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoints", nargs="+", help="List of checkpoint paths")
    group.add_argument("--checkpoints_glob", type=str, help="Glob pattern for checkpoints")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save merged checkpoint")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test_set.json")
    parser.add_argument("--data_root", type=str, default=None, help="Root directory if test_json has no dirs")
    parser.add_argument("--bag_size", type=int, default=None, help="Bag size for test (default: None = full)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--skip_save", action="store_true", help="Skip saving merged checkpoint")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.checkpoints_glob:
        ckpt_paths = sorted(glob.glob(args.checkpoints_glob))
    else:
        ckpt_paths = args.checkpoints

    if len(ckpt_paths) == 0:
        raise ValueError("No checkpoints found")

    print(f"Loading {len(ckpt_paths)} checkpoints for merging")
    state_dicts = [load_checkpoint_state(p) for p in ckpt_paths]
    merged_state = average_state_dicts(state_dicts)

    if not args.skip_save:
        if args.output_path is None:
            raise ValueError("--output_path is required unless --skip_save is set")
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "merge_type": "simple_average",
                "num_models": len(ckpt_paths),
                "source_checkpoints": ckpt_paths,
                "model_state_dict": merged_state,
            },
            output_path
        )
        print(f"[✓] Merged checkpoint saved: {output_path}")

    # Evaluation
    test_pos_files, test_neg_files, test_pos_dir, test_neg_dir = load_test_split(
        args.test_json, data_root=args.data_root
    )

    test_dataset = EnsembleDataset(
        positive_files=test_pos_files,
        negative_files=test_neg_files,
        pos_dir=test_pos_dir,
        neg_dir=test_neg_dir,
        bag_size=args.bag_size
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    config = ABMILGatedBaseConfig()
    model = ABMILModel(config).to(device)
    model.load_state_dict(merged_state)

    metrics = evaluate_model_on_test(model, test_loader, device, threshold=args.threshold)

    print("\nMerged Model Test Metrics")
    print("-" * 60)
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1:          {metrics['f1']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"PPV:         {metrics['ppv']:.4f}")
    print(f"NPV:         {metrics['npv']:.4f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
