import os
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _infer_label(filepath: str) -> int:
    """Infer WSI label from its path using case-insensitive directory checks."""
    normalized = filepath.replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]

    for part in parts:
        if "nonmeta" in part or "non_meta" in part:
            return 0

    for part in parts:
        if "meta" in part and "nonmeta" not in part and "non_meta" not in part:
            return 1

    return 0


class WSIFullDataset(Dataset):
    """Dataset that loads the entire NPY file per WSI (no bag sampling)."""

    def __init__(self, wsi_files: List[str], validate_files: bool = True) -> None:
        self.wsi_list = []
        failed_files = []

        for filepath in wsi_files:
            if validate_files and not os.path.exists(filepath):
                failed_files.append(filepath)
                continue

            label = _infer_label(filepath)
            self.wsi_list.append(
                {
                    "filepath": filepath,
                    "label": label,
                    "filename": os.path.basename(filepath),
                }
            )

        if failed_files:
            preview = ", ".join(failed_files[:3])
            suffix = "..." if len(failed_files) > 3 else ""
            print(
                f"[WSIFullDataset] Skipped {len(failed_files)} missing files "
                f"(e.g., {preview}{suffix})"
            )

        labels = [wsi["label"] for wsi in self.wsi_list]
        print(
            f"[WSIFullDataset] {len(self.wsi_list)} WSIs "
            f"(BRAF+: {sum(labels)}, BRAF-: {len(labels) - sum(labels)})"
        )

    def __len__(self) -> int:
        return len(self.wsi_list)

    def __getitem__(self, idx: int):
        wsi = self.wsi_list[idx]
        features = np.load(wsi["filepath"])
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(wsi["label"], dtype=torch.long)
        return features, label, wsi["filename"]


def _resolve_path(filename: str, data_root: str) -> str:
    if os.path.exists(filename):
        return filename

    candidates = [
        os.path.join(data_root, filename),
        os.path.join(data_root, "meta", filename),
        os.path.join(data_root, "nonmeta", filename),
        os.path.join(data_root, "npy", filename),
        os.path.join(data_root, "embeddings", filename),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return os.path.join(data_root, filename)


def load_json_splits_wsi(cv_split_file: str, data_root: str) -> dict:
    """
    Load K-fold splits and resolve file paths with data_root when needed.
    Expected keys in JSON: folds -> [{train_wsis, val_wsis, test_wsis, fold}, ...]
    """
    with open(cv_split_file, "r") as f:
        cv_splits = json.load(f)

    for fold in cv_splits.get("folds", []):
        for split_key in ("train_wsis", "val_wsis", "test_wsis"):
            files = fold.get(split_key, [])
            resolved = [
                _resolve_path(name, data_root)
                for name in files
            ]
            fold[f"{split_key}_paths"] = resolved

    return cv_splits
