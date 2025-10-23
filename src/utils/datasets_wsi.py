import os
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _infer_label(filepath: str) -> int:
    """Infer WSI label from its path using case-insensitive directory checks."""
    normalized = filepath.replace("\\", "/").lower()
    path_parts = [
        part for part in normalized.split("/") if part
    ]
    for part in path_parts:
        if part.startswith("nonmeta"):
            return 0
    for part in path_parts:
        if part.startswith("meta"):
            return 1
    return 0


class WSIBagDataset(Dataset):
    """Dataset that samples multiple bags per WSI."""

    def __init__(
        self,
        wsi_files: List[str],
        bag_size: int = 2000,
        use_variance: bool = False,
        bags_per_wsi: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            wsi_files: List of paths to .npy files containing WSI embeddings.
            bag_size: Number of tiles to sample per bag.
            use_variance: If True, select highest-variance tiles instead of random.
            bags_per_wsi: How many bags to generate per WSI.
            seed: Optional seed to make bag sampling reproducible.
        """
        self.wsi_list: List[dict] = []
        self.bag_size = bag_size
        self.use_variance = use_variance
        self.bags_per_wsi = max(1, int(bags_per_wsi))
        self.seed = seed

        for filepath in wsi_files:
            label = _infer_label(filepath)
            self.wsi_list.append(
                {
                    "filepath": filepath,
                    "label": label,
                    "filename": os.path.basename(filepath),
                }
            )

        self._insufficient_tile_wsis: List[str] = []
        self.bag_indices: List[List[np.ndarray]] = []

        for idx, wsi in enumerate(self.wsi_list):
            arr = np.load(wsi["filepath"], mmap_mode="r")
            num_tiles = arr.shape[0]

            if num_tiles == 0:
                raise ValueError(f"{wsi['filepath']} has no tiles.")

            if num_tiles <= self.bag_size:
                indices = np.arange(num_tiles)
                self.bag_indices.append([indices for _ in range(self.bags_per_wsi)])
                if num_tiles < self.bag_size and wsi["filename"] not in self._insufficient_tile_wsis:
                    self._insufficient_tile_wsis.append(wsi["filename"])
                del arr
                continue

            rng = (
                np.random.default_rng(self.seed + idx)
                if self.seed is not None
                else np.random.default_rng()
            )

            if self.use_variance:
                variances = np.var(arr, axis=1)
                perm = np.argsort(variances)[::-1]
            else:
                perm = rng.permutation(num_tiles)

            if num_tiles >= self.bag_size * self.bags_per_wsi:
                bag_list = [
                    perm[i * self.bag_size : (i + 1) * self.bag_size]
                    for i in range(self.bags_per_wsi)
                ]
            else:
                repeats = int(np.ceil(self.bag_size * self.bags_per_wsi / num_tiles))
                expanded = np.tile(perm, repeats + 1)
                bag_list = []
                for i in range(self.bags_per_wsi):
                    start = i * self.bag_size
                    end = start + self.bag_size
                    if end > len(expanded):
                        extra = rng.permutation(num_tiles)
                        expanded = np.concatenate([expanded, extra])
                    bag_list.append(expanded[start:end])
                if wsi["filename"] not in self._insufficient_tile_wsis:
                    self._insufficient_tile_wsis.append(wsi["filename"])

            self.bag_indices.append(bag_list)
            del arr

        # Mapping from dataset idx -> (wsi_idx, bag_repeat_idx)
        self.index_map: List[Tuple[int, int]] = [
            (wsi_idx, bag_idx)
            for wsi_idx in range(len(self.wsi_list))
            for bag_idx in range(self.bags_per_wsi)
        ]

        labels = [wsi["label"] for wsi in self.wsi_list]
        print(
            f"Dataset: {len(self.wsi_list)} WSIs (BRAF+: {sum(labels)}, "
            f"BRAF-: {len(labels) - sum(labels)})"
        )
        if self.bags_per_wsi > 1:
            print(
                f"         generating {self.bags_per_wsi} bags per WSI "
                f"-> total {len(self)} bags"
            )
        if self._insufficient_tile_wsis:
            preview = ", ".join(self._insufficient_tile_wsis[:3])
            suffix = "..." if len(self._insufficient_tile_wsis) > 3 else ""
            print(
                "         warning: insufficient tiles for disjoint sampling in "
                f"{len(self._insufficient_tile_wsis)} WSI(s) (e.g., {preview}{suffix})"
            )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        wsi_idx, bag_idx = self.index_map[idx]
        wsi = self.wsi_list[wsi_idx]

        features = np.load(wsi["filepath"])
        indices = self.bag_indices[wsi_idx][bag_idx]
        if len(features) <= self.bag_size:
            selected = features
        else:
            selected = features[indices]

        features = torch.tensor(selected, dtype=torch.float32)
        label = torch.tensor(wsi["label"], dtype=torch.long)
        return features, label, wsi["filename"]


class WSIFullDataset(Dataset):
    """
    Dataset that loads the entire NPY file for each WSI (no bag sampling).
    Used for full WSI-level training.
    """

    def __init__(self, wsi_files: List[str]) -> None:
        """
        Args:
            wsi_files: List of paths to .npy files containing WSI embeddings.
        """
        self.wsi_list: List[dict] = []

        for filepath in wsi_files:
            # Label detection based on path
            label = _infer_label(filepath)
            self.wsi_list.append(
                {
                    "filepath": filepath,
                    "label": label,
                    "filename": os.path.basename(filepath),
                }
            )

        # Count labels for dataset summary
        labels = [wsi["label"] for wsi in self.wsi_list]
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        print(
            f"WSIFullDataset: {len(self.wsi_list)} WSIs "
            f"(BRAF+: {pos_count}, BRAF-: {neg_count})"
        )

    def __len__(self) -> int:
        return len(self.wsi_list)

    def __getitem__(self, idx: int):
        """
        Returns:
            features: Tensor of shape (N, D) where N is the number of tiles
            label: Integer label (0 or 1)
            filename: String filename of the WSI
        """
        wsi = self.wsi_list[idx]
        
        # Load entire NPY file
        features = np.load(wsi["filepath"])
        
        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(wsi["label"], dtype=torch.long)
        
        return features, label, wsi["filename"]


def load_json_splits_wsi(json_path: str, data_root: str) -> dict:
    """
    Load split json and expand into lists of file paths.
    
    Args:
        json_path: Path to the CV split JSON file
        data_root: Root directory containing 'meta' and 'nonmeta' subdirectories
        
    Returns:
        Dictionary with fold information and resolved file paths
    """
    with open(json_path, "r") as f:
        split_data = json.load(f)

    data_root = os.path.abspath(data_root)

    def iter_split_entries(split_section) -> Iterable[Tuple[str, Optional[str]]]:
        """
        Yield (filename, preferred_subdir) pairs for a split section which may be a list,
        dict keyed by subdir/label, or a single string.
        """
        if isinstance(split_section, dict):
            for subdir_key, filenames in split_section.items():
                if isinstance(filenames, (list, tuple)):
                    for name in filenames:
                        yield name, subdir_key
                elif isinstance(filenames, str):
                    yield filenames, subdir_key
                else:
                    print(f"Warning: Unsupported entry type {type(filenames)} in split section {subdir_key}")
        elif isinstance(split_section, (list, tuple)):
            for name in split_section:
                yield name, None
        elif isinstance(split_section, str):
            yield split_section, None
        elif split_section is None:
            return
        else:
            print(f"Warning: Unsupported split section type {type(split_section)} encountered.")

    def resolve_wsi_path(filename: str, preferred_subdir: Optional[str]) -> Optional[str]:
        if not filename:
            return None

        filename = filename.strip()
        if not filename:
            return None

        if os.path.isabs(filename) and os.path.exists(filename):
            return filename

        norm_filename = filename.replace("\\", "/")
        basename = os.path.basename(norm_filename)

        candidates = []
        if preferred_subdir:
            cleaned_subdir = preferred_subdir.strip("/\\")
            if cleaned_subdir:
                candidates.append(os.path.join(data_root, cleaned_subdir, filename))
                candidates.append(os.path.join(data_root, cleaned_subdir, basename))

        # original relative path attempts
        candidates.append(os.path.join(data_root, norm_filename))
        candidates.append(os.path.join(data_root, filename))
        candidates.append(os.path.join(data_root, basename))

        # common subdirectory conventions
        common_subdirs = ["meta", "nonmeta", "Meta", "Nonmeta", "non_meta", "Meta_WSI", "Nonmeta_WSI"]
        for subdir in common_subdirs:
            candidates.append(os.path.join(data_root, subdir, filename))
            candidates.append(os.path.join(data_root, subdir, basename))

        # remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for path in candidates:
            norm_path = os.path.normpath(path)
            if norm_path not in seen:
                seen.add(norm_path)
                unique_candidates.append(norm_path)

        for path in unique_candidates:
            if os.path.exists(path):
                return path

        # fallback: search by basename anywhere under data_root (avoids repeated os.walk)
        if basename:
            matches = list(Path(data_root).rglob(basename))
            if matches:
                return str(matches[0])

        return None

    fold_datasets: list = []
    for fold_info in split_data["folds"]:
        fold_idx = fold_info["fold"]
        fold_entry = {"fold": fold_idx}
        unresolved: List[str] = []
        
        for split_name in ["train_wsis", "val_wsis", "test_wsis"]:
            resolved_paths: List[str] = []
            for filename, preferred_subdir in iter_split_entries(fold_info.get(split_name, [])):
                resolved = resolve_wsi_path(filename, preferred_subdir)
                if resolved:
                    resolved_paths.append(resolved)
                else:
                    unresolved.append(f"{split_name}:{filename}")
            
            fold_entry[f"{split_name}_paths"] = resolved_paths
        
        if unresolved:
            preview = ", ".join(unresolved[:5])
            suffix = "..." if len(unresolved) > 5 else ""
            print(
                f"Warning: {len(unresolved)} item(s) could not be resolved in fold {fold_idx}: {preview}{suffix}"
            )

        fold_datasets.append(fold_entry)

    return {"folds": fold_datasets}
