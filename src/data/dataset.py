import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class ThyroidWSIDataset(Dataset):
    def __init__(self, root, split_file, split):
        self.root = root
        with open(split_file, "r") as f:
            splits = json.load(f)
        self.slide_ids = splits[split]

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        # meta / nonmeta 구분 없는 통일된 naming
        feat_path = [f for f in os.listdir(self.root) if f.endswith(f"{slide_id}.npy")][0]
        label_path = f"label_{slide_id}.json"

        features = np.load(os.path.join(self.root, feat_path))
        with open(os.path.join(self.root, label_path), "r") as f:
            label = json.load(f)["label"]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
