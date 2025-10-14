import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class ThyroidWSIDataset(Dataset):
    """
    WSI 단위 Dataset
    - JSON으로 미리 정의된 split 사용
    """
    def __init__(self, wsi_files, bag_size=2000, use_variance=False):
        """
        Args:
            wsi_files: WSI 파일 경로 리스트
            bag_size: 각 WSI에서 사용할 타일 개수
            use_variance: True면 분산 기준, False면 랜덤 (기본)
        """
        self.wsi_list = []
        self.bag_size = bag_size
        self.use_variance = use_variance
        
        for filepath in wsi_files:
            if '/meta/' in filepath or '\\meta\\' in filepath:
                label = 1  # BRAF+
            else:
                label = 0  # BRAF-
            
            self.wsi_list.append({
                'filepath': filepath,
                'label': label,
                'filename': os.path.basename(filepath)
            })
        
        labels = [wsi['label'] for wsi in self.wsi_list]
        print(f"Dataset: {len(self.wsi_list)} WSIs (BRAF+: {sum(labels)}, BRAF-: {len(labels) - sum(labels)})")
    
    def _select_tiles(self, features):
        """타일 선택: Random (use_variance=False)"""
        if len(features) <= self.bag_size:
            return features
        
        if self.use_variance:
            variances = np.var(features, axis=1)
            top_k_indices = np.argsort(variances)[::-1][:self.bag_size]
            return features[top_k_indices]
        else:
            # Random Sampling (기존 방식과 동일)
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


def load_json_splits(json_path, data_root):
    """
    JSON 파일에서 미리 정의된 K-Fold split 로드
    """
    with open(json_path, 'r') as f:
        split_data = json.load(f)
    
    fold_datasets = []
    
    for fold_info in split_data['folds']:
        fold_idx = fold_info['fold']
        
        # Train WSI 파일 경로 생성
        train_files = []
        for filename in fold_info['train_wsis']:
            # meta 또는 nonmeta 폴더에서 찾기
            meta_path = os.path.join(data_root, 'meta', filename)
            nonmeta_path = os.path.join(data_root, 'nonmeta', filename)
            
            if os.path.exists(meta_path):
                train_files.append(meta_path)
            elif os.path.exists(nonmeta_path):
                train_files.append(nonmeta_path)
            else:
                print(f"Warning: {filename} not found")
        
        # Val WSI 파일 경로 생성
        val_files = []
        for filename in fold_info['val_wsis']:
            meta_path = os.path.join(data_root, 'meta', filename)
            nonmeta_path = os.path.join(data_root, 'nonmeta', filename)
            
            if os.path.exists(meta_path):
                val_files.append(meta_path)
            elif os.path.exists(nonmeta_path):
                val_files.append(nonmeta_path)
        
        # Test WSI 파일 경로 생성
        test_files = []
        for filename in fold_info['test_wsis']:
            meta_path = os.path.join(data_root, 'meta', filename)
            nonmeta_path = os.path.join(data_root, 'nonmeta', filename)
            
            if os.path.exists(meta_path):
                test_files.append(meta_path)
            elif os.path.exists(nonmeta_path):
                test_files.append(nonmeta_path)
        
        # Dataset 생성
        train_dataset = ThyroidWSIDataset(train_files, bag_size=2000, use_variance=False)
        val_dataset = ThyroidWSIDataset(val_files, bag_size=2000, use_variance=False)
        test_dataset = ThyroidWSIDataset(test_files, bag_size=2000, use_variance=False)
        
        fold_datasets.append({
            'fold': fold_idx,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        })
        
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_files)} WSIs")
        print(f"  Val: {len(val_files)} WSIs")
        print(f"  Test: {len(test_files)} WSIs")
    
    return fold_datasets


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ========================================
# 사용 예시
# ========================================
if __name__ == "__main__":
    set_seed(42)
    
    # JSON 기반 K-Fold 로드
    fold_datasets = load_json_splits(
        json_path="/home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model/outputs/Thyroid_prediction_model_v0.1.0/cv_splits/cv_splits_k5_seed42.json",
        data_root="data"  # meta/, nonmeta/ 폴더가 있는 경로
    )
    
    # Fold 1 확인
    fold_1 = fold_datasets[0]
    train_dataset = fold_1['train_dataset']
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    for features, label, filename in train_loader:
        print(f"\nWSI: {filename}")
        print(f"Features: {features.shape}")
        print(f"Label: {label}")
        break
