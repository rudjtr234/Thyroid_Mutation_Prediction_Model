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
            # 경로에서 label 추론 (더 구체적인 패턴 매칭)
            if 'final_meta_dataset' in filepath:
                label = 1  # BRAF+
            elif 'final_nonmeta_dataset' in filepath:
                label = 0  # BRAF-
            elif 'meta_test_final' in filepath or '/meta/' in filepath or '\\meta\\' in filepath:
                label = 1  # BRAF+
            elif 'nonmeta_test_final' in filepath or '/nonmeta/' in filepath or '\\nonmeta\\' in filepath:
                label = 0  # BRAF-
            else:
                # 파일명으로 추론 불가능한 경우 경고
                print(f"Warning: Cannot infer label from path: {filepath}")
                label = 0

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


def load_json_splits(json_path):
    """
    JSON 파일에서 미리 정의된 K-Fold split 로드
    - JSON에 전체 경로가 저장되어 있으므로 그대로 사용
    """
    with open(json_path, 'r') as f:
        split_data = json.load(f)

    fold_datasets = []

    for fold_info in split_data['folds']:
        fold_idx = fold_info['fold']

        # JSON에 전체 경로가 저장되어 있으므로 그대로 사용
        train_files = fold_info['train_wsis']
        val_files = fold_info['val_wsis']
        test_files = fold_info['test_wsis']

        # 파일 존재 여부 확인
        train_files_exist = [f for f in train_files if os.path.exists(f)]
        val_files_exist = [f for f in val_files if os.path.exists(f)]
        test_files_exist = [f for f in test_files if os.path.exists(f)]

        # 누락된 파일 경고
        if len(train_files_exist) != len(train_files):
            print(f"Warning: Fold {fold_idx} - {len(train_files) - len(train_files_exist)} train files not found")
        if len(val_files_exist) != len(val_files):
            print(f"Warning: Fold {fold_idx} - {len(val_files) - len(val_files_exist)} val files not found")
        if len(test_files_exist) != len(test_files):
            print(f"Warning: Fold {fold_idx} - {len(test_files) - len(test_files_exist)} test files not found")

        # Dataset 생성
        train_dataset = ThyroidWSIDataset(train_files_exist, bag_size=2000, use_variance=False)
        val_dataset = ThyroidWSIDataset(val_files_exist, bag_size=2000, use_variance=False)
        test_dataset = ThyroidWSIDataset(test_files_exist, bag_size=2000, use_variance=False)

        fold_datasets.append({
            'fold': fold_idx,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        })

        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_files_exist)} WSIs")
        print(f"  Val: {len(val_files_exist)} WSIs")
        print(f"  Test: {len(test_files_exist)} WSIs")

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

    # JSON 기반 K-Fold 로드 (data_root 파라미터 제거)
    fold_datasets = load_json_splits(
        json_path="/home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model/outputs/Thyroid_prediction_model_v0.1.0/cv_splits/cv_splits_balanced_k5_seed42_v0.2.1.json"
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