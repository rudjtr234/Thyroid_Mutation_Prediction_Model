import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

def get_wsi_list(data_root):
    """
    meta / nonmeta 디렉토리에서 npy 파일 경로 목록 가져오기
    - meta → label=1
    - nonmeta → label=0
    """
    meta_dir = Path(data_root) / "meta"
    nonmeta_dir = Path(data_root) / "nonmeta"
    
    meta_wsis = [str(f.resolve()) for f in meta_dir.glob("*.npy")]
    nonmeta_wsis = [str(f.resolve()) for f in nonmeta_dir.glob("*.npy")]
    
    wsis = [{"filename": f, "label": 1} for f in meta_wsis] + \
           [{"filename": f, "label": 0} for f in nonmeta_wsis]
    
    return wsis


def create_stratified_cv_splits_8_1_1(data_root, k_folds=5, seed=42):
    """
    Stratified 5-Fold Cross Validation 생성 (Train:Val:Test = 8:1:1)
    - ✅ Train/Val/Test 모두 Stratified
    - meta / nonmeta 비율 유지
    - fold 간 중복 없음
    """
    all_wsis = get_wsi_list(data_root)
    filenames = np.array([w["filename"] for w in all_wsis])
    labels = np.array([w["label"] for w in all_wsis])
    
    # StratifiedKFold로 10% test 분리
    skf_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    cv_splits = {
        "seed": seed,
        "k_folds": k_folds,
        "total_wsis": len(filenames),
        "braf_pos_count": int(labels.sum()),
        "braf_neg_count": int((labels == 0).sum()),
        "split_ratio": "8:1:1 (train:val:test)",
        "folds": []
    }
    
    folds = []
    for train_val_idx, test_idx in skf_outer.split(filenames, labels):
        folds.append((train_val_idx, test_idx))
    
    # 앞쪽 5개의 fold만 사용
    for fold_idx in range(k_folds):
        train_val_idx, test_idx = folds[fold_idx]
        
        train_val_files = filenames[train_val_idx]
        train_val_labels = labels[train_val_idx]
        test_files = filenames[test_idx]
        test_labels = labels[test_idx]
        
        # ✅ Train/Val도 Stratified로 split (9:1 비율)
        train_files, val_files, train_labels_split, val_labels_split = train_test_split(
            train_val_files,
            train_val_labels,
            test_size=0.1111,  # 전체의 10%가 val (train_val의 1/9)
            stratify=train_val_labels,  # ✅ Stratified!
            random_state=seed + fold_idx
        )
        
        # Label distribution 출력
        train_pos = int(train_labels_split.sum())
        train_neg = len(train_labels_split) - train_pos
        val_pos = int(val_labels_split.sum())
        val_neg = len(val_labels_split) - val_pos
        test_pos = int(test_labels.sum())
        test_neg = len(test_labels) - test_pos
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: Pos={train_pos:3d} ({train_pos/len(train_labels_split)*100:5.1f}%), "
              f"Neg={train_neg:3d} ({train_neg/len(train_labels_split)*100:5.1f}%), Total={len(train_labels_split):3d}")
        print(f"  Val  : Pos={val_pos:3d} ({val_pos/len(val_labels_split)*100:5.1f}%), "
              f"Neg={val_neg:3d} ({val_neg/len(val_labels_split)*100:5.1f}%), Total={len(val_labels_split):3d}")
        print(f"  Test : Pos={test_pos:3d} ({test_pos/len(test_labels)*100:5.1f}%), "
              f"Neg={test_neg:3d} ({test_neg/len(test_labels)*100:5.1f}%), Total={len(test_labels):3d}")
        
        fold_data = {
            "fold": fold_idx + 1,
            "train_wsis": train_files.tolist(),
            "train_count": len(train_files),
            "train_pos_count": train_pos,
            "train_neg_count": train_neg,
            "val_wsis": val_files.tolist(),
            "val_count": len(val_files),
            "val_pos_count": val_pos,
            "val_neg_count": val_neg,
            "test_wsis": test_files.tolist(),
            "test_count": len(test_files),
            "test_pos_count": test_pos,
            "test_neg_count": test_neg
        }
        
        cv_splits["folds"].append(fold_data)
    
    return cv_splits


if __name__ == "__main__":
    data_root = "/data/143/member/jks/Thyroid_Mutation_dataset/embeddings"
    save_dir = "./cv_splits"
    
    cv_splits = create_stratified_cv_splits_8_1_1(data_root, k_folds=5, seed=42)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f"cv_splits_k{cv_splits['k_folds']}_seed{cv_splits['seed']}_stratified_8_1_1.json"
    
    with open(save_path, "w") as f:
        json.dump(cv_splits, f, indent=2)
    
    print(f"\n[✓] Stratified CV splits (8:1:1, fully stratified) saved at: {save_path}")
    
    # Summary
    print(f"\nTotal WSIs: {cv_splits['total_wsis']}")
    print(f"Positive: {cv_splits['braf_pos_count']} ({cv_splits['braf_pos_count']/cv_splits['total_wsis']*100:.1f}%)")
    print(f"Negative: {cv_splits['braf_neg_count']} ({cv_splits['braf_neg_count']/cv_splits['total_wsis']*100:.1f}%)")
