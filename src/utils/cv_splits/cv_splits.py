import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

def get_wsi_list(data_root):
    """
    meta_test_final / nonmeta_test_final 디렉토리에서 npy 파일 경로 목록 가져오기
    - meta_test_final → label=1
    - nonmeta_test_final → label=0
    """
    meta_dir = Path(data_root) / "meta_test_final"
    nonmeta_dir = Path(data_root) / "nonmeta_test_final"
    
    meta_wsis = [str(f.resolve()) for f in meta_dir.glob("*.npy")]
    nonmeta_wsis = [str(f.resolve()) for f in nonmeta_dir.glob("*.npy")]
    
    wsis = [{"filename": f, "label": 1} for f in meta_wsis] + \
           [{"filename": f, "label": 0} for f in nonmeta_wsis]
    
    print(f"\n[Data Info]")
    print(f"Meta files: {len(meta_wsis)}")
    print(f"Nonmeta files: {len(nonmeta_wsis)}")
    print(f"Total files: {len(wsis)}")
    
    return wsis


def create_stratified_cv_splits_8_1_1(data_root, k_folds=5, seed=42):
    """
    Stratified K-Fold Cross Validation 생성 (Train:Val:Test = 8:1:1)
    - 500장 데이터를 정확히 8:1:1 (400:50:50)로 나눔
    - ✅ 모든 split에서 Stratified 유지
    - meta / nonmeta 비율 유지
    """
    all_wsis = get_wsi_list(data_root)
    filenames = np.array([w["filename"] for w in all_wsis])
    labels = np.array([w["label"] for w in all_wsis])
    
    total_count = len(filenames)
    pos_count = int(labels.sum())
    neg_count = int((labels == 0).sum())
    
    print(f"\n[Original Distribution]")
    print(f"Total: {total_count}")
    print(f"Positive (meta): {pos_count} ({pos_count/total_count*100:.1f}%)")
    print(f"Negative (nonmeta): {neg_count} ({neg_count/total_count*100:.1f}%)")
    
    cv_splits = {
        "seed": seed,
        "k_folds": k_folds,
        "total_wsis": total_count,
        "braf_pos_count": pos_count,
        "braf_neg_count": neg_count,
        "split_ratio": "8:1:1 (train:val:test)",
        "folds": []
    }
    
    # 10-fold로 나눠서 각 fold가 10%씩 되도록 함
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    folds = []
    for train_val_idx, test_idx in skf.split(filenames, labels):
        folds.append((train_val_idx, test_idx))
    
    # 앞쪽 k_folds개의 fold만 사용
    for fold_idx in range(k_folds):
        train_val_idx, test_idx = folds[fold_idx]
        
        # Test set (10% of total)
        test_files = filenames[test_idx]
        test_labels = labels[test_idx]
        
        # Train+Val set (90% of total)
        train_val_files = filenames[train_val_idx]
        train_val_labels = labels[train_val_idx]
        
        # Train+Val을 다시 8:1로 split (전체 대비 80%:10%)
        # train_val의 1/9를 val로 분리 → 전체의 10%
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files,
            train_val_labels,
            test_size=1/9,  # train_val의 1/9 = 전체의 10%
            stratify=train_val_labels,
            random_state=seed + fold_idx
        )
        
        # Distribution 계산
        train_pos = int(train_labels.sum())
        train_neg = len(train_labels) - train_pos
        val_pos = int(val_labels.sum())
        val_neg = len(val_labels) - val_pos
        test_pos = int(test_labels.sum())
        test_neg = len(test_labels) - test_pos
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: Pos={train_pos:3d} ({train_pos/len(train_labels)*100:5.1f}%), "
              f"Neg={train_neg:3d} ({train_neg/len(train_labels)*100:5.1f}%), "
              f"Total={len(train_labels):3d} ({len(train_labels)/total_count*100:.1f}%)")
        print(f"  Val  : Pos={val_pos:3d} ({val_pos/len(val_labels)*100:5.1f}%), "
              f"Neg={val_neg:3d} ({val_neg/len(val_labels)*100:5.1f}%), "
              f"Total={len(val_labels):3d} ({len(val_labels)/total_count*100:.1f}%)")
        print(f"  Test : Pos={test_pos:3d} ({test_pos/len(test_labels)*100:5.1f}%), "
              f"Neg={test_neg:3d} ({test_neg/len(test_labels)*100:5.1f}%), "
              f"Total={len(test_labels):3d} ({len(test_labels)/total_count*100:.1f}%)")
        
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
    # 데이터 루트 경로 (meta_test_final과 nonmeta_test_final의 부모 디렉토리)
    data_root = "/data/143/member/jks/Thyroid_Mutation_dataset/embeddings"
    save_dir = "./cv_splits"
    
    # 5-Fold CV 생성
    cv_splits = create_stratified_cv_splits_8_1_1(data_root, k_folds=5, seed=42)
    
    # 저장
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f"cv_splits_k{cv_splits['k_folds']}_seed{cv_splits['seed']}_.json_v0.2.0"
    
    with open(save_path, "w") as f:
        json.dump(cv_splits, f, indent=2)
    
    print(f"\n[✓] Stratified CV splits (8:1:1) saved at: {save_path}")
    
    # Summary
    print(f"\n[Summary]")
    print(f"Total WSIs: {cv_splits['total_wsis']}")
    print(f"Meta (Positive): {cv_splits['braf_pos_count']} ({cv_splits['braf_pos_count']/cv_splits['total_wsis']*100:.1f}%)")
    print(f"Nonmeta (Negative): {cv_splits['braf_neg_count']} ({cv_splits['braf_neg_count']/cv_splits['total_wsis']*100:.1f}%)")
    print(f"K-Folds: {cv_splits['k_folds']}")
