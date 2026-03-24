## -*- coding: utf-8 -*-

import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

def get_wsi_list(meta_dir, nonmeta_dir, balance=True, seed=42):
    """
    meta / nonmeta 디렉토리에서 npy 파일 경로 목록 가져오기
    - meta → label=1
    - nonmeta → label=0
    - balance=True일 경우, meta에서 nonmeta 개수만큼만 랜덤 샘플링
    """
    meta_path = Path(meta_dir)
    nonmeta_path = Path(nonmeta_dir)

    meta_wsis = [str(f.resolve()) for f in meta_path.glob("*.npy")]
    nonmeta_wsis = [str(f.resolve()) for f in nonmeta_path.glob("*.npy")]

    print(f"\n[Original Data Info]")
    print(f"Meta files: {len(meta_wsis)}")
    print(f"Nonmeta files: {len(nonmeta_wsis)}")

    # Class Balance 맞추기
    if balance and len(meta_wsis) > len(nonmeta_wsis):
        np.random.seed(seed)
        meta_wsis = np.random.choice(meta_wsis, size=len(nonmeta_wsis), replace=False).tolist()
        print(f"\n[Balanced Sampling]")
        print(f"Meta sampled down to: {len(meta_wsis)}")
    elif balance and len(nonmeta_wsis) > len(meta_wsis):
        np.random.seed(seed)
        nonmeta_wsis = np.random.choice(nonmeta_wsis, size=len(meta_wsis), replace=False).tolist()
        print(f"\n[Balanced Sampling]")
        print(f"Nonmeta sampled down to: {len(nonmeta_wsis)}")

    wsis = [{"filename": f, "label": 1} for f in meta_wsis] + \
           [{"filename": f, "label": 0} for f in nonmeta_wsis]

    print(f"\n[Final Data Info]")
    print(f"Meta files: {len(meta_wsis)}")
    print(f"Nonmeta files: {len(nonmeta_wsis)}")
    print(f"Total files: {len(wsis)}")

    return wsis


def create_stratified_cv_splits_8_1_1(meta_dir, nonmeta_dir, k_folds=5, seed=42, balance=True):
    """
    Stratified K-Fold Cross Validation 생성 (Train:Val:Test = 8:1:1)
    - balance=True: meta/nonmeta 개수를 동일하게 맞춤
    - 모든 split에서 Stratified 유지
    """
    all_wsis = get_wsi_list(meta_dir, nonmeta_dir, balance=balance, seed=seed)
    filenames = np.array([w["filename"] for w in all_wsis])
    labels = np.array([w["label"] for w in all_wsis])

    total_count = len(filenames)
    pos_count = int(labels.sum())
    neg_count = int((labels == 0).sum())

    print(f"\n[Distribution]")
    print(f"Total: {total_count}")
    print(f"Positive (meta): {pos_count} ({pos_count/total_count*100:.1f}%)")
    print(f"Negative (nonmeta): {neg_count} ({neg_count/total_count*100:.1f}%)")

    cv_splits = {
        "seed": seed,
        "k_folds": k_folds,
        "total_wsis": total_count,
        "meta_count": pos_count,
        "nonmeta_count": neg_count,
        "split_ratio": "8:1:1 (train:val:test)",
        "balanced": balance,
        "meta_dir": str(meta_dir),
        "nonmeta_dir": str(nonmeta_dir),
        "folds": []
    }

    # k-fold로 나눠서 각 fold가 동일한 비율이 되도록 함
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

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


def convert_paths_to_target(cv_splits, source_prefix, target_prefix):
    """JSON 내 경로를 source_prefix에서 target_prefix로 변환"""
    cv_splits["meta_dir"] = cv_splits["meta_dir"].replace(source_prefix, target_prefix)
    cv_splits["nonmeta_dir"] = cv_splits["nonmeta_dir"].replace(source_prefix, target_prefix)

    for fold_data in cv_splits["folds"]:
        fold_data["train_wsis"] = [p.replace(source_prefix, target_prefix) for p in fold_data["train_wsis"]]
        fold_data["val_wsis"] = [p.replace(source_prefix, target_prefix) for p in fold_data["val_wsis"]]
        fold_data["test_wsis"] = [p.replace(source_prefix, target_prefix) for p in fold_data["test_wsis"]]

    return cv_splits


if __name__ == "__main__":
    # UNI2-H 20x 256 Dataset (meta v0.2.0 500장, non_meta v0.2.0 500장)
    # 로컬 경로로 파일 읽기
    meta_dir = "/data/member/jks/dataset/Thyroid_Mutation_dataset/uni2_embeddings/final_meta_dataset_v0.2.0/npy"
    nonmeta_dir = "/data/member/jks/dataset/Thyroid_Mutation_dataset/uni2_embeddings/final_non_meta_dataset_v0.2.0/npy"
    save_dir = str(Path(__file__).resolve().parent)

    # 5-Fold CV 생성 (8:1:1 ratio)
    cv_splits = create_stratified_cv_splits_8_1_1(
        meta_dir=meta_dir,
        nonmeta_dir=nonmeta_dir,
        k_folds=5,
        seed=42,
        balance=True  # 500:500으로 balanced
    )

    # 경로 변환: /data/member/ → /data/143/member/ (143 서버용)
    cv_splits = convert_paths_to_target(cv_splits, "/data/member/", "/data/143/member/")

    # 저장
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / "cv_splits_braf_uni2h_20x256_k5_seed42.json"

    with open(save_path, "w") as f:
        json.dump(cv_splits, f, indent=2)

    print(f"\n[✓] Stratified CV splits (8:1:1, balanced) saved at: {save_path}")

    # Summary
    print(f"\n[Summary]")
    print(f"Total WSIs: {cv_splits['total_wsis']}")
    print(f"Meta (Positive): {cv_splits['meta_count']} ({cv_splits['meta_count']/cv_splits['total_wsis']*100:.1f}%)")
    print(f"Nonmeta (Negative): {cv_splits['nonmeta_count']} ({cv_splits['nonmeta_count']/cv_splits['total_wsis']*100:.1f}%)")
    print(f"K-Folds: {cv_splits['k_folds']}")
    print(f"Balanced: {cv_splits['balanced']}")
