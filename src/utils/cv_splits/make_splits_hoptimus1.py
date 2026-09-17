## -*- coding: utf-8 -*-
"""H-optimus-1 40x512 (862 balanced, 1724 WSI) 5-Fold CV split 생성.

cv_splits.py의 생성 로직을 그대로 사용한다. 실행은 임베딩이 있는 서버에서:
    cd src/utils/cv_splits && python make_splits_hoptimus1.py
"""

import json
from pathlib import Path

from cv_splits import create_stratified_cv_splits_8_1_1

EMB = "/path/to/dataset/h_optimus_1_embeddings"
META_DIR = f"{EMB}/final_meta_dataset_v0.1.0_40x512/npy"
NONMETA_DIR = f"{EMB}/final_non_meta_dataset_v0.1.0_40x512/npy"

if __name__ == "__main__":
    save_dir = Path(__file__).resolve().parent / "h-optimus-1"
    save_dir.mkdir(parents=True, exist_ok=True)

    # meta 4038 → non_meta 862에 맞춰 다운샘플 (balance=True), 1724 WSI
    cv_splits = create_stratified_cv_splits_8_1_1(
        meta_dir=META_DIR,
        nonmeta_dir=NONMETA_DIR,
        k_folds=5,
        seed=42,
        balance=True,
    )

    # 이미 /path/to 절대경로이므로 convert_paths_to_target 불필요
    save_path = save_dir / "cv_splits_braf_hoptimus1_862wsi_40x512_k5_seed42.json"
    with open(save_path, "w") as f:
        json.dump(cv_splits, f, indent=2)

    print(f"\n[✓] saved: {save_path}")
    print(f"Total={cv_splits['total_wsis']} "
          f"meta={cv_splits['meta_count']} nonmeta={cv_splits['nonmeta_count']}")
