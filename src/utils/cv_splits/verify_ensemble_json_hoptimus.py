# -*- coding: utf-8 -*-
"""
convert_ensemble_json_to_hoptimus.py로 생성한 H-optimus-0 앙상블 JSON을
학습 시작 전에 검증한다. 실패 시 non-zero exit으로 종료 — 학습을 시작하지 않는다.

검증 항목:
1. 신규 JSON이 가리키는 모든 train/val/test .npy 경로가 실제로 존재하는지 전수 확인
2. (중복 제거 후 1회씩만) .npy의 마지막 차원이 1536인지 mmap으로 확인 (전체 배열을 메모리에 올리지 않음)
3. heatmap을 생성하는 test WSI에 한해, 좌표 JSON이 존재하고 patch_coords(또는 tiles/coords) 길이가
   대응 .npy의 shape[0]과 일치하는지 확인

사용법:
    python verify_ensemble_json_hoptimus.py \
        --ensemble_json .../ensemble_5models_cv_hoptimus0.json \
        --test_json .../test_set_hoptimus0.json \
        --json_meta_dir .../h_optimus_embeddings/final_meta_dataset_v0.5.0_40x512/json \
        --json_nonmeta_dir .../h_optimus_embeddings/final_non_meta_dataset_v0.5.0_40x512/json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

EXPECTED_DIM = 1536


def collect_npy_paths(ensemble_data: dict, test_data: dict):
    """train/val/test 전체에서 (positive/negative 파일, dir)을 순회해 절대경로 리스트 생성."""
    paths = []

    for model_info in ensemble_data.get("models", []):
        for split in ("train", "val"):
            split_data = model_info.get(split, {})
            for cls in ("positive", "negative"):
                cls_data = split_data.get(cls, {})
                files = cls_data.get("files", [])
                d = cls_data.get("dir")
                if d is None:
                    continue
                paths.extend(os.path.join(d, fn) for fn in files)

    if isinstance(test_data.get("positive"), dict):
        for cls in ("positive", "negative"):
            cls_data = test_data[cls]
            paths.extend(os.path.join(cls_data["dir"], fn) for fn in cls_data["files"])
    else:
        pos_dir = test_data["positive_dir"]
        neg_dir = test_data["negative_dir"]
        paths.extend(os.path.join(pos_dir, fn) for fn in test_data["positive"])
        paths.extend(os.path.join(neg_dir, fn) for fn in test_data["negative"])

    return paths


def collect_test_files(test_data: dict):
    """heatmap 대상인 test WSI의 (filename, dir) 리스트."""
    result = []
    if isinstance(test_data.get("positive"), dict):
        for cls in ("positive", "negative"):
            cls_data = test_data[cls]
            for fn in cls_data["files"]:
                result.append((fn, cls_data["dir"]))
    else:
        for fn in test_data["positive"]:
            result.append((fn, test_data["positive_dir"]))
        for fn in test_data["negative"]:
            result.append((fn, test_data["negative_dir"]))
    return result


def extract_coord_len(metadata) -> int:
    if isinstance(metadata, dict):
        if "tiles" in metadata:
            return len(metadata["tiles"])
        if "patch_coords" in metadata:
            return len(metadata["patch_coords"])
        if "coords" in metadata:
            return len(metadata["coords"])
    if isinstance(metadata, list):
        return len(metadata)
    raise ValueError("좌표 JSON에서 tiles/patch_coords/coords 키를 찾을 수 없음")


def find_coord_json(wsi_stem: str, json_meta_dir: Path, json_nonmeta_dir: Path):
    # visualization.py의 check_json_metadata_exists와 동일한 탐색 순서
    # (coords_selected_ 접두사는 uni2 meta 구조용, h-optimus-0 extract_features.py는 접두사 없이 저장)
    candidates = [
        json_meta_dir / f"coords_selected_{wsi_stem}.json",
        json_meta_dir / f"{wsi_stem}.json",
        json_nonmeta_dir / f"{wsi_stem}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="H-optimus-0 ensemble JSON 사전 검증")
    parser.add_argument("--ensemble_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--json_meta_dir", type=str, required=True)
    parser.add_argument("--json_nonmeta_dir", type=str, required=True)
    args = parser.parse_args()

    errors = []

    with open(args.ensemble_json) as f:
        ensemble_data = json.load(f)
    with open(args.test_json) as f:
        test_data = json.load(f)

    # 1. 전수 존재 확인
    all_paths = collect_npy_paths(ensemble_data, test_data)
    print(f"[INFO] 전체 .npy 경로 수(중복 포함): {len(all_paths)}")

    missing = [p for p in all_paths if not os.path.exists(p)]
    if missing:
        errors.append(f".npy 파일 {len(missing)}개 누락 (예: {missing[:5]})")

    # 2. unique path 기준 1회씩만 shape 검증 (mmap, 전체 로드 금지)
    unique_paths = sorted(set(all_paths) - set(missing))
    print(f"[INFO] shape 검증 대상 (unique): {len(unique_paths)}")

    bad_shape = []
    for p in unique_paths:
        try:
            arr = np.load(p, mmap_mode="r", allow_pickle=False)
            if arr.shape[-1] != EXPECTED_DIM:
                bad_shape.append((p, arr.shape))
        except Exception as e:
            errors.append(f"{p} 로드 실패: {e}")

    if bad_shape:
        errors.append(f"차원 불일치 {len(bad_shape)}건 (예: {bad_shape[:5]})")

    # 3. heatmap 대상 test WSI의 좌표 JSON 검증
    json_meta_dir = Path(args.json_meta_dir)
    json_nonmeta_dir = Path(args.json_nonmeta_dir)
    test_files = collect_test_files(test_data)
    print(f"[INFO] heatmap 좌표 검증 대상 test WSI: {len(test_files)}")

    coord_missing = []
    coord_len_mismatch = []
    for fn, d in test_files:
        npy_path = os.path.join(d, fn)
        if npy_path in missing:
            continue  # 이미 위에서 보고됨
        stem = Path(fn).stem
        coord_path = find_coord_json(stem, json_meta_dir, json_nonmeta_dir)
        if coord_path is None:
            coord_missing.append(stem)
            continue
        try:
            with open(coord_path) as f:
                metadata = json.load(f)
            coord_len = extract_coord_len(metadata)
            npy_shape0 = np.load(npy_path, mmap_mode="r", allow_pickle=False).shape[0]
            if coord_len != npy_shape0:
                coord_len_mismatch.append((stem, coord_len, npy_shape0))
        except Exception as e:
            errors.append(f"{stem} 좌표 검증 실패: {e}")

    if coord_missing:
        errors.append(f"좌표 JSON 누락 {len(coord_missing)}건 (예: {coord_missing[:5]})")
    if coord_len_mismatch:
        errors.append(f"좌표 개수 불일치 {len(coord_len_mismatch)}건 (예: {coord_len_mismatch[:5]})")

    if errors:
        print("[FAIL] 사전 검증 실패:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    print("[✓] 사전 검증 통과")


if __name__ == "__main__":
    main()
