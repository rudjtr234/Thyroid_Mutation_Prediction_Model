"""
ABMIL Preprocessing for UNI2-h Embeddings - Nonmeta V2
- 기존 nonmeta 100개 제외, 나머지 150개만 임베딩
- 출력: embeddings/nonmeta_v2/
"""

import os
import argparse
import torch
import torch.distributed as dist
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import timm
import random


# =====================
# 모델 준비
# =====================
def build_model(device):
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model.to(device)
    model.eval()
    data_cfg = resolve_data_config({}, model=model)
    transform = create_transform(**data_cfg)
    return model, transform


def embed_batch(img_paths, model, transform, device):
    imgs = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))
    tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        feats = model(tensor)
    return feats.cpu().numpy()


def load_excluded_slides(exclude_file):
    """이미 처리된 슬라이드 ID 로드"""
    if not exclude_file or not Path(exclude_file).exists():
        return set()
    
    excluded = set()
    with open(exclude_file, 'r') as f:
        for line in f:
            slide_id = line.strip()
            if slide_id:
                if slide_id.endswith('.npy'):
                    slide_id = slide_id[:-4]
                excluded.add(slide_id)
    
    return excluded


def process_slide(slide_dir, args, model, transform, device, rank, world_size):
    """단일 슬라이드 폴더 임베딩"""
    tile_paths = sorted(slide_dir.glob("*.png"))
    if len(tile_paths) == 0:
        if rank == 0:
            print(f"[⚠] No tiles found in {slide_dir}")
        return None

    sub_paths = tile_paths[rank::world_size]

    features, coords, tiles_info = [], [], []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(sub_paths), batch_size),
                  desc=f"GPU {rank} - {slide_dir.name}" if rank == 0 else None,
                  disable=(rank != 0)):
        batch_paths = sub_paths[i:i+batch_size]

        batch_coords, batch_info = [], []
        for p in batch_paths:
            fname = p.stem
            parts = fname.split("_")
            try:
                x = int(parts[-2][1:])
                y = int(parts[-1][1:])
            except:
                x, y = 0, 0
            batch_coords.append([x, y])
            batch_info.append({"name": p.name, "x": x, "y": y})

        feats = embed_batch(batch_paths, model, transform, device)
        features.append(feats)
        coords.extend(batch_coords)
        tiles_info.extend(batch_info)

    features = np.vstack(features) if len(features) > 0 else np.empty((0, 1536))

    all_features = [None for _ in range(world_size)]
    all_coords = [None for _ in range(world_size)]
    all_info = [None for _ in range(world_size)]

    dist.all_gather_object(all_features, features)
    dist.all_gather_object(all_coords, coords)
    dist.all_gather_object(all_info, tiles_info)

    if rank == 0:
        all_features = np.vstack([f for f in all_features if f is not None and f.shape[0] > 0])
        all_coords = sum([c for c in all_coords if c is not None], [])
        all_info = sum([i for i in all_info if i is not None], [])

        slide_id = slide_dir.name
        
        # .npy 저장 (nonmeta_v2 폴더에)
        npy_dir = Path(args.out_dir) / "embeddings" / "nonmeta_v2"
        npy_dir.mkdir(parents=True, exist_ok=True)
        
        save_vec = npy_dir / f"{slide_id}.npy"
        np.save(save_vec, all_features)
        print(f"[✓] Saved {save_vec} {all_features.shape}")

        # .json 저장
        json_dir = Path(args.out_dir) / "json_metadata"
        json_dir.mkdir(parents=True, exist_ok=True)
        
        save_json = json_dir / f"coords_nonmeta_v2_{slide_id}.json"
        coord_data = {
            "slide_id": slide_id,
            "split": "nonmeta_v2",
            "num_tiles": len(tile_paths),
            "embedding_path": str(save_vec),
            "tiles": all_info
        }
        with open(save_json, "w") as f:
            json.dump(coord_data, f, indent=2)
        print(f"[✓] Saved {save_json} ({len(all_info)} tiles)")

        return coord_data
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_dir", type=str, 
                       default="/data/member/kwk/dl/thyroid/image/slide-v1-240412/patch/Train/braf_non_meta",
                       help="braf_non_meta 타일 디렉토리")
    parser.add_argument("--out_dir", type=str, 
                       default="/data/143/member/jks/Thyroid_Mutation_dataset",
                       help="출력 디렉토리")
    parser.add_argument("--exclude_list", type=str,
                       default="/data/143/member/jks/Thyroid_Mutation_dataset/json_metadata/slide_ids_nonmeta.txt",
                       help="제외할 슬라이드 목록 (이미 처리된 100개)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # DDP 초기화
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # 모델 준비
    model, transform = build_model(device)

    # 제외할 슬라이드 로드
    excluded_slides = set()
    if rank == 0:
        excluded_slides = load_excluded_slides(args.exclude_list)
        if excluded_slides:
            print(f"[INFO] Loaded {len(excluded_slides)} slides to exclude")
            print(f"[INFO] First 5 excluded: {list(excluded_slides)[:5]}")

    # 타일 디렉토리 처리
    tile_dir = Path(args.tile_dir)
    if not tile_dir.exists():
        raise ValueError(f"Tile directory not found: {tile_dir}")
    
    master_data = []
    
    slide_dirs = [p for p in tile_dir.iterdir() if p.is_dir()]
    
    if rank == 0:
        original_count = len(slide_dirs)
        
        filtered_dirs = []
        for slide_dir in slide_dirs:
            slide_name = slide_dir.name
            if slide_name not in excluded_slides:
                filtered_dirs.append(slide_dir)
        
        slide_dirs = filtered_dirs
        excluded_count = original_count - len(slide_dirs)
        
        print(f"[INFO] Total slides found: {original_count}")
        print(f"[INFO] Excluded (already processed): {excluded_count}")
        print(f"[INFO] Remaining to process: {len(slide_dirs)}")
        
        if len(slide_dirs) == 0:
            print("[WARNING] No slides left to process!")
    
    slide_dirs_list = [slide_dirs if rank == 0 else None]
    dist.broadcast_object_list(slide_dirs_list, src=0)
    slide_dirs = slide_dirs_list[0]
    
    for slide_dir in slide_dirs:
        coord_data = process_slide(slide_dir, args, model, transform, device, rank, world_size)
        if coord_data:
            master_data.append(coord_data)

    if rank == 0 and master_data:
        json_dir = Path(args.out_dir) / "json_metadata"
        json_dir.mkdir(parents=True, exist_ok=True)
        
        master_json = json_dir / "coords_nonmeta_v2_ALL.json"
        with open(master_json, "w") as f:
            json.dump(master_data, f, indent=2)
        print(f"[✓] Master JSON saved: {master_json} ({len(master_data)} slides)")
        
        slide_ids = [item["slide_id"] for item in master_data]
        slide_ids_txt = json_dir / "slide_ids_nonmeta_v2.txt"
        with open(slide_ids_txt, "w") as f:
            f.write("\n".join(slide_ids))
        print(f"[✓] Slide IDs saved: {slide_ids_txt} ({len(slide_ids)} slides)")
        
        metadata_json = json_dir / "metadata_nonmeta_v2.json"
        metadata = {
            "split": "nonmeta_v2",
            "total_slides": len(slide_ids),
            "excluded_slides": len(excluded_slides),
            "seed": args.seed,
            "slide_ids": slide_ids,
            "tile_dir": str(tile_dir),
            "exclude_list": args.exclude_list
        }
        with open(metadata_json, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[✓] Metadata saved: {metadata_json}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()