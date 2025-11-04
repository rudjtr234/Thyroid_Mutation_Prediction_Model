"""
Non-Meta BRAF 데이터 임베딩 (862개 슬라이드)

torchrun --nproc_per_node=3 preprocess_data.py \
    --tile_dir /data/143/member/kwk/dl/thyroid/image/slide-v1-240412/patch/Train/preprocess_data/braf_meta_v0.2.0 \
    --out_dir /data/143/member/jks/Thyroid_Mutation_dataset/embeddings/preprocess_data/braf_meta_v0.2.0 \
    --batch_size 1024
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


def process_slide(slide_dir, args, model, transform, device, rank, world_size):
    tile_paths = sorted(slide_dir.glob("*.png"))
    if len(tile_paths) == 0:
        return None

    sub_paths = tile_paths[rank::world_size]
    features, coords, tiles_info = [], [], []

    for i in tqdm(range(0, len(sub_paths), args.batch_size),
                  desc=f"GPU {rank} - {slide_dir.name}" if rank == 0 else None,
                  disable=(rank != 0)):
        batch_paths = sub_paths[i:i+args.batch_size]
        batch_coords, batch_info = [], []
        
        for p in batch_paths:
            parts = p.stem.split("_")
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

    # 모든 GPU 결과 수집
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

        # npy 폴더에 임베딩 저장
        npy_dir = Path(args.out_dir) / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        save_vec = npy_dir / f"{slide_id}.npy"
        np.save(save_vec, all_features)
        print(f"[✓] {save_vec} {all_features.shape}")

        # json 폴더에 패치 좌표 정보 저장
        json_dir = Path(args.out_dir) / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        save_json = json_dir / f"{slide_id}.json"
        
        coord_data = {
            "slide_id": slide_id,
            "num_tiles": len(tile_paths),
            "embedding_path": str(save_vec),
            "patch_coords": all_info  # 각 패치의 좌표 정보
        }
        with open(save_json, "w") as f:
            json.dump(coord_data, f, indent=2)

        return coord_data
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # DDP 초기화
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model, transform = build_model(device)

    tile_dir = Path(args.tile_dir)
    if rank == 0:
        slide_dirs = sorted([p for p in tile_dir.iterdir() if p.is_dir()])
        print(f"[INFO] 총 슬라이드 개수: {len(slide_dirs)}")
    else:
        slide_dirs = None

    slide_dirs_list = [slide_dirs]
    dist.broadcast_object_list(slide_dirs_list, src=0)
    slide_dirs = slide_dirs_list[0]

    master_data = []
    for idx, slide_dir in enumerate(slide_dirs):
        if rank == 0:
            print(f"\n[{idx+1}/{len(slide_dirs)}] {slide_dir.name}")
        
        coord_data = process_slide(slide_dir, args, model, transform, device, rank, world_size)
        if coord_data:
            master_data.append(coord_data)

    # 최종 저장
    if rank == 0 and master_data:
        json_dir = Path(args.out_dir) / "json"
        
        # 전체 슬라이드 정보를 하나의 마스터 JSON에 저장
        with open(json_dir / "all_slides.json", "w") as f:
            json.dump(master_data, f, indent=2)
        
        slide_ids = [item["slide_id"] for item in master_data]
        with open(json_dir / "slide_ids.txt", "w") as f:
            f.write("\n".join(slide_ids))
        
        print(f"\n[✓✓✓] 완료! 총 {len(slide_ids)}개 슬라이드")
        print(f"[INFO] npy 저장: {args.out_dir}/npy/")
        print(f"[INFO] json 저장: {args.out_dir}/json/")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
