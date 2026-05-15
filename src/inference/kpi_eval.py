"""
KPI 평가 스크립트 — 여러 슬라이드 배치 추론 후 p50/p95 계산.

학습된 체크포인트(.pt)를 직접 로드해서 추론합니다.

E1 (추론 시간 p95):
  - 슬라이드 n장을 배치 추론하여 슬라이드별 total_sec 기록
  - p50, p95 계산

사용법:
    # 앙상블 모델 (5개 .pt)
    python src/inference/kpi_eval.py \
        --checkpoint_dir outputs/braf_ensemble_v1.0.0/checkpoints \
        --embedding_dir /path/to/embeddings/npy \
        --n_slides 50

    # 단일 모델 (fold .pt 하나)
    python src/inference/kpi_eval.py \
        --checkpoint_dir outputs/Thyroid_prediction_model_v1.0.0/checkpoints \
        --embedding_dir /path/to/embeddings/npy \
        --n_slides 50 \
        --single_model
"""

import os
import json
import time
import argparse
import random
import sys
from pathlib import Path
import importlib.util


def _ensure_torch_available():
    if importlib.util.find_spec("torch") is None:
        print("[!] ModuleNotFoundError: No module named 'torch'")
        print(f"[!] Python: {sys.executable}")
        print("[!] Install: pip install -r requirements.txt")
        sys.exit(1)


_ensure_torch_available()

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.abmil import ABMILModel, ABMILGatedBaseConfig


def load_model(pt_path: str, device: torch.device) -> ABMILModel:
    """학습된 체크포인트에서 ABMILModel 로드"""
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    config = ABMILGatedBaseConfig()
    model = ABMILModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model


def infer_single(model: ABMILModel, npy_path: Path, device: torch.device) -> dict:
    """슬라이드 1장 추론 + 타이밍 측정"""
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    embeddings = np.load(str(npy_path))
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        h = torch.from_numpy(embeddings).float().unsqueeze(0).to(device)
        results_dict, _ = model(h, return_attention=False)
        probs = F.softmax(results_dict["logits"], dim=-1).cpu().numpy()[0]
    t_infer = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total

    return {
        "slide_id": npy_path.stem,
        "num_patches": embeddings.shape[0],
        "braf_pos_prob": float(probs[1]),
        "sec_load": round(t_load, 3),
        "sec_infer": round(t_infer, 3),
        "sec_total": round(t_total, 3),
    }


def infer_ensemble(models: list, npy_path: Path, device: torch.device) -> dict:
    """앙상블 5개 모델로 슬라이드 1장 추론 + 타이밍 측정"""
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    embeddings = np.load(str(npy_path))
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    all_probs = []
    with torch.no_grad():
        h = torch.from_numpy(embeddings).float().unsqueeze(0).to(device)
        for model in models:
            results_dict, _ = model(h, return_attention=False)
            probs = F.softmax(results_dict["logits"], dim=-1).cpu().numpy()[0]
            all_probs.append(probs)
    ensemble_prob = float(np.mean([p[1] for p in all_probs]))
    t_infer = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total

    return {
        "slide_id": npy_path.stem,
        "num_patches": embeddings.shape[0],
        "braf_pos_prob": ensemble_prob,
        "sec_load": round(t_load, 3),
        "sec_infer": round(t_infer, 3),
        "sec_total": round(t_total, 3),
    }


def run_kpi_eval(checkpoint_dir: str, embedding_dir: str, output_dir: str,
                 n_slides: int = 50, seed: int = 42, single_model: bool = False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback_dir = Path.home() / ".cache" / "thyroid_kpi_eval"
        print(f"[!] Permission denied: {out_dir} → fallback: {fallback_dir}")
        out_dir = fallback_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(Path(checkpoint_dir).glob("*.pt"))
    if not pt_files:
        print(f"[!] No .pt files in {checkpoint_dir}")
        return

    if single_model:
        pt_files = pt_files[:1]
        models = [load_model(str(pt_files[0]), device)]
        print(f"[KPI] Single model: {pt_files[0].name}")
    else:
        models = [load_model(str(p), device) for p in pt_files]
        print(f"[KPI] Ensemble: {len(models)} models loaded")

    npy_files = sorted(Path(embedding_dir).glob("*.npy"))
    if not npy_files:
        print(f"[!] No .npy files in {embedding_dir}")
        return

    random.seed(seed)
    slides = random.sample(npy_files, min(n_slides, len(npy_files)))
    print(f"[KPI] {len(slides)} slides sampled")

    timings = []
    errors = []
    for npy_path in slides:
        try:
            if single_model:
                result = infer_single(models[0], npy_path, device)
            else:
                result = infer_ensemble(models, npy_path, device)
            timings.append(result)
            print(f"  {npy_path.stem}: total={result['sec_total']:.2f}s  infer={result['sec_infer']:.2f}s")
        except Exception as e:
            errors.append({"slide_id": npy_path.stem, "error": str(e)})
            print(f"  [!] {npy_path.stem}: {e}")

    if not timings:
        print("[!] No timing data collected")
        return

    total_secs = np.array([t["sec_total"] for t in timings])
    infer_secs = np.array([t["sec_infer"] for t in timings])

    kpi = {
        "n_slides": len(timings),
        "n_errors": len(errors),
        "n_models": len(models),
        "total_p50_sec": round(float(np.percentile(total_secs, 50)), 3),
        "total_p95_sec": round(float(np.percentile(total_secs, 95)), 3),
        "total_mean_sec": round(float(np.mean(total_secs)), 3),
        "infer_p50_sec": round(float(np.percentile(infer_secs, 50)), 3),
        "infer_p95_sec": round(float(np.percentile(infer_secs, 95)), 3),
    }

    print(f"\n[KPI] Results:")
    print(f"  Total  p50={kpi['total_p50_sec']}s  p95={kpi['total_p95_sec']}s")
    print(f"  Infer  p50={kpi['infer_p50_sec']}s  p95={kpi['infer_p95_sec']}s")
    print(f"  Errors: {len(errors)}/{len(slides)}")

    result_path = out_dir / "kpi_timing.json"
    with open(result_path, "w") as f:
        json.dump({"kpi": kpi, "timings": timings, "errors": errors}, f, indent=2)
    print(f"[✓] Saved: {result_path}")


if __name__ == "__main__":
    default_output_dir = str(Path(__file__).resolve().parent / "kpi_eval_results")
    parser = argparse.ArgumentParser(description="KPI Evaluation (inference timing p50/p95)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing .pt checkpoints")
    parser.add_argument("--embedding_dir", type=str, required=True,
                        help="Directory containing .npy embedding files")
    parser.add_argument("--output_dir", type=str,
                        default=default_output_dir,
                        help="Output directory")
    parser.add_argument("--n_slides", type=int, default=50,
                        help="Number of slides to sample (default: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single_model", action="store_true",
                        help="Use only one .pt (first file). Default: ensemble all .pt in dir")
    args = parser.parse_args()

    run_kpi_eval(
        checkpoint_dir=args.checkpoint_dir,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        n_slides=args.n_slides,
        seed=args.seed,
        single_model=args.single_model,
    )
