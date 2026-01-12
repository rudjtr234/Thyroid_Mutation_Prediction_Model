"""
WSI attention heatmap visualization (full-WSI embeddings).
"""

import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache_wsi"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json_metadata(json_path: Path) -> Dict:
    with open(json_path, "r") as f:
        return json.load(f)


def check_json_metadata_exists(
    wsi_name: str,
    json_meta_dir: Path,
    json_nonmeta_dir: Path,
) -> Optional[Path]:
    candidates = [
        json_meta_dir / f"coords_meta_{wsi_name}.json",
        json_nonmeta_dir / f"coords_meta_{wsi_name}.json",
        json_meta_dir / f"{wsi_name}.json",
        json_nonmeta_dir / f"{wsi_name}.json",
    ]

    for path in candidates:
        if path.exists():
            return path
    return None


def filter_wsis_with_metadata(
    wsi_names: List[str],
    json_meta_dir: Path,
    json_nonmeta_dir: Path,
) -> List[str]:
    valid = []
    missing = []
    for name in wsi_names:
        if check_json_metadata_exists(name, json_meta_dir, json_nonmeta_dir):
            valid.append(name)
        else:
            missing.append(name)

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        print(f"[!] Missing JSON metadata for {len(missing)} WSIs (e.g., {preview}{suffix})")

    print(f"[OK] Found {len(valid)} WSIs with metadata (out of {len(wsi_names)})")
    return valid


def create_heatmap_overlay(
    wsi_name: str,
    attention_scores: np.ndarray,
    json_metadata: Dict,
    output_path: Path,
    downsample_factor: int = 32,
    colormap_name: str = "hot",
    show_colorbar: bool = True,
    pred_info: Optional[Dict] = None,
) -> None:
    if "tiles" in json_metadata:
        tiles_info = json_metadata["tiles"]
    elif "patch_coords" in json_metadata:
        tiles_info = json_metadata["patch_coords"]
    else:
        print(f"[!] {wsi_name}: JSON missing 'tiles' or 'patch_coords', skipping")
        return

    if attention_scores.max() - attention_scores.min() > 0:
        attention_scores = (
            attention_scores - attention_scores.min()
        ) / (attention_scores.max() - attention_scores.min())

    x_coords = [tile["x"] for tile in tiles_info]
    y_coords = [tile["y"] for tile in tiles_info]

    max_x = max(x_coords) + 512
    max_y = max(y_coords) + 512

    width = max_x // downsample_factor
    height = max_y // downsample_factor
    patch_size_ds = 512 // downsample_factor

    heatmap = np.zeros((height, width))
    counts = np.zeros((height, width))

    for tile_info, attn in zip(tiles_info, attention_scores):
        x = tile_info["x"] // downsample_factor
        y = tile_info["y"] // downsample_factor
        heatmap[y:y + patch_size_ds, x:x + patch_size_ds] += attn
        counts[y:y + patch_size_ds, x:x + patch_size_ds] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        heatmap = np.divide(heatmap, counts, where=counts > 0)

    fig = plt.figure(figsize=(20, 20))
    plt.imshow(heatmap, cmap=colormap_name, interpolation="nearest")
    if show_colorbar:
        plt.colorbar(label="Attention Score", shrink=0.5)

    if pred_info:
        status = "CORRECT" if pred_info["is_correct"] else "INCORRECT"
        label_text = "BRAF+" if pred_info["label"] == 1 else "BRAF-"
        pred_text = "BRAF+" if pred_info["prediction"] == 1 else "BRAF-"
        title = (
            f"{wsi_name}\n"
            f"{status} | True: {label_text} | Pred: {pred_text} | Prob: {pred_info['probability']:.4f}"
        )
    else:
        title = f"Attention Heatmap - {wsi_name}"

    plt.title(title, fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    gc.collect()

    print(f"[OK] Heatmap saved: {output_path}")


def visualize_wsi_attention_heatmaps(
    attention_scores_dict: Dict[str, Dict],
    json_meta_dir: Path,
    json_nonmeta_dir: Path,
    save_dir: Path,
    n_correct: int = 3,
    n_incorrect: int = 3,
    show_colorbar: bool = True,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    correct = []
    incorrect = []
    for wsi_name, info in attention_scores_dict.items():
        if "true_label" not in info or "predicted_label" not in info:
            continue
        is_correct = info["true_label"] == info["predicted_label"]
        if is_correct:
            correct.append(wsi_name)
        else:
            incorrect.append(wsi_name)

    correct = filter_wsis_with_metadata(correct, json_meta_dir, json_nonmeta_dir)[:n_correct]
    incorrect = filter_wsis_with_metadata(incorrect, json_meta_dir, json_nonmeta_dir)[:n_incorrect]

    for wsi_name in correct + incorrect:
        json_path = check_json_metadata_exists(wsi_name, json_meta_dir, json_nonmeta_dir)
        if not json_path:
            continue
        metadata = load_json_metadata(json_path)
        info = attention_scores_dict[wsi_name]
        pred_info = {
            "label": info.get("true_label"),
            "prediction": info.get("predicted_label"),
            "probability": info.get("pred_prob", 0.0),
            "is_correct": info.get("true_label") == info.get("predicted_label"),
        }
        output_path = save_dir / f"heatmap_{wsi_name}.png"
        create_heatmap_overlay(
            wsi_name=wsi_name,
            attention_scores=np.array(info["scores"]),
            json_metadata=metadata,
            output_path=output_path,
            show_colorbar=show_colorbar,
            pred_info=pred_info,
        )


def generate_attention_heatmaps_from_results(
    results_json_path: str,
    json_meta_dir: str,
    json_nonmeta_dir: str,
    save_dir: str,
    fold_num: str = "best",
    n_correct: int = 3,
    n_incorrect: int = 3,
    interpolation: str = "gaussian",
    dpi: int = 200,
) -> None:
    _ = interpolation
    _ = dpi

    results_path = Path(results_json_path)
    with open(results_path, "r") as f:
        results = json.load(f)

    folds = results.get("folds", [])
    if not folds:
        print("[!] No folds found in results JSON")
        return

    if fold_num == "best":
        best = max(
            folds,
            key=lambda f: f.get("test_metrics", {}).get("auc", 0.0),
        )
        target_fold = best.get("fold")
    else:
        target_fold = int(fold_num)

    attention_path = results_path.parent / "attention_scores" / f"attention_scores_fold{target_fold}.json"
    if not attention_path.exists():
        print(f"[!] Attention scores not found: {attention_path}")
        return

    with open(attention_path, "r") as f:
        attention_data = json.load(f)

    attention_scores_dict = attention_data.get("attention_scores", {})
    if not attention_scores_dict:
        print("[!] No attention scores found for visualization")
        return

    visualize_wsi_attention_heatmaps(
        attention_scores_dict=attention_scores_dict,
        json_meta_dir=Path(json_meta_dir),
        json_nonmeta_dir=Path(json_nonmeta_dir),
        save_dir=Path(save_dir),
        n_correct=n_correct,
        n_incorrect=n_incorrect,
    )
