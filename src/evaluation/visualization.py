"""
WSI Attention Heatmap Visualization (히트맵 + SVS Overlay)

✅ v2: Updated for separated JSON structure
   - Reads attention scores from attention_scores/attention_scores_fold{N}.json
   - Reads CV metrics from results_cv_summary.json
✅ v3: SVS Thumbnail Overlay 지원 (TERT 모델에서 포팅)
   - OpenSlide로 SVS 파일 로드 → heatmap을 WSI thumbnail 위에 오버레이
   - meta_braf / non_braf 두 폴더에서 SVS 자동 탐색
"""

import os
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

try:
    import openslide
except ImportError:
    openslide = None

# Matplotlib 설정
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache_wsi"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_attention_heatmap_colormap():
    """Attention score용 colormap (blue -> green -> yellow -> red)"""
    colors = ['#2E3192', '#1BFFFF', '#00FF00', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=256)
    return cmap


def load_json_metadata(json_path: Path) -> Dict:
    """JSON 메타데이터 로드"""
    with open(json_path, 'r') as f:
        return json.load(f)


def check_json_metadata_exists(wsi_name: str, json_meta_dir: Path, json_nonmeta_dir: Path) -> Optional[Path]:
    """
    JSON 메타데이터 파일 존재 여부 확인 및 경로 반환
    
    Returns:
        JSON 파일 경로 또는 None
    """
    # .npy 확장자 제거
    wsi_name = wsi_name.replace('.npy', '').replace('.pt', '')
    
    # Try meta directory with coords_selected_ prefix
    json_path = json_meta_dir / f"coords_selected_{wsi_name}.json"
    if json_path.exists():
        return json_path
    
    # Try meta directory without prefix
    json_path = json_meta_dir / f"{wsi_name}.json"
    if json_path.exists():
        return json_path
    
    # Try nonmeta directory
    json_path = json_nonmeta_dir / f"{wsi_name}.json"
    if json_path.exists():
        return json_path
    
    return None


def extract_coordinates_from_json(metadata):
    """
    JSON 메타데이터에서 좌표 정보 추출 (meta + non-meta + list 구조 완전 지원)
    
    Args:
        metadata (dict or list): JSON 메타데이터
    
    Returns:
        coords_dict (dict): {patch_idx: (row, col)} 매핑
        grid_shape (tuple): (n_rows, n_cols)
    """
    # 1. meta / non-meta / list / index dict 구조 자동 판별
    if isinstance(metadata, dict):
        if 'tiles' in metadata:
            tiles = metadata['tiles']  # meta 구조
        elif 'patch_coords' in metadata:
            tiles = metadata['patch_coords']  # non-meta 구조
        elif 'coords' in metadata:
            tiles = metadata['coords']  # 일부 변형 구조
        else:
            # 예: {"0": {"x":..., "y":...}, "1": {...}} 형태 대응
            if all(isinstance(v, dict) and 'x' in v and 'y' in v for v in metadata.values()):
                tiles = list(metadata.values())
            else:
                raise KeyError(
                    f"Cannot find coordinate keys in JSON. Available keys: {list(metadata.keys())}"
                )
    elif isinstance(metadata, list):
        # JSON 자체가 [ {x:..., y:...}, ... ] 형태
        tiles = metadata
    else:
        raise TypeError(f"Unsupported metadata type: {type(metadata)}")

    # 2. 좌표 수집 및 유효성 검사
    x_coords = [tile['x'] for tile in tiles if 'x' in tile]
    y_coords = [tile['y'] for tile in tiles if 'y' in tile]

    if len(x_coords) == 0 or len(y_coords) == 0:
        raise ValueError(
            f"No coordinate data found in metadata. "
            f"Example entry: {tiles[0] if len(tiles) > 0 else 'EMPTY'}"
        )

    # 유니크 정렬 (좌표 → grid index 변환용)
    x_coords = sorted(set(x_coords))
    y_coords = sorted(set(y_coords))
    n_cols = len(x_coords)
    n_rows = len(y_coords)
    grid_shape = (n_rows, n_cols)

    # 3. 좌표 → (row, col) 인덱스 매핑
    x_to_col = {x: i for i, x in enumerate(x_coords)}
    y_to_row = {y: i for i, y in enumerate(y_coords)}

    coords_dict = {}
    for idx, tile in enumerate(tiles):
        if 'x' not in tile or 'y' not in tile:
            continue
        x = tile['x']
        y = tile['y']
        row = y_to_row.get(y, 0)
        col = x_to_col.get(x, 0)
        coords_dict[idx] = (row, col)

    return coords_dict, grid_shape


def extract_tiles_from_metadata(metadata: Dict) -> List[Dict]:
    """JSON 메타데이터에서 tile 리스트 추출 (flexible key handling)"""
    if isinstance(metadata, dict):
        if 'tiles' in metadata:
            tiles = metadata['tiles']
        elif 'patch_coords' in metadata:
            tiles = metadata['patch_coords']
        elif 'coords' in metadata:
            tiles = metadata['coords']
        else:
            if all(isinstance(v, dict) and 'x' in v and 'y' in v for v in metadata.values()):
                tiles = list(metadata.values())
            else:
                raise KeyError(f"Cannot find coordinate keys in JSON. Available keys: {list(metadata.keys())}")
    elif isinstance(metadata, list):
        tiles = metadata
    else:
        raise TypeError(f"Unsupported metadata type: {type(metadata)}")
    return tiles


def _estimate_patch_size(sorted_coords: List[int], default_size: int = 512) -> int:
    """좌표 간격으로 patch size 추정"""
    if len(sorted_coords) < 2:
        return default_size
    diffs = np.diff(sorted_coords)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return default_size
    return int(np.median(diffs))


def _find_svs_path(wsi_name: str, svs_base_dir: Optional[Path]) -> Optional[Path]:
    """
    SVS 파일 경로 탐색 (meta_braf / non_braf 두 폴더 지원)

    디렉토리 구조:
        svs_base_dir/
            meta_braf/TC_04_XXXX.svs
            non_braf/TC_04_XXXX.svs
    """
    if svs_base_dir is None:
        return None

    # 1) 직접 경로
    path = svs_base_dir / f"{wsi_name}.svs"
    if path.exists():
        return path

    # 2) meta_braf 하위
    path = svs_base_dir / "meta_braf" / f"{wsi_name}.svs"
    if path.exists():
        return path

    # 3) non_braf 하위
    path = svs_base_dir / "non_braf" / f"{wsi_name}.svs"
    if path.exists():
        return path

    return None


def save_heatmap_thumbnail_overlay(
    wsi_name: str,
    heatmap: np.ndarray,
    metadata: Dict,
    save_dir: Path,
    case_type: str,
    case_idx: int,
    cmap,
    svs_base_dir: Optional[Path],
    thumbnail_max_side: int = 2048,
    overlay_alpha: float = 0.85,
    overlay_gamma: float = 0.60,
    overlay_alpha_floor: float = 0.15,
) -> Optional[Path]:
    """SVS thumbnail 위에 heatmap을 오버레이하여 저장"""
    if openslide is None:
        print("  [i] Skipping overlay: openslide-python not installed")
        return None

    svs_path = _find_svs_path(wsi_name, svs_base_dir)
    if svs_path is None:
        print(f"  [i] Skipping overlay: SVS not found for {wsi_name}")
        return None

    tiles = extract_tiles_from_metadata(metadata)
    x_coords = sorted(set([t["x"] for t in tiles if "x" in t]))
    y_coords = sorted(set([t["y"] for t in tiles if "y" in t]))
    if len(x_coords) == 0 or len(y_coords) == 0:
        print(f"  [i] Skipping overlay: invalid coordinates for {wsi_name}")
        return None

    patch_w = _estimate_patch_size(x_coords, default_size=512)
    patch_h = _estimate_patch_size(y_coords, default_size=512)

    min_x, max_x = x_coords[0], x_coords[-1]
    min_y, max_y = y_coords[0], y_coords[-1]
    bbox_x0 = int(min_x)
    bbox_y0 = int(min_y)
    bbox_x1 = int(max_x + patch_w)
    bbox_y1 = int(max_y + patch_h)

    slide = openslide.OpenSlide(str(svs_path))
    slide_w, slide_h = slide.dimensions
    scale = min(float(thumbnail_max_side) / max(slide_w, slide_h), 1.0)
    thumb_w = max(1, int(slide_w * scale))
    thumb_h = max(1, int(slide_h * scale))
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    slide.close()

    thumb_np = np.array(thumb).astype(np.float32) / 255.0

    x0 = max(0, min(thumb_w - 1, int(round(bbox_x0 * scale))))
    y0 = max(0, min(thumb_h - 1, int(round(bbox_y0 * scale))))
    x1 = max(x0 + 1, min(thumb_w, int(round(bbox_x1 * scale))))
    y1 = max(y0 + 1, min(thumb_h, int(round(bbox_y1 * scale))))

    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32),
        (x1 - x0, y1 - y0),
        interpolation=cv2.INTER_LINEAR,
    )

    score = np.clip(heatmap_resized, 0.0, 1.0)
    score_enhanced = np.power(score, overlay_gamma)
    heat_rgb = cmap(score_enhanced)[..., :3]

    alpha_mask = np.where(
        score > 0,
        (overlay_alpha_floor + (1.0 - overlay_alpha_floor) * score_enhanced) * overlay_alpha,
        0.0,
    )
    alpha_mask = np.clip(alpha_mask, 0.0, 1.0)[..., None]

    region = thumb_np[y0:y1, x0:x1, :]
    blended = region * (1.0 - alpha_mask) + heat_rgb * alpha_mask
    thumb_np[y0:y1, x0:x1, :] = blended

    overlay_img = np.clip(thumb_np * 255.0, 0, 255).astype(np.uint8)
    overlay_path = save_dir / f"{case_type}_{case_idx:02d}_{wsi_name}_overlay.png"
    plt.figure(figsize=(12, 8))
    plt.imshow(overlay_img)
    plt.title(f"{wsi_name} - Heatmap Overlay", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200, bbox_inches="tight")
    plt.close()

    return overlay_path


def create_attention_heatmap(attention_scores, coords_dict, grid_shape,
                             patch_indices=None, interpolation='gaussian'):
    """Attention scores를 spatial heatmap으로 변환"""
    n_rows, n_cols = grid_shape
    heatmap = np.zeros((n_rows, n_cols))
    count_map = np.zeros((n_rows, n_cols))
    
    if patch_indices is None:
        patch_indices = range(len(attention_scores))
    
    for patch_idx in patch_indices:
        if patch_idx >= len(attention_scores):
            continue
        
        score = attention_scores[patch_idx]
        
        if patch_idx in coords_dict:
            row, col = coords_dict[patch_idx]
            if 0 <= row < n_rows and 0 <= col < n_cols:
                heatmap[row, col] += score
                count_map[row, col] += 1
    
    mask = count_map > 0
    heatmap[mask] /= count_map[mask]
    
    if interpolation == 'gaussian':
        if np.sum(~mask) > 0:
            heatmap_filled = cv2.inpaint(
                (heatmap * 255).astype(np.uint8),
                (~mask).astype(np.uint8),
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            ) / 255.0
        else:
            heatmap_filled = heatmap
        
        heatmap = gaussian_filter(heatmap_filled, sigma=1.0)
    
    elif interpolation == 'bilinear':
        from scipy.ndimage import zoom
        scale = 4
        heatmap_upscaled = zoom(heatmap, scale, order=1)
        heatmap = zoom(heatmap_upscaled, 1/scale, order=1)
    
    return heatmap


def visualize_single_heatmap(wsi_name, wsi_data, json_meta_dir, json_nonmeta_dir,
                             save_dir, cmap, case_type='correct', case_idx=1,
                             interpolation='gaussian', dpi=200,
                             svs_base_dir=None, thumbnail_max_side=2048,
                             overlay_alpha=0.85, overlay_gamma=0.60,
                             overlay_alpha_floor=0.15, title_prefix=''):
    """단일 WSI의 attention heatmap 시각화 + SVS overlay"""
    
    attention_scores = np.array(wsi_data['scores'])
    n_patches = wsi_data['n_patches']
    true_label = wsi_data.get('true_label', None)
    pred_label = wsi_data.get('predicted_label', None)
    pred_prob = wsi_data.get('pred_prob', None)
    
    case_type_text = {
        'correct': '✓ Correct',
        'incorrect': '✗ Incorrect',
        'positive': 'BRAF+',
        'negative': 'BRAF-',
    }.get(case_type, case_type)

    print(f"\n[{case_idx}] {wsi_name}")
    print(f"  Type: {case_type_text}")
    if true_label is not None:
        print(f"  True Label: {'BRAF+' if true_label==1 else 'BRAF-'}")
    if pred_label is not None and pred_prob is not None:
        print(f"  Predicted: {'BRAF+' if pred_label==1 else 'BRAF-'} (prob={pred_prob:.3f})")
    print(f"  Patches: {n_patches}")
    print(f"  Score Range (Raw): [{attention_scores.min():.6f}, {attention_scores.max():.6f}]")
    
    # Min-Max 정규화 (가시성 향상)
    if attention_scores.max() > attention_scores.min():
        attention_scores_normalized = (attention_scores - attention_scores.min()) / \
                                      (attention_scores.max() - attention_scores.min())
    else:
        attention_scores_normalized = attention_scores
    
    print(f"  Score Range (Normalized): [{attention_scores_normalized.min():.3f}, {attention_scores_normalized.max():.3f}]")
    
    # JSON 메타데이터 로드
    json_path = check_json_metadata_exists(wsi_name, json_meta_dir, json_nonmeta_dir)
    
    if json_path is None:
        print(f"  ⚠️ Skipping: Cannot load JSON metadata")
        return
    
    metadata = load_json_metadata(json_path)
    print(f"  📁 Loaded: {json_path.name}")
    
    # 좌표 정보 추출
    coords_dict, grid_shape = extract_coordinates_from_json(metadata)
    
    print(f"  Grid Shape: {grid_shape[0]} rows × {grid_shape[1]} cols")
    
    # 정규화된 scores로 heatmap 생성
    heatmap = create_attention_heatmap(
        attention_scores_normalized,
        coords_dict, 
        grid_shape, 
        patch_indices=range(n_patches),
        interpolation=interpolation
    )
    
    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(heatmap, cmap=cmap, aspect='auto', interpolation='bilinear')
    
    title_parts = []
    if title_prefix:
        title_parts.append(f'[{title_prefix}] {wsi_name}')
    else:
        title_parts.append(f'{wsi_name}')
    if case_type == 'correct':
        title_parts.append('✓ Correct Prediction')
    elif case_type == 'incorrect':
        title_parts.append('✗ Incorrect Prediction')
    elif case_type == 'positive':
        title_parts.append('BRAF+ Case')
    elif case_type == 'negative':
        title_parts.append('BRAF- Case')
    else:
        title_parts.append(str(case_type))
    
    if true_label is not None and pred_label is not None:
        title_parts.append(
            f'True: {"BRAF+" if true_label==1 else "BRAF-"} | '
            f'Pred: {"BRAF+" if pred_label==1 else "BRAF-"} ({pred_prob:.3f})'
        )
    
    ax.set_title('\n'.join(title_parts), fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'Attention Score (Normalized)\n[0.000, 1.000]',
                   fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_xlabel('Column Index', fontsize=11)
    ax.set_ylabel('Row Index', fontsize=11)
    ax.grid(False)
    
    filename = f"{case_type}_{case_idx:02d}_{wsi_name}_heatmap.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {filename}")

    # SVS Thumbnail Overlay
    if svs_base_dir is not None:
        overlay_path = save_heatmap_thumbnail_overlay(
            wsi_name=wsi_name,
            heatmap=heatmap,
            metadata=metadata,
            save_dir=save_dir,
            case_type=case_type,
            case_idx=case_idx,
            cmap=cmap,
            svs_base_dir=Path(svs_base_dir) if isinstance(svs_base_dir, str) else svs_base_dir,
            thumbnail_max_side=thumbnail_max_side,
            overlay_alpha=overlay_alpha,
            overlay_gamma=overlay_gamma,
            overlay_alpha_floor=overlay_alpha_floor,
        )
        if overlay_path is not None:
            print(f"  ✓ Saved: {overlay_path.name}")

    gc.collect()


def generate_attention_heatmaps_from_results(
    results_dir: str,
    json_meta_dir: str,
    json_nonmeta_dir: str,
    save_dir: str,
    fold_num: str = 'best',
    n_positive: int = 5,
    n_negative: int = 5,
    interpolation: str = 'gaussian',
    dpi: int = 200,
    svs_base_dir: Optional[str] = None,
    thumbnail_max_side: int = 2048,
    overlay_alpha: float = 0.85,
    overlay_gamma: float = 0.60,
    overlay_alpha_floor: float = 0.15,
):
    """
    Wrapper function for generating attention heatmaps from separated JSON structure
    ✅ v4: Positive/Negative 기반 선택 (BRAF+ / BRAF-)
       - Reads CV summary from results_cv_summary.json
       - Reads attention scores from attention_scores/attention_scores_fold{N}.json
       - SVS Thumbnail Overlay 지원

    Args:
        results_dir: Directory containing results (not a JSON file, but a directory!)
        json_meta_dir: Directory containing JSON metadata for meta cases
        json_nonmeta_dir: Directory containing JSON metadata for nonmeta cases
        save_dir: Directory to save heatmaps
        fold_num: 'best' or fold number (int)
        n_positive: Number of positive (BRAF+) cases to visualize
        n_negative: Number of negative (BRAF-) cases to visualize
        interpolation: Interpolation method ('gaussian' or 'bilinear')
        dpi: DPI for saved images
        svs_base_dir: SVS 파일 베이스 디렉토리 (meta_braf/, non_braf/ 하위 자동 탐색)
        thumbnail_max_side: Overlay thumbnail 최대 변 길이
        overlay_alpha: Overlay 전체 투명도
        overlay_gamma: Overlay 감마 보정 (<1 = mid-range 강화)
        overlay_alpha_floor: 비제로 영역 최소 alpha
    """
    print(f"\n{'='*80}")
    print(f"Loading results from: {results_dir}")
    print(f"{'='*80}\n")
    
    results_dir = Path(results_dir)
    json_meta_dir = Path(json_meta_dir)
    json_nonmeta_dir = Path(json_nonmeta_dir)
    
    # ============================================================
    # 1. Load CV Summary JSON
    # ============================================================
    # Try optimal threshold version first (new format), fallback to old format
    cv_summary_path = results_dir / 'results_cv_summary_optimal.json'
    if not cv_summary_path.exists():
        cv_summary_path = results_dir / 'results_cv_summary.json'

    if not cv_summary_path.exists():
        print(f"❌ CV summary not found: {cv_summary_path}")
        print(f"   Tried: results_cv_summary_optimal.json and results_cv_summary.json")
        return
    
    with open(cv_summary_path, 'r') as f:
        cv_summary = json.load(f)
    
    print(f"✓ Loaded CV summary from: {cv_summary_path}")
    
    # ============================================================
    # 2. Select fold
    # ============================================================
    if fold_num == 'best':
        # Find best fold by test AUC
        best_fold = max(cv_summary['folds'], key=lambda x: x['test_metrics']['auc'])
        fold_idx = best_fold['fold']
        print(f"✓ Selected best fold: Fold {fold_idx} (AUC: {best_fold['test_metrics']['auc']:.4f})")
    else:
        fold_idx = int(fold_num)
        best_fold = next((f for f in cv_summary['folds'] if f['fold'] == fold_idx), None)
        if not best_fold:
            print(f"❌ Fold {fold_idx} not found in results")
            return
        print(f"✓ Selected fold: Fold {fold_idx}")
    
    # ============================================================
    # 3. Load Attention Scores JSON (from separate file)
    # ============================================================
    attention_dir = results_dir / 'attention_scores'
    attention_path = attention_dir / f'attention_scores_fold{fold_idx}.json'
    
    if not attention_path.exists():
        print(f"❌ Attention scores not found: {attention_path}")
        print(f"   Make sure you ran training with --generate_plots flag")
        return
    
    with open(attention_path, 'r') as f:
        attention_data = json.load(f)
    
    attention_scores_dict = attention_data['attention_scores']
    print(f"✓ Loaded attention scores from: {attention_path}")
    print(f"✓ Found attention scores for {len(attention_scores_dict)} WSIs")
    
    # ============================================================
    # 4. Prepare prediction results
    # ============================================================
    prediction_results = {}
    for wsi_name, attn_data in attention_scores_dict.items():
        if 'true_label' in attn_data and 'predicted_label' in attn_data:
            prediction_results[wsi_name] = {
                'label': attn_data['true_label'],
                'prediction': attn_data['predicted_label'],
                'probability': attn_data.get('pred_prob', 0.5),
                'is_correct': attn_data['true_label'] == attn_data['predicted_label']
            }
    
    print(f"✓ Prepared prediction results for {len(prediction_results)} WSIs")
    
    # ============================================================
    # 5. Select positive/negative cases (BRAF+ / BRAF-)
    # ============================================================
    positive_cases = []
    negative_cases = []

    for wsi_name, pred_info in prediction_results.items():
        if pred_info['label'] == 1:
            positive_cases.append((wsi_name, attention_scores_dict[wsi_name]))
        else:
            negative_cases.append((wsi_name, attention_scores_dict[wsi_name]))

    # Sort by confidence (highest first)
    positive_cases.sort(
        key=lambda x: abs(x[1].get('pred_prob', 0.5) - 0.5),
        reverse=True
    )
    negative_cases.sort(
        key=lambda x: abs(x[1].get('pred_prob', 0.5) - 0.5),
        reverse=True
    )

    # Select top N
    positive_cases = positive_cases[:n_positive]
    negative_cases = negative_cases[:n_negative]

    print(f"\n📊 Selected Cases (Fold {fold_idx}):")
    print(f"  Positive (BRAF+): {len(positive_cases)}/{n_positive}")
    print(f"  Negative (BRAF-): {len(negative_cases)}/{n_negative}")

    # ============================================================
    # 6. Create save directory
    # ============================================================
    save_dir = Path(save_dir) / f"fold_{fold_idx}_attention_heatmaps"
    save_dir.mkdir(parents=True, exist_ok=True)

    cmap = create_attention_heatmap_colormap()

    # ============================================================
    # 7. Generate heatmaps
    # ============================================================
    print(f"\n{'─'*80}")
    print(f"Processing Positive Cases - BRAF+ ({len(positive_cases)})")
    print(f"{'─'*80}")

    for idx, (wsi_name, wsi_data) in enumerate(positive_cases, 1):
        visualize_single_heatmap(
            wsi_name, wsi_data, json_meta_dir, json_nonmeta_dir, save_dir, cmap,
            case_type='positive', case_idx=idx, interpolation=interpolation, dpi=dpi,
            svs_base_dir=svs_base_dir, thumbnail_max_side=thumbnail_max_side,
            overlay_alpha=overlay_alpha, overlay_gamma=overlay_gamma,
            overlay_alpha_floor=overlay_alpha_floor,
        )

    print(f"\n{'─'*80}")
    print(f"Processing Negative Cases - BRAF- ({len(negative_cases)})")
    print(f"{'─'*80}")

    for idx, (wsi_name, wsi_data) in enumerate(negative_cases, 1):
        visualize_single_heatmap(
            wsi_name, wsi_data, json_meta_dir, json_nonmeta_dir, save_dir, cmap,
            case_type='negative', case_idx=idx, interpolation=interpolation, dpi=dpi,
            svs_base_dir=svs_base_dir, thumbnail_max_side=thumbnail_max_side,
            overlay_alpha=overlay_alpha, overlay_gamma=overlay_gamma,
            overlay_alpha_floor=overlay_alpha_floor,
        )
    
    print(f"\n{'='*80}")
    print(f"✓ Attention heatmap generation complete!")
    print(f"✓ All heatmaps saved to: {save_dir}")
    print(f"{'='*80}\n")


def generate_ensemble_attention_heatmaps(
    all_model_attention_scores: List[Dict[str, Dict]],
    ensemble_probs: List[float],
    test_labels: List[int],
    test_filenames: List[str],
    json_meta_dir: str,
    json_nonmeta_dir: str,
    save_dir: str,
    n_positive: int = 5,
    n_negative: int = 5,
    interpolation: str = 'gaussian',
    dpi: int = 200,
    svs_base_dir: Optional[str] = None,
    thumbnail_max_side: int = 2048,
    overlay_alpha: float = 0.85,
    overlay_gamma: float = 0.60,
    overlay_alpha_floor: float = 0.15,
    best_model_idx: int = 0,
):
    """
    Ensemble attention heatmap generation

    1) Best individual model (val AUC best): positive N + negative N (heatmap + SVS overlay)
    2) Ensemble average attention (5 models mean): positive N + negative N (heatmap + SVS overlay)

    Args:
        all_model_attention_scores: List of 5 dicts, each {wsi_name: {scores, n_patches, ...}}
        ensemble_probs: Ensemble averaged probabilities (list of float)
        test_labels: Ground truth labels (list of int)
        test_filenames: Test set filenames (list of str)
        json_meta_dir: JSON metadata directory for meta cases
        json_nonmeta_dir: JSON metadata directory for nonmeta cases
        save_dir: Output directory
        n_positive/n_negative: Number of positive/negative cases to visualize
        best_model_idx: Index of best model (0-based) in all_model_attention_scores
        svs_base_dir: SVS base directory for overlay
    """
    print(f"\n{'='*80}")
    print(f"Generating Ensemble Attention Heatmaps")
    print(f"{'='*80}\n")

    save_dir = Path(save_dir)
    json_meta_dir = Path(json_meta_dir)
    json_nonmeta_dir = Path(json_nonmeta_dir)
    cmap = create_attention_heatmap_colormap()

    # Build prediction results from ensemble probs
    prediction_results = {}
    for filename, prob, label in zip(test_filenames, ensemble_probs, test_labels):
        wsi_name = Path(filename).stem
        pred = 1 if prob >= 0.5 else 0
        prediction_results[wsi_name] = {
            'label': int(label),
            'prediction': pred,
            'probability': float(prob),
            'is_correct': pred == int(label),
        }

    # ============================================================
    # A) Best individual model heatmaps
    # ============================================================
    best_attention = all_model_attention_scores[best_model_idx]
    available_wsis = set(best_attention.keys())

    # Select class-balanced samples (only WSIs with available attention in best model)
    positive_cases = [
        (wsi, info) for wsi, info in prediction_results.items()
        if info['label'] == 1 and wsi in available_wsis
    ]
    negative_cases = [
        (wsi, info) for wsi, info in prediction_results.items()
        if info['label'] == 0 and wsi in available_wsis
    ]

    # Sort by confidence (highest first)
    positive_cases.sort(key=lambda x: abs(x[1]['probability'] - 0.5), reverse=True)
    negative_cases.sort(key=lambda x: abs(x[1]['probability'] - 0.5), reverse=True)

    positive_cases = positive_cases[:n_positive]
    negative_cases = negative_cases[:n_negative]
    selected_wsis = [wsi for wsi, _ in positive_cases] + [wsi for wsi, _ in negative_cases]

    print(f"  Positive selected: {len(positive_cases)}/{n_positive}")
    print(f"  Negative selected: {len(negative_cases)}/{n_negative}")
    print(f"  Total selected: {len(selected_wsis)} WSIs")
    if len(positive_cases) < n_positive:
        print(f"  [i] Positive cases are fewer than requested ({n_positive})")
    if len(negative_cases) < n_negative:
        print(f"  [i] Negative cases are fewer than requested ({n_negative})")

    individual_dir = save_dir / "individual_model_best"
    individual_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*80}")
    print(f"[A] Best Individual Model (model {best_model_idx + 1}) Heatmaps")
    print(f"    Save dir: {individual_dir}")
    print(f"{'─'*80}")

    _generate_heatmaps_for_selected(
        attention_dict=best_attention,
        positive_wsis=[wsi for wsi, _ in positive_cases],
        negative_wsis=[wsi for wsi, _ in negative_cases],
        prediction_results=prediction_results,
        json_meta_dir=json_meta_dir,
        json_nonmeta_dir=json_nonmeta_dir,
        save_dir=individual_dir,
        cmap=cmap,
        interpolation=interpolation,
        dpi=dpi,
        svs_base_dir=svs_base_dir,
        thumbnail_max_side=thumbnail_max_side,
        overlay_alpha=overlay_alpha,
        overlay_gamma=overlay_gamma,
        overlay_alpha_floor=overlay_alpha_floor,
        title_prefix="Best Model",
    )

    # ============================================================
    # B) Ensemble average attention heatmaps
    # ============================================================
    ensemble_dir = save_dir / "ensemble_attention"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*80}")
    print(f"[B] Ensemble Average Attention (5 models) Heatmaps")
    print(f"    Save dir: {ensemble_dir}")
    print(f"{'─'*80}")

    # Average attention scores across 5 models
    ensemble_attention = {}
    all_wsi_names = set()
    for attn_dict in all_model_attention_scores:
        all_wsi_names.update(attn_dict.keys())

    for wsi_name in all_wsi_names:
        model_scores = []
        n_patches = None
        for attn_dict in all_model_attention_scores:
            if wsi_name in attn_dict:
                scores = np.array(attn_dict[wsi_name]['scores'])
                model_scores.append(scores)
                if n_patches is None:
                    n_patches = attn_dict[wsi_name]['n_patches']

        if len(model_scores) == 0:
            continue

        # Average across models
        avg_scores = np.mean(model_scores, axis=0)
        ensemble_attention[wsi_name] = {
            'scores': avg_scores.tolist(),
            'n_patches': n_patches,
            'num_models_averaged': len(model_scores),
        }

    print(f"  Averaged attention for {len(ensemble_attention)} WSIs "
          f"(from {len(all_model_attention_scores)} models)")

    _generate_heatmaps_for_selected(
        attention_dict=ensemble_attention,
        positive_wsis=[wsi for wsi, _ in positive_cases],
        negative_wsis=[wsi for wsi, _ in negative_cases],
        prediction_results=prediction_results,
        json_meta_dir=json_meta_dir,
        json_nonmeta_dir=json_nonmeta_dir,
        save_dir=ensemble_dir,
        cmap=cmap,
        interpolation=interpolation,
        dpi=dpi,
        svs_base_dir=svs_base_dir,
        thumbnail_max_side=thumbnail_max_side,
        overlay_alpha=overlay_alpha,
        overlay_gamma=overlay_gamma,
        overlay_alpha_floor=overlay_alpha_floor,
        title_prefix="Ensemble (5-model avg)",
    )

    print(f"\n{'='*80}")
    print(f"Ensemble attention heatmap generation complete!")
    print(f"  Individual: {individual_dir}")
    print(f"  Ensemble:   {ensemble_dir}")
    print(f"{'='*80}\n")


def _generate_heatmaps_for_selected(
    attention_dict: Dict[str, Dict],
    positive_wsis: List[str],
    negative_wsis: List[str],
    prediction_results: Dict[str, Dict],
    json_meta_dir: Path,
    json_nonmeta_dir: Path,
    save_dir: Path,
    cmap,
    interpolation: str = 'gaussian',
    dpi: int = 200,
    svs_base_dir: Optional[str] = None,
    thumbnail_max_side: int = 2048,
    overlay_alpha: float = 0.85,
    overlay_gamma: float = 0.60,
    overlay_alpha_floor: float = 0.15,
    title_prefix: str = "",
):
    """Helper: generate heatmap + SVS overlay for selected positive/negative WSIs."""

    for case_type, wsi_list in [('positive', positive_wsis), ('negative', negative_wsis)]:
        print(f"\n  Processing {case_type} cases ({len(wsi_list)})")

        for idx, wsi_name in enumerate(wsi_list, 1):
            if wsi_name not in attention_dict:
                print(f"    [{idx}] {wsi_name} - no attention data, skipping")
                continue

            wsi_data = attention_dict[wsi_name]

            # Add prediction info from ensemble results
            pred_info = prediction_results.get(wsi_name, {})
            wsi_data_with_pred = wsi_data.copy()
            wsi_data_with_pred['true_label'] = pred_info.get('label')
            wsi_data_with_pred['predicted_label'] = pred_info.get('prediction')
            wsi_data_with_pred['pred_prob'] = pred_info.get('probability')

            visualize_single_heatmap(
                wsi_name=wsi_name,
                wsi_data=wsi_data_with_pred,
                json_meta_dir=json_meta_dir,
                json_nonmeta_dir=json_nonmeta_dir,
                save_dir=save_dir,
                cmap=cmap,
                case_type=case_type,
                case_idx=idx,
                interpolation=interpolation,
                dpi=dpi,
                svs_base_dir=svs_base_dir,
                thumbnail_max_side=thumbnail_max_side,
                overlay_alpha=overlay_alpha,
                overlay_gamma=overlay_gamma,
                overlay_alpha_floor=overlay_alpha_floor,
                title_prefix=title_prefix,
            )


if __name__ == '__main__':
    print("This is a library module. Import and use the functions in your training script.")
