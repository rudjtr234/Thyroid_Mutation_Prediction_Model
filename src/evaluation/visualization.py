"""
WSI Attention Heatmap Visualization (히트맵만 생성)

✅ v2: Updated for separated JSON structure
   - Reads attention scores from attention_scores/attention_scores_fold{N}.json
   - Reads CV metrics from results_cv_summary.json
"""

import os
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

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
                             interpolation='gaussian', dpi=200):
    """단일 WSI의 attention heatmap 시각화"""
    
    attention_scores = np.array(wsi_data['scores'])
    n_patches = wsi_data['n_patches']
    true_label = wsi_data.get('true_label', None)
    pred_label = wsi_data.get('predicted_label', None)
    pred_prob = wsi_data.get('pred_prob', None)
    
    print(f"\n[{case_idx}] {wsi_name}")
    print(f"  Type: {'✓ Correct' if case_type == 'correct' else '✗ Incorrect'}")
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
    
    title_parts = [f'{wsi_name}']
    if case_type == 'correct':
        title_parts.append('✓ Correct Prediction')
    else:
        title_parts.append('✗ Incorrect Prediction')
    
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
    
    gc.collect()


def generate_attention_heatmaps_from_results(
    results_dir: str,
    json_meta_dir: str,
    json_nonmeta_dir: str,
    save_dir: str,
    fold_num: str = 'best',
    n_correct: int = 3,
    n_incorrect: int = 3,
    interpolation: str = 'gaussian',
    dpi: int = 200
):
    """
    Wrapper function for generating attention heatmaps from separated JSON structure
    ✅ v2: Updated to work with separated JSONs
       - Reads CV summary from results_cv_summary.json
       - Reads attention scores from attention_scores/attention_scores_fold{N}.json
    
    Args:
        results_dir: Directory containing results (not a JSON file, but a directory!)
        json_meta_dir: Directory containing JSON metadata for meta cases
        json_nonmeta_dir: Directory containing JSON metadata for nonmeta cases
        save_dir: Directory to save heatmaps
        fold_num: 'best' or fold number (int)
        n_correct: Number of correct predictions to visualize
        n_incorrect: Number of incorrect predictions to visualize
        interpolation: Not used (for compatibility)
        dpi: DPI for saved images
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
    # 5. Select correct/incorrect cases
    # ============================================================
    correct_cases = []
    incorrect_cases = []
    
    for wsi_name, pred_info in prediction_results.items():
        if pred_info['is_correct']:
            correct_cases.append((wsi_name, attention_scores_dict[wsi_name]))
        else:
            incorrect_cases.append((wsi_name, attention_scores_dict[wsi_name]))
    
    # Sort by confidence
    correct_cases.sort(
        key=lambda x: abs(x[1].get('pred_prob', 0.5) - 0.5), 
        reverse=True
    )
    incorrect_cases.sort(
        key=lambda x: abs(x[1].get('pred_prob', 0.5) - 0.5), 
        reverse=True
    )
    
    # Select top N
    correct_cases = correct_cases[:n_correct]
    incorrect_cases = incorrect_cases[:n_incorrect]
    
    print(f"\n📊 Selected Cases (Fold {fold_idx}):")
    print(f"  Correct predictions: {len(correct_cases)}/{n_correct}")
    print(f"  Incorrect predictions: {len(incorrect_cases)}/{n_incorrect}")
    
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
    print(f"Processing Correct Predictions ({len(correct_cases)})")
    print(f"{'─'*80}")
    
    for idx, (wsi_name, wsi_data) in enumerate(correct_cases, 1):
        visualize_single_heatmap(
            wsi_name, wsi_data, json_meta_dir, json_nonmeta_dir, save_dir, cmap,
            case_type='correct', case_idx=idx, interpolation=interpolation, dpi=dpi
        )
    
    print(f"\n{'─'*80}")
    print(f"Processing Incorrect Predictions ({len(incorrect_cases)})")
    print(f"{'─'*80}")
    
    for idx, (wsi_name, wsi_data) in enumerate(incorrect_cases, 1):
        visualize_single_heatmap(
            wsi_name, wsi_data, json_meta_dir, json_nonmeta_dir, save_dir, cmap,
            case_type='incorrect', case_idx=idx, interpolation=interpolation, dpi=dpi
        )
    
    print(f"\n{'='*80}")
    print(f"✓ Attention heatmap generation complete!")
    print(f"✓ All heatmaps saved to: {save_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    print("This is a library module. Import and use the functions in your training script.")