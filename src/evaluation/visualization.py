"""
WSI Attention Heatmap Visualization (Heatmap Only)
- Test set에서 정답 3장 + 오답 3장 선택
- 가장 성능 좋은 fold 자동 선택
- JSON 메타데이터에서 좌표 정보 추출 (meta + non-meta)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import cv2


def create_attention_heatmap_colormap():
    """Attention score용 colormap (blue -> green -> yellow -> red)"""
    colors = ['#2E3192', '#1BFFFF', '#00FF00', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=256)
    return cmap


def get_best_fold_num(results_json_path, metric='auc'):
    """가장 성능이 좋은 fold 번호 반환"""
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    best_fold = max(results['folds'], key=lambda x: x['test_metrics'][metric])
    best_fold_num = best_fold['fold']
    best_metric_value = best_fold['test_metrics'][metric]
    
    print(f"\n{'='*80}")
    print(f"Best Fold Selection (Based on Test {metric.upper()})")
    print(f"{'='*80}")
    print(f"Best Fold: {best_fold_num}")
    print(f"Test {metric.upper()}: {best_metric_value:.4f}")
    print(f"{'='*80}\n")
    
    return best_fold_num


def load_json_metadata(wsi_name, json_meta_dir, json_nonmeta_dir):
    """
    JSON 메타데이터 파일에서 좌표 정보 로드 (meta + non-meta 지원)
    
    Args:
        wsi_name: WSI 이름 (예: TC_04_3947)
        json_meta_dir: meta JSON 디렉토리 경로
        json_nonmeta_dir: non-meta JSON 디렉토리 경로
    
    Returns:
        metadata: JSON 메타데이터 딕셔너리
    """
    # 1. meta 경로 시도 (coords_selected_ 접두사)
    json_path_meta = Path(json_meta_dir) / f"coords_selected_{wsi_name}.json"
    
    # 2. non-meta 경로 시도 (접두사 없음)
    json_path_nonmeta = Path(json_nonmeta_dir) / f"{wsi_name}.json"
    
    # meta 경로 먼저 확인
    if json_path_meta.exists():
        with open(json_path_meta, 'r') as f:
            metadata = json.load(f)
        print(f"  📁 Loaded from meta: {json_path_meta.name}")
        return metadata
    
    # non-meta 경로 확인
    elif json_path_nonmeta.exists():
        with open(json_path_nonmeta, 'r') as f:
            metadata = json.load(f)
        print(f"  📁 Loaded from non-meta: {json_path_nonmeta.name}")
        return metadata
    
    # 둘 다 없으면 None
    else:
        print(f"  ⚠️ Warning: JSON metadata not found for {wsi_name}")
        print(f"      Tried: {json_path_meta}")
        print(f"      Tried: {json_path_nonmeta}")
        return None


def extract_coordinates_from_json(metadata):
    """
    JSON 메타데이터에서 좌표 정보 추출
    
    Args:
        metadata: JSON 메타데이터 딕셔너리
    
    Returns:
        coords_dict: {patch_idx: (row, col)} 딕셔너리
        grid_shape: (n_rows, n_cols) 그리드 크기
    """
    tiles = metadata['tiles']
    
    # 모든 좌표 수집
    x_coords = []
    y_coords = []
    
    for tile in tiles:
        x_coords.append(tile['x'])
        y_coords.append(tile['y'])
    
    # 유니크한 좌표 정렬
    x_coords = sorted(set(x_coords))
    y_coords = sorted(set(y_coords))
    
    n_cols = len(x_coords)
    n_rows = len(y_coords)
    grid_shape = (n_rows, n_cols)
    
    # 좌표 → 그리드 인덱스 매핑
    x_to_col = {x: i for i, x in enumerate(x_coords)}
    y_to_row = {y: i for i, y in enumerate(y_coords)}
    
    # 패치 인덱스 → 그리드 좌표 매핑
    coords_dict = {}
    for idx, tile in enumerate(tiles):
        x = tile['x']
        y = tile['y']
        row = y_to_row[y]
        col = x_to_col[x]
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
        from scipy.ndimage import gaussian_filter
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


def select_correct_incorrect_cases(fold_data, n_correct=3, n_incorrect=3):
    """정답 케이스와 오답 케이스를 선택"""
    attention_scores_dict = fold_data.get('test_attention_scores', {})
    
    correct_cases = []
    incorrect_cases = []
    
    for wsi_name, wsi_data in attention_scores_dict.items():
        true_label = wsi_data.get('true_label')
        pred_label = wsi_data.get('predicted_label')
        
        if true_label is None or pred_label is None:
            continue
        
        if true_label == pred_label:
            correct_cases.append((wsi_name, wsi_data))
        else:
            incorrect_cases.append((wsi_name, wsi_data))
    
    correct_cases.sort(key=lambda x: abs(x[1].get('pred_prob', 0.5) - 0.5), reverse=True)
    incorrect_cases.sort(key=lambda x: abs(x[1].get('pred_prob', 0.5) - 0.5), reverse=True)
    
    return correct_cases[:n_correct], incorrect_cases[:n_incorrect]


def visualize_attention_heatmaps(results_json_path, json_meta_dir, json_nonmeta_dir, save_dir,
                                fold_num='best', n_correct=3, n_incorrect=3,
                                interpolation='gaussian', dpi=200):
    """
    Test set에서 정답/오답 케이스의 attention heatmap 시각화
    
    Args:
        results_json_path: results.json 파일 경로
        json_meta_dir: meta JSON 메타데이터 디렉토리 경로
        json_nonmeta_dir: non-meta JSON 메타데이터 디렉토리 경로
        save_dir: 저장 디렉토리
        fold_num: 시각화할 fold 번호 ('best' 또는 정수)
        n_correct: 정답 케이스 개수
        n_incorrect: 오답 케이스 개수
        interpolation: 'none', 'bilinear', 'gaussian'
        dpi: 저장 이미지 해상도
    """
    print(f"\n{'='*80}")
    print(f"WSI Attention Heatmap Visualization")
    print(f"{'='*80}")
    
    # results.json 로드
    print(f"Loading: {results_json_path}")
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    # 최고 성능 fold 자동 선택
    if fold_num == 'best':
        fold_num = get_best_fold_num(results_json_path, metric='auc')
    
    # 해당 fold 찾기
    fold_data = None
    for fold in results['folds']:
        if fold['fold'] == fold_num:
            fold_data = fold
            break
    
    if fold_data is None:
        raise ValueError(f"Fold {fold_num} not found in results.json")
    
    if 'test_attention_scores' not in fold_data:
        raise ValueError(f"No attention scores found in fold {fold_num}")
    
    # 정답/오답 케이스 선택
    correct_cases, incorrect_cases = select_correct_incorrect_cases(
        fold_data, n_correct, n_incorrect
    )
    
    print(f"\n📊 Selected Cases (Fold {fold_num}):")
    print(f"  Correct predictions: {len(correct_cases)}/{n_correct}")
    print(f"  Incorrect predictions: {len(incorrect_cases)}/{n_incorrect}")
    
    # 저장 디렉토리
    save_dir = Path(save_dir) / f"fold_{fold_num}_attention_heatmaps"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    cmap = create_attention_heatmap_colormap()
    
    # 정답 케이스 시각화
    print(f"\n{'─'*80}")
    print(f"Processing Correct Predictions ({len(correct_cases)})")
    print(f"{'─'*80}")
    
    for idx, (wsi_name, wsi_data) in enumerate(correct_cases, 1):
        visualize_single_heatmap(
            wsi_name, wsi_data, json_meta_dir, json_nonmeta_dir, save_dir, cmap,
            case_type='correct', case_idx=idx, interpolation=interpolation, dpi=dpi
        )
    
    # 오답 케이스 시각화
    print(f"\n{'─'*80}")
    print(f"Processing Incorrect Predictions ({len(incorrect_cases)})")
    print(f"{'─'*80}")
    
    for idx, (wsi_name, wsi_data) in enumerate(incorrect_cases, 1):
        visualize_single_heatmap(
            wsi_name, wsi_data, json_meta_dir, json_nonmeta_dir, save_dir, cmap,
            case_type='incorrect', case_idx=idx, interpolation=interpolation, dpi=dpi
        )
    
    print(f"\n{'='*80}")
    print(f"[✓] All attention heatmaps saved to:")
    print(f"    {save_dir}")
    print(f"{'='*80}\n")
    
    return save_dir


def visualize_single_heatmap(wsi_name, wsi_data, json_meta_dir, json_nonmeta_dir, save_dir, cmap,
                             case_type='correct', case_idx=1, 
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
    print(f"  Score Range: [{attention_scores.min():.6f}, {attention_scores.max():.6f}]")
    
    # JSON 메타데이터 로드 (meta + non-meta 자동 탐색)
    metadata = load_json_metadata(wsi_name, json_meta_dir, json_nonmeta_dir)
    
    if metadata is None:
        print(f"  ⚠️ Skipping: Cannot load JSON metadata")
        return
    
    # 좌표 정보 추출
    coords_dict, grid_shape = extract_coordinates_from_json(metadata)
    
    print(f"  Grid Shape: {grid_shape[0]} rows × {grid_shape[1]} cols")
    print(f"  Total tiles in JSON: {len(metadata['tiles'])}")
    
    # Attention heatmap 생성
    heatmap = create_attention_heatmap(
        attention_scores, coords_dict, grid_shape, 
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
    cbar.set_label(f'Attention Score\n[{attention_scores.min():.4f}, {attention_scores.max():.4f}]',
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


# =========================
# 사용 예시
# =========================
if __name__ == "__main__":
    """
    사용 예시: meta + non-meta JSON 메타데이터 사용
    """
    
    # 경로 설정
    results_json_path = "/home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model_v2/outputs/Thyroid_prediction_model_v0.5.0/results.json"
    
    # meta JSON 경로
    json_meta_dir = "/data/143/member/jks/Thyroid_Mutation_dataset/embeddings/final_meta_dataset_v0.1.0/json_metadata"
    
    # non-meta JSON 경로
    json_nonmeta_dir = "/data/143/member/jks/Thyroid_Mutation_dataset/embeddings/final_nonmeta_dataset_v0.1.0/json"
    
    save_dir = "./attention_heatmaps"
    
    # 최고 성능 fold 자동 선택
    visualize_attention_heatmaps(
        results_json_path=results_json_path,
        json_meta_dir=json_meta_dir,
        json_nonmeta_dir=json_nonmeta_dir,
        save_dir=save_dir,
        fold_num='best',
        n_correct=3,
        n_incorrect=3,
        interpolation='gaussian',
        dpi=200
    )