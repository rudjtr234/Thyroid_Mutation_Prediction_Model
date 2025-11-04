"""
WSI Attention Heatmap Visualization (히트맵만 생성)

이 스크립트는 attention heatmap만 생성합니다.
썸네일은 시간이 오래 걸리므로 주석 처리되었습니다.
✨ Modified: 정답 3개 + 오답 3개 선택 (JSON 메타데이터 존재하는 것만)
"""

import os
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Matplotlib 설정
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache_wsi"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_json_metadata(json_path: Path) -> Dict:
    """JSON 메타데이터 로드"""
    with open(json_path, 'r') as f:
        return json.load(f)


def check_json_metadata_exists(wsi_name: str, json_metadata_dir: Path) -> bool:
    """JSON 메타데이터 파일 존재 여부 확인"""
    json_path = json_metadata_dir / f"coords_meta_{wsi_name}.json"
    return json_path.exists()


def filter_wsis_with_metadata(
    wsi_names: List[str],
    json_metadata_dir: Path
) -> List[str]:
    """
    JSON 메타데이터가 존재하는 WSI만 필터링
    
    Args:
        wsi_names: 전체 WSI 이름 리스트
        json_metadata_dir: JSON 메타데이터 디렉토리
    
    Returns:
        메타데이터가 존재하는 WSI 리스트
    """
    valid_wsis = []
    missing_wsis = []
    
    for wsi_name in wsi_names:
        if check_json_metadata_exists(wsi_name, json_metadata_dir):
            valid_wsis.append(wsi_name)
        else:
            missing_wsis.append(wsi_name)
    
    if missing_wsis:
        print(f"\n  ⚠️ Missing JSON metadata for {len(missing_wsis)} WSIs:")
        for wsi in missing_wsis[:5]:  # 처음 5개만 출력
            print(f"     - {wsi}")
        if len(missing_wsis) > 5:
            print(f"     ... and {len(missing_wsis) - 5} more")
    
    print(f"\n  ✅ Found {len(valid_wsis)} WSIs with metadata (out of {len(wsi_names)})")
    
    return valid_wsis


def create_heatmap_overlay(
    wsi_name: str,
    attention_scores: np.ndarray,
    json_metadata: Dict,
    output_path: Path,
    downsample_factor: int = 32,
    colormap_name: str = 'hot',
    show_colorbar: bool = True,
    pred_info: Optional[Dict] = None
):
    """
    실제 WSI 좌표에 맞춰 attention heatmap 생성
    """
    tiles_info = json_metadata['tiles']

    # 좌표 범위 계산
    x_coords = [tile['x'] for tile in tiles_info]
    y_coords = [tile['y'] for tile in tiles_info]

    max_x = max(x_coords) + 512
    max_y = max(y_coords) + 512

    # Heatmap 배열 크기
    width = max_x // downsample_factor
    height = max_y // downsample_factor
    patch_size_ds = 512 // downsample_factor

    print(f"    Creating heatmap: {width}x{height} pixels")

    # Heatmap 초기화
    heatmap = np.zeros((height, width))
    counts = np.zeros((height, width))

    # Attention을 좌표에 매핑
    for tile_info, attn in zip(tiles_info, attention_scores):
        x = tile_info['x'] // downsample_factor
        y = tile_info['y'] // downsample_factor

        heatmap[y:y+patch_size_ds, x:x+patch_size_ds] += attn
        counts[y:y+patch_size_ds, x:x+patch_size_ds] += 1

    # 평균 계산 (중복 영역)
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap = np.divide(heatmap, counts, where=counts>0)

    # 시각화
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(heatmap, cmap=colormap_name, interpolation='nearest')

    if show_colorbar:
        plt.colorbar(label='Attention Score', shrink=0.5)

    # Title with prediction info
    if pred_info:
        status = "✅ CORRECT" if pred_info['is_correct'] else "❌ INCORRECT"
        label_text = "BRAF+" if pred_info['label'] == 1 else "BRAF-"
        pred_text = "BRAF+" if pred_info['prediction'] == 1 else "BRAF-"
        title = (f"{wsi_name}\n"
                 f"{status} | True: {label_text} | Pred: {pred_text} | Prob: {pred_info['probability']:.4f}")
    else:
        title = f'Attention Heatmap - {wsi_name}'

    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.close('all')
    gc.collect()

    print(f"    ✓ Heatmap saved: {output_path}")


def visualize_wsi_attention_thumbnail(
    model,
    dataloader,
    device,
    patch_base_dir: str,
    save_dir: Path,
    fold_num: int,
    wsi_names: List[str],
    max_thumbnail_size: int = 4500,
    overlay_alpha: float = 0.5,
    show_colorbar: bool = True,
    precomputed_attention: Optional[Dict[str, np.ndarray]] = None,
    prediction_results: Optional[Dict[str, Dict]] = None,
    num_workers: int = 16
):
    """
    WSI attention 히트맵 시각화 메인 함수 (썸네일 제외)
    ✨ 정답 3개 + 오답 3개 선택 (JSON 메타데이터 존재하는 것만)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # JSON 메타데이터 디렉토리
    json_metadata_dir = Path("/data/143/member/jks/Thyroid_Mutation_dataset/json_metadata")

    if precomputed_attention is None:
        print("  ⚠ No precomputed attention provided")
        return

    # 🔥 STEP 1: JSON 메타데이터가 존재하는 WSI만 필터링
    valid_wsi_names = filter_wsis_with_metadata(wsi_names, json_metadata_dir)
    
    if not valid_wsi_names:
        print("\n  ❌ No WSIs with valid JSON metadata found!")
        return

    # 🔥 STEP 2: 정답 3개 + 오답 3개 선택 (메타데이터 존재하는 것 중에서)
    selected_wsis = []

    if prediction_results:
        # 메타데이터가 있는 것 중에서 정답/오답 분류
        correct_wsis = [wsi for wsi in valid_wsi_names
                       if wsi in prediction_results and prediction_results[wsi]['is_correct']]
        incorrect_wsis = [wsi for wsi in valid_wsi_names
                         if wsi in prediction_results and not prediction_results[wsi]['is_correct']]

        # Confidence 기준으로 정렬
        correct_wsis.sort(
            key=lambda x: abs(prediction_results[x]['probability'] - 0.5), 
            reverse=True
        )
        incorrect_wsis.sort(
            key=lambda x: abs(prediction_results[x]['probability'] - 0.5), 
            reverse=True
        )

        # 🔥 각각 3개씩 선택 (또는 가능한 만큼)
        n_correct = min(3, len(correct_wsis))
        n_incorrect = min(3, len(incorrect_wsis))
        
        selected_wsis.extend(correct_wsis[:n_correct])
        selected_wsis.extend(incorrect_wsis[:n_incorrect])

        print(f"\n  📊 Selected {len(selected_wsis)} WSIs for visualization:")
        print(f"     ✅ Correct: {n_correct} (out of {len(correct_wsis)} available)")
        print(f"     ❌ Incorrect: {n_incorrect} (out of {len(incorrect_wsis)} available)")
        
        if len(selected_wsis) < 6:
            print(f"\n  ⚠️ Warning: Only {len(selected_wsis)} WSIs selected (target: 6)")
            print(f"     This is because only {len(valid_wsi_names)} WSIs have JSON metadata")
    else:
        # prediction_results가 없으면 처음 6개 (또는 가능한 만큼)
        selected_wsis = valid_wsi_names[:min(6, len(valid_wsi_names))]
        print(f"\n  ⚠ No prediction results provided. Visualizing first {len(selected_wsis)} WSIs")

    # 🔥 STEP 3: 선택된 WSI 시각화
    for idx, wsi_name in enumerate(selected_wsis, 1):
        if wsi_name not in precomputed_attention:
            print(f"\n  [{idx}/{len(selected_wsis)}] ⚠ No attention data for {wsi_name}")
            continue

        # 정답 여부 표시
        status_str = ""
        pred_info = None
        if prediction_results and wsi_name in prediction_results:
            pred_info = prediction_results[wsi_name]
            is_correct = pred_info['is_correct']
            status = "CORRECT" if is_correct else "INCORRECT"
            status_emoji = "✅" if is_correct else "❌"
            status_str = f" [{status_emoji} {status}]"

            print(f"\n  [{idx}/{len(selected_wsis)}] Processing: {wsi_name}{status_str}")
            print(f"    Label: {'BRAF+' if pred_info['label']==1 else 'BRAF-'}, "
                  f"Pred: {'BRAF+' if pred_info['prediction']==1 else 'BRAF-'}, "
                  f"Prob: {pred_info['probability']:.4f}")
        else:
            print(f"\n  [{idx}/{len(selected_wsis)}] Processing: {wsi_name}")

        # JSON 메타데이터 로드
        json_path = json_metadata_dir / f"coords_meta_{wsi_name}.json"
        
        try:
            metadata = load_json_metadata(json_path)
            attention_scores = precomputed_attention[wsi_name]

            # 파일명 태그
            if prediction_results and wsi_name in prediction_results:
                status_tag = "correct" if prediction_results[wsi_name]['is_correct'] else "incorrect"
            else:
                status_tag = "unknown"

            # Heatmap 생성
            print(f"    Creating heatmap overlay...")
            heatmap_path = save_dir / f"{wsi_name}_heatmap_{status_tag}.png"
            create_heatmap_overlay(
                wsi_name=wsi_name,
                attention_scores=attention_scores,
                json_metadata=metadata,
                output_path=heatmap_path,
                show_colorbar=show_colorbar,
                pred_info=pred_info
            )

            gc.collect()
            
        except Exception as e:
            print(f"    ❌ Error processing {wsi_name}: {str(e)}")
            continue

    print(f"\n  ✓ All heatmaps complete! ({len(selected_wsis)} WSIs processed)")


if __name__ == '__main__':
    print("This is a library module. Import and use visualize_wsi_attention_thumbnail()")