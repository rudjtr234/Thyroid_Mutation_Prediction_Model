"""
500개 Bag Attention Score Overlay Visualization (원본 패치 이미지 + 반투명 오버레이)
- 원본 패치 이미지 위에 attention score를 반투명 색상으로 오버레이
- 저장된 results.json의 attention scores 사용
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def get_patch_image_path(wsi_name, patch_idx, patch_base_dir):
    """WSI 이름과 patch index로 실제 patch 이미지 경로 찾기"""
    # BRAF+ 경로
    meta_dir = Path(patch_base_dir) / "Train" / "braf_meta" / wsi_name
    # BRAF- 경로
    nonmeta_dir = Path(patch_base_dir) / "Train" / "braf_nonmeta" / wsi_name
    
    # 두 경로 모두 확인
    for base_dir in [meta_dir, nonmeta_dir]:
        if base_dir.exists():
            # 다양한 naming convention 시도
            for ext in ['.png', '.jpg', '.jpeg']:
                for pattern in [f"patch_{patch_idx}{ext}", f"{patch_idx}{ext}"]:
                    patch_path = base_dir / pattern
                    if patch_path.exists():
                        return patch_path
    
    return None


def create_attention_heatmap_colormap():
    """Attention score용 colormap (blue -> green -> yellow -> red)"""
    colors = ['#2E3192', '#1BFFFF', '#00FF00', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=256)
    return cmap


def add_score_overlay(img, score, cmap, min_score, max_score, alpha=0.5):
    """
    패치 이미지에 attention score 오버레이 추가
    
    Args:
        img: PIL Image (원본 패치)
        score: attention score
        cmap: colormap
        min_score: 최소 score
        max_score: 최대 score
        alpha: 오버레이 투명도 (0~1, 낮을수록 원본이 잘 보임)
    """
    # Score 정규화
    norm_score = (score - min_score) / (max_score - min_score + 1e-8)
    
    # Colormap에서 색상 가져오기
    color_rgba = cmap(norm_score)
    color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
    
    # 색상 오버레이 생성
    overlay = Image.new('RGB', img.size, color_rgb)
    
    # 원본 이미지와 블렌딩 (alpha가 낮을수록 원본이 더 보임)
    blended = Image.blend(img.convert('RGB'), overlay, alpha=alpha)
    
    return blended


def visualize_500_patches_with_overlay(results_json_path, patch_base_dir, save_dir,
                                       fold_num=1, wsi_names=None,
                                       thumbnail_size=(96, 96), 
                                       grid_layout=(25, 20),  # (n_cols, n_rows)
                                       overlay_alpha=0.4,  # 0.4 = 원본 60% + 오버레이 40%
                                       dpi=200):
    """
    500개 패치의 원본 이미지 위에 attention score를 오버레이로 시각화
    
    Args:
        results_json_path: results.json 파일 경로
        patch_base_dir: 패치 이미지 base 디렉토리
        save_dir: 저장 디렉토리
        fold_num: 시각화할 fold 번호
        wsi_names: 시각화할 WSI 이름 리스트 (None이면 첫 3개)
        thumbnail_size: 썸네일 크기 (픽셀)
        grid_layout: (n_cols, n_rows) 그리드 레이아웃
        overlay_alpha: 오버레이 투명도 (0~1, 낮을수록 원본이 더 보임)
        dpi: 저장 이미지 해상도
    """
    # results.json 로드
    print(f"\n{'='*80}")
    print(f"Loading attention scores from: {results_json_path}")
    print(f"{'='*80}")
    
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
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
    
    attention_scores_dict = fold_data['test_attention_scores']
    
    # 저장 디렉토리 생성
    save_dir = Path(save_dir) / f"fold_{fold_num}_500patches_overlay"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    cmap = create_attention_heatmap_colormap()
    n_cols, n_rows = grid_layout
    
    print(f"\n{'='*80}")
    print(f"Generating 500-Patch Attention Overlays - Fold {fold_num}")
    print(f"Grid Layout: {n_cols} cols × {n_rows} rows = {n_cols * n_rows} patches")
    print(f"Thumbnail Size: {thumbnail_size}")
    print(f"Overlay Alpha: {overlay_alpha} (원본 {(1-overlay_alpha)*100:.0f}% + 오버레이 {overlay_alpha*100:.0f}%)")
    print(f"{'='*80}")
    print(f"Total WSIs in JSON: {len(attention_scores_dict)}")
    
    # wsi_names가 None이면 첫 3개만 선택
    if wsi_names is None:
        wsi_names = list(attention_scores_dict.keys())[:3]
        print(f"Processing first 3 WSIs: {wsi_names}")
    else:
        print(f"Processing {len(wsi_names)} specified WSIs")
    
    processed_count = 0
    for wsi_name, wsi_data in attention_scores_dict.items():
        # 특정 WSI만 처리
        if wsi_name not in wsi_names:
            continue
        
        processed_count += 1
        
        # Attention scores와 메타데이터 추출
        attention_scores = np.array(wsi_data['scores'])
        n_patches = wsi_data['n_patches']
        true_label = wsi_data.get('true_label', None)
        pred_label = wsi_data.get('predicted_label', None)
        pred_prob = wsi_data.get('pred_prob', None)
        
        print(f"\n[{processed_count}/3] {wsi_name}")
        if true_label is not None:
            print(f"  True: {'BRAF+' if true_label==1 else 'BRAF-'}", end="")
        if pred_label is not None and pred_prob is not None:
            print(f" | Pred: {'BRAF+' if pred_label==1 else 'BRAF-'} ({pred_prob:.3f})")
        print(f"  Patches: {n_patches}")
        print(f"  Score Range: [{attention_scores.min():.6f}, {attention_scores.max():.6f}]")
        print(f"  Score Mean: {attention_scores.mean():.6f} ± {attention_scores.std():.6f}")
        
        # Figure 크기 계산
        fig_width = n_cols * (thumbnail_size[0] / 100) * 1.1
        fig_height = n_rows * (thumbnail_size[1] / 100) * 1.15
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # axes를 2D 배열로 변환
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        min_score = attention_scores.min()
        max_score = attention_scores.max()
        
        # 각 패치 시각화
        patches_found = 0
        patches_missing = 0
        
        for patch_idx in range(min(n_patches, n_cols * n_rows)):
            row = patch_idx // n_cols
            col = patch_idx % n_cols
            ax = axes[row, col]
            
            score = attention_scores[patch_idx]
            
            # 패치 이미지 로드
            patch_path = get_patch_image_path(wsi_name, patch_idx, patch_base_dir)
            
            if patch_path and patch_path.exists():
                # 원본 이미지 로드
                img = Image.open(patch_path)
                img = img.resize(thumbnail_size, Image.Resampling.LANCZOS)
                
                # Attention score 오버레이 추가
                img_with_overlay = add_score_overlay(
                    img, score, cmap, min_score, max_score, alpha=overlay_alpha
                )
                
                ax.imshow(img_with_overlay)
                patches_found += 1
            else:
                # 이미지가 없으면 colormap 색상으로만 채움
                norm_score = (score - min_score) / (max_score - min_score + 1e-8)
                color_rgba = cmap(norm_score)
                color_rgb = color_rgba[:3]
                
                ax.imshow(np.ones((thumbnail_size[1], thumbnail_size[0], 3)) * color_rgb)
                patches_missing += 1
            
            ax.axis('off')
        
        # 빈 subplot 제거
        for i in range(n_patches, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < n_rows and col < n_cols:
                fig.delaxes(axes[row, col])
        
        # 전체 제목
        title_parts = [f'{wsi_name} - {n_patches} Patches']
        if true_label is not None and pred_label is not None:
            title_parts.append(
                f'True: {"BRAF+" if true_label==1 else "BRAF-"} | '
                f'Pred: {"BRAF+" if pred_label==1 else "BRAF-"} ({pred_prob:.3f})'
            )
        
        fig.suptitle('\n'.join(title_parts), fontsize=14, fontweight='bold', y=0.995)
        
        # Colorbar 추가
        sm = plt.cm.ScalarMappable(cmap=cmap, 
                                  norm=plt.Normalize(vmin=min_score, vmax=max_score))
        sm.set_array([])
        
        cbar_ax = fig.add_axes([0.15, 0.005, 0.7, 0.01])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(f'Attention Score: {min_score:.6f} → {max_score:.6f}', 
                      fontsize=9, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)
        
        plt.subplots_adjust(left=0.01, right=0.99, top=0.975, bottom=0.025, 
                          wspace=0.02, hspace=0.02)
        
        # 저장
        save_path = save_dir / f"{wsi_name}_500patches_overlay.png"
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {save_path.name}")
        print(f"    Found: {patches_found}/{n_patches} patches", end="")
        if patches_missing > 0:
            print(f" | Missing: {patches_missing}")
        else:
            print()
    
    print(f"\n{'='*80}")
    print(f"[✓] All 500-patch attention overlays saved to:")
    print(f"    {save_dir}")
    print(f"{'='*80}\n")
    
    return save_dir


# =========================
# 사용 예시
# =========================
if __name__ == "__main__":
    """
    사용 예시: 원본 패치 이미지 + attention score 오버레이
    """
    
    # 경로 설정
    results_json_path = "/home/mts/ssd_16tb/member/jks/Thyroid_Mutation_model_v2/outputs/Thyroid_prediction_model_v0.2.0/results.json"
    patch_base_dir = "/data/143/member/kwk/dl/thyroid/image/slide-v1-240412/patch"
    save_dir = "./attention_visualizations"
    
    # 기본 설정 (첫 3개 WSI)
    visualize_500_patches_with_overlay(
        results_json_path=results_json_path,
        patch_base_dir=patch_base_dir,
        save_dir=save_dir,
        fold_num=1,
        wsi_names=None,  # None이면 첫 3개 자동 선택
        thumbnail_size=(96, 96),  # 각 패치 썸네일 크기
        grid_layout=(25, 20),  # 25 cols × 20 rows = 500
        overlay_alpha=0.4,  # 원본 60% + 오버레이 40%
        dpi=200
    )
    
    # 더 강한 오버레이 효과 (원본이 덜 보임)
    """
    visualize_500_patches_with_overlay(
        results_json_path=results_json_path,
        patch_base_dir=patch_base_dir,
        save_dir=save_dir,
        fold_num=1,
        wsi_names=['TC_04_7900'],
        thumbnail_size=(128, 128),  # 더 큰 썸네일
        grid_layout=(25, 20),
        overlay_alpha=0.6,  # 원본 40% + 오버레이 60%
        dpi=300
    )
    """