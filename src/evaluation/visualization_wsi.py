"""
WSI Thumbnail Attention Visualization
- 패치 이미지들을 격자로 합쳐서 전체 WSI 썸네일 생성
- 각 패치 위에 attention score 오버레이 표시
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import glob


def get_patch_image_path(wsi_name, patch_idx, patch_base_dir):
    """
    WSI 이름과 patch index로 실제 patch 이미지 경로 찾기
    
    Args:
        wsi_name: WSI 파일명 (e.g., 'TC_04_3712')
        patch_idx: patch 인덱스
        patch_base_dir: patch 이미지 base 디렉토리
    
    Returns:
        patch 이미지 경로 또는 None
    """
    # BRAF+ 경로
    meta_dir = Path(patch_base_dir) / "Train" / "braf_meta" / wsi_name
    # BRAF- 경로
    nonmeta_dir = Path(patch_base_dir) / "Train" / "braf_nonmeta" / wsi_name
    
    # 두 경로 모두 확인
    for base_dir in [meta_dir, nonmeta_dir]:
        if base_dir.exists():
            # patch_{idx}.png 또는 patch_{idx}.jpg 형식 찾기
            for ext in ['.png', '.jpg', '.jpeg']:
                patch_path = base_dir / f"patch_{patch_idx}{ext}"
                if patch_path.exists():
                    return patch_path
                # 다른 naming convention도 시도
                patch_path = base_dir / f"{patch_idx}{ext}"
                if patch_path.exists():
                    return patch_path
    
    return None


def create_attention_heatmap_colormap():
    """Attention score용 colormap (blue -> green -> yellow -> red)"""
    colors = ['#2E3192', '#1BFFFF', '#00FF00', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=256)
    return cmap


def create_wsi_thumbnail_with_attention(wsi_name, patch_indices, attention_scores, 
                                        patch_base_dir, max_thumbnail_size=2048, 
                                        grid_cols=None, overlay_alpha=0.6):
    """
    패치 이미지들을 격자로 합쳐서 WSI 썸네일 생성 + attention overlay
    
    Args:
        wsi_name: WSI 파일명
        patch_indices: 패치 인덱스 리스트
        attention_scores: attention score 배열
        patch_base_dir: 패치 이미지 base 디렉토리
        max_thumbnail_size: 최종 썸네일의 최대 크기 (픽셀)
        grid_cols: 그리드 컬럼 수 (None이면 자동 계산)
        overlay_alpha: 오버레이 투명도
    
    Returns:
        PIL Image (WSI 썸네일 with attention overlay)
    """
    n_patches = len(patch_indices)
    
    # Grid 크기 계산
    if grid_cols is None:
        grid_cols = int(np.ceil(np.sqrt(n_patches)))
    grid_rows = int(np.ceil(n_patches / grid_cols))
    
    # 각 패치 크기 계산 (최종 썸네일이 max_thumbnail_size를 넘지 않도록)
    # 패치가 많으면(>1000) 더 큰 썸네일 허용
    if n_patches > 1000:
        max_size_adjusted = max_thumbnail_size * 2  # 4096px까지 허용
    else:
        max_size_adjusted = max_thumbnail_size
    
    patch_size = min(
        max_size_adjusted // grid_cols,
        max_size_adjusted // grid_rows,
        64  # 최대 크기 제한
    )
    patch_size = max(patch_size, 32)  # 최소 32px 유지 (가독성)
    
    # 전체 썸네일 이미지 크기
    thumbnail_width = grid_cols * patch_size
    thumbnail_height = grid_rows * patch_size
    
    # 빈 캔버스 생성 (흰색 배경)
    thumbnail = Image.new('RGB', (thumbnail_width, thumbnail_height), (255, 255, 255))
    
    # Colormap 생성
    cmap = create_attention_heatmap_colormap()
    min_score = attention_scores.min()
    max_score = attention_scores.max()
    
    # 각 패치를 격자에 배치
    for idx, patch_idx in enumerate(patch_indices):
        row = idx // grid_cols
        col = idx % grid_cols
        
        # 패치 이미지 로드
        patch_path = get_patch_image_path(wsi_name, patch_idx, patch_base_dir)
        
        if patch_path and patch_path.exists():
            patch_img = Image.open(patch_path)
            # 리사이즈
            patch_img = patch_img.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
            
            # Attention overlay 적용
            score = attention_scores[idx]
            norm_score = (score - min_score) / (max_score - min_score + 1e-8)
            
            # Colormap에서 색상 가져오기
            color_rgba = cmap(norm_score)
            color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
            
            # 오버레이 생성
            overlay = Image.new('RGB', patch_img.size, color_rgb)
            patch_img = Image.blend(patch_img.convert('RGB'), overlay, alpha=overlay_alpha)
        else:
            # 이미지가 없으면 회색 박스
            patch_img = Image.new('RGB', (patch_size, patch_size), (128, 128, 128))
        
        # 썸네일에 붙이기
        x = col * patch_size
        y = row * patch_size
        thumbnail.paste(patch_img, (x, y))
    
    return thumbnail, (grid_rows, grid_cols), (min_score, max_score), patch_size


def visualize_wsi_attention_thumbnail(model, dataloader, device, patch_base_dir, 
                                      save_dir, fold_num, wsi_names=None,
                                      max_thumbnail_size=2048, grid_cols=None,
                                      overlay_alpha=0.5, show_colorbar=True):
    """
    전체 WSI를 패치 썸네일로 합쳐서 attention 시각화
    
    Args:
        model: 학습된 ABMIL 모델
        dataloader: 데이터로더
        device: cuda/cpu
        patch_base_dir: patch 이미지 base 디렉토리
        save_dir: 저장 디렉토리
        fold_num: fold 번호
        wsi_names: 시각화할 WSI 이름 리스트 (None이면 전체)
        max_thumbnail_size: 최종 썸네일의 최대 크기 (픽셀)
        grid_cols: 그리드 컬럼 수 (None이면 자동)
        overlay_alpha: 오버레이 투명도 (0~1)
        show_colorbar: colorbar 표시 여부
    """
    model.eval()
    save_dir = Path(save_dir) / f"fold_{fold_num}_wsi_thumbnail"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Generating WSI Thumbnail with Attention - Fold {fold_num}")
    print(f"{'='*80}")
    
    with torch.no_grad():
        for idx, (features, label, filename) in enumerate(dataloader):
            # WSI 이름 추출
            wsi_name = filename[0] if isinstance(filename, (list, tuple)) else filename
            wsi_name = Path(wsi_name).stem
            
            # 특정 WSI만 처리
            if wsi_names is not None and wsi_name not in wsi_names:
                continue
            
            features = features.to(device)
            label = label.to(device)
            
            # Forward pass to get attention scores
            results_dict, attention_dict = model(h=features, loss_fn=None, label=None)
            
            # Attention scores 추출
            attention_scores = attention_dict['A'].cpu().numpy().squeeze()  # (N,)
            n_patches = len(attention_scores)
            
            logits = results_dict['logits']
            pred_prob = torch.softmax(logits, dim=1)[0, 1].item()
            pred_label = torch.argmax(logits, dim=1).item()
            true_label = label.item()
            
            print(f"\n[{wsi_name}]")
            print(f"  True Label: {'BRAF+' if true_label==1 else 'BRAF-'}")
            print(f"  Prediction: {'BRAF+' if pred_label==1 else 'BRAF-'} ({pred_prob:.3f})")
            print(f"  Total Patches: {n_patches}")
            print(f"  Attention Score Range: [{attention_scores.min():.4f}, {attention_scores.max():.4f}]")
            
            # 패치 인덱스 (0부터 n_patches-1)
            patch_indices = list(range(n_patches))
            
            # WSI 썸네일 생성
            print(f"  Creating WSI thumbnail...")
            wsi_thumbnail, (grid_rows, grid_cols), (min_score, max_score), patch_thumbnail_size = \
                create_wsi_thumbnail_with_attention(
                    wsi_name=wsi_name,
                    patch_indices=patch_indices,
                    attention_scores=attention_scores,
                    patch_base_dir=patch_base_dir,
                    max_thumbnail_size=max_thumbnail_size,
                    grid_cols=grid_cols,
                    overlay_alpha=overlay_alpha
                )
            
            print(f"  Grid: {grid_rows} x {grid_cols}, Patch size: {patch_thumbnail_size}px")
            print(f"  Thumbnail Size: {wsi_thumbnail.size}")
            
            # Matplotlib figure 생성 (썸네일 크기에 비례)
            # 너무 큰 figure는 메모리 문제 방지
            max_fig_size = 30
            fig_width = min(max_fig_size, wsi_thumbnail.size[0] / 150)
            fig_height = min(max_fig_size, fig_width * wsi_thumbnail.size[1] / wsi_thumbnail.size[0] + 2)
            
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # 메인 이미지
            ax = fig.add_subplot(111)
            ax.imshow(wsi_thumbnail)
            ax.axis('off')
            
            # 제목
            title = (f'{wsi_name}\n'
                    f'True: {"BRAF+" if true_label==1 else "BRAF-"} | '
                    f'Pred: {"BRAF+" if pred_label==1 else "BRAF-"} ({pred_prob:.3f})\n'
                    f'{n_patches} Patches ({grid_rows}×{grid_cols})')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # Colorbar 추가
            if show_colorbar:
                from matplotlib.colors import Normalize
                from matplotlib.cm import ScalarMappable
                
                cmap = create_attention_heatmap_colormap()
                norm = Normalize(vmin=min_score, vmax=max_score)
                sm = ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                
                # Colorbar 위치 조정
                cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', 
                                   fraction=0.046, pad=0.04)
                cbar.set_label('Attention Score (Low → High)', 
                              fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # 저장 (큰 이미지는 낮은 DPI로)
            dpi = 150 if wsi_thumbnail.size[0] < 3000 else 100
            save_path = save_dir / f"{wsi_name}_thumbnail.png"
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {save_path}")
    
    print(f"\n{'='*80}")
    print(f"[✓] All WSI thumbnails saved: {save_dir}")
    print(f"{'='*80}\n")
    
    return save_dir


# =========================
# 사용 예시
# =========================
if __name__ == "__main__":
    """
    Test용 standalone 실행
    """
    import sys
    sys.path.append('..')
    
    from abmil import ABMILModel, ABMILGatedBaseConfig
    from utils.datasets import ThyroidWSIDataset
    from torch.utils.data import DataLoader
    
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patch_base_dir = "/data/143/member/kwk/dl/thyroid/image/slide-v1-240412/patch"
    
    # Fold 1 test set 중 BRAF+ 3개만
    test_wsi_names = ['TC_04_3712', 'TC_04_4612', 'TC_04_5975']
    
    # 모델 로드 (학습된 checkpoint)
    config = ABMILGatedBaseConfig()
    model = ABMILModel(config).to(device)
    
    checkpoint_path = "path/to/your/checkpoint.pt"  # 실제 경로로 수정
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[✓] Model loaded from {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint not found, using untrained model")
    
    # Dataset 로드
    test_files = [f"/data/member/jks/Thyroid_Mutation_dataset/embeddings/meta_test_final/{name}.npy" 
                  for name in test_wsi_names]
    test_dataset = ThyroidWSIDataset(test_files, bag_size=2000, use_variance=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Visualization 생성
    visualize_wsi_attention_thumbnail(
        model=model,
        dataloader=test_loader,
        device=device,
        patch_base_dir=patch_base_dir,
        save_dir="./attention_results",
        fold_num=1,
        wsi_names=test_wsi_names,
        max_thumbnail_size=2048,  # 최종 썸네일 최대 크기 (자동으로 패치 크기 조절)
        grid_cols=None,  # 자동 계산 (정사각형에 가깝게)
        overlay_alpha=0.5,  # 오버레이 투명도
        show_colorbar=True
    )
