"""
UNI2-h 모델을 TorchScript로 변환

사용법:
    python export_uni2h_torchscript.py --output_dir exports/uni2h_torchscript
"""

import os
import argparse
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import numpy as np


def build_uni2h_model(device: str = "cuda"):
    """
    UNI2-h 모델 로드
    """
    print("[INFO] Loading UNI2-h model from HuggingFace...")

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,  # feature extraction mode
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=True,
        **timm_kwargs
    )
    model.to(device)
    model.eval()

    # Transform 설정
    data_cfg = resolve_data_config({}, model=model)
    transform = create_transform(**data_cfg)

    print(f"[INFO] UNI2-h loaded on {device}")
    print(f"[INFO] Output dim: 1536")

    return model, transform, data_cfg


def export_torchscript(model, output_path: str, device: str = "cuda"):
    """
    UNI2-h를 TorchScript로 변환
    """
    model.eval()

    # 예시 입력 (224x224 RGB 이미지)
    example_input = torch.randn(1, 3, 224, 224).to(device)

    print(f"[INFO] Tracing model with input shape: {example_input.shape}")

    with torch.no_grad():
        # Trace (strict=False for compatibility)
        traced_model = torch.jit.trace(model, example_input, strict=False)
        # Note: optimize_for_inference() 제거 - 로드 시 호환성 문제 발생

    # 저장
    traced_model.save(output_path)
    print(f"[✓] Exported TorchScript model to: {output_path}")

    return traced_model


def verify_model(original_model, traced_model, device: str = "cuda"):
    """
    변환된 모델 검증
    """
    example_input = torch.randn(1, 3, 224, 224).to(device)

    original_model.eval()
    traced_model.eval()

    with torch.no_grad():
        original_output = original_model(example_input)
        traced_output = traced_model(example_input)

    is_close = torch.allclose(original_output, traced_output, rtol=1e-4, atol=1e-4)

    if is_close:
        print(f"[✓] Verification passed!")
        print(f"    Output shape: {traced_output.shape}")
    else:
        max_diff = (original_output - traced_output).abs().max().item()
        print(f"[✗] Verification failed! Max diff: {max_diff}")

    return is_close


def save_transform_info(data_cfg: dict, output_path: str):
    """
    Transform 설정 저장 (전처리에 필요)
    """
    info = f"""UNI2-h Transform Configuration
================================

Input Size: {data_cfg.get('input_size', (3, 224, 224))}
Interpolation: {data_cfg.get('interpolation', 'bicubic')}
Mean: {data_cfg.get('mean', (0.485, 0.456, 0.406))}
Std: {data_cfg.get('std', (0.229, 0.224, 0.225))}

Preprocessing Steps:
1. Resize to 224x224 (bicubic interpolation)
2. Convert to tensor (0-1 range)
3. Normalize with ImageNet mean/std

Code Example:
-------------
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
"""
    with open(output_path, 'w') as f:
        f.write(info)
    print(f"[✓] Transform info saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export UNI2-h to TorchScript')
    parser.add_argument('--output_dir', type=str, default='exports/uni2h_torchscript',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    # 디바이스 설정
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        args.device = "cpu"

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 모델 로드
    print("\n" + "=" * 60)
    print("Step 1: Loading UNI2-h Model")
    print("=" * 60)
    model, transform, data_cfg = build_uni2h_model(args.device)

    # 2. TorchScript 변환
    print("\n" + "=" * 60)
    print("Step 2: Converting to TorchScript")
    print("=" * 60)
    output_path = os.path.join(args.output_dir, "uni2h.pt")
    traced_model = export_torchscript(model, output_path, args.device)

    # 3. 검증
    print("\n" + "=" * 60)
    print("Step 3: Verification")
    print("=" * 60)
    verify_model(model, traced_model, args.device)

    # 4. Transform 정보 저장
    print("\n" + "=" * 60)
    print("Step 4: Saving Transform Info")
    print("=" * 60)
    info_path = os.path.join(args.output_dir, "uni2h_info.txt")
    save_transform_info(data_cfg, info_path)

    # 5. 모델 정보 저장
    model_info_path = os.path.join(args.output_dir, "uni2h_model_info.txt")
    model_size = os.path.getsize(output_path) / (1024 * 1024)

    with open(model_info_path, 'w') as f:
        f.write(f"""UNI2-h TorchScript Model
========================

Model: UNI2-h (MahmoodLab)
Source: hf-hub:MahmoodLab/UNI2-h
Type: Vision Transformer (ViT)

Input:
  - Shape: [batch_size, 3, 224, 224]
  - Type: float32
  - Range: Normalized (ImageNet mean/std)

Output:
  - Shape: [batch_size, 1536]
  - Type: float32
  - Description: Feature embedding vector

Architecture:
  - Image size: 224x224
  - Patch size: 14
  - Embed dim: 1536
  - Depth: 24
  - Num heads: 24
  - MLP: SwiGLU

File size: {model_size:.2f} MB
PyTorch version: {torch.__version__}
""")

    print(f"[✓] Model info saved to: {model_info_path}")

    # 최종 요약
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"  Model:     {output_path}")
    print(f"  Size:      {model_size:.2f} MB")
    print(f"  Info:      {model_info_path}")
    print(f"  Transform: {info_path}")


if __name__ == '__main__':
    main()
