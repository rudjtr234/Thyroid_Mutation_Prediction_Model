"""
TorchScript 모델 변환 스크립트

ABMIL 모델을 TorchScript로 변환하여 Triton Server 배포용 .pt 파일 생성

사용법:
    python export_torchscript.py \
        --checkpoint outputs/Thyroid_prediction_model_v0.7.0/checkpoints/best_model_fold1_auc0.9491.pt \
        --output_dir exports/abmil_torchscript \
        --model_name thyroid_abmil
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.layers import GlobalGatedAttention, create_mlp


class ABMILForExport(nn.Module):
    """
    TorchScript 변환을 위한 ABMIL 모델 (단순화된 forward)

    Triton Server에서 사용할 추론 전용 모델
    - 입력: [batch_size, num_patches, 1536] 임베딩
    - 출력: [batch_size, 2] logits (BRAF-/BRAF+ 확률)
    """

    def __init__(self,
                 in_dim: int = 1536,
                 embed_dim: int = 512,
                 attn_dim: int = 384,
                 num_fc_layers: int = 1,
                 dropout: float = 0.0,  # 추론 시 dropout 비활성화
                 num_classes: int = 2):
        super().__init__()

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        self.global_attn = GlobalGatedAttention(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        단순화된 forward (추론 전용)

        Args:
            h: [batch_size, num_patches, 1536] UNI2-h 임베딩

        Returns:
            logits: [batch_size, 2] BRAF 분류 logits
        """
        # Patch embedding
        h = self.patch_embed(h)  # [B, N, 512]

        # Attention
        A = self.global_attn(h)  # [B, N, 1]
        A = torch.transpose(A, -2, -1)  # [B, 1, N]
        A = torch.softmax(A, dim=-1)  # Softmax over patches

        # Attention pooling
        h = torch.bmm(A, h).squeeze(1)  # [B, 512]

        # Classification
        logits = self.classifier(h)  # [B, 2]

        return logits

    def forward_with_attention(self, h: torch.Tensor) -> tuple:
        """
        Attention 가중치도 반환하는 forward (시각화용)

        Args:
            h: [batch_size, num_patches, 1536] UNI2-h 임베딩

        Returns:
            logits: [batch_size, 2] BRAF 분류 logits
            attention: [batch_size, num_patches] attention 가중치
        """
        # Patch embedding
        h = self.patch_embed(h)  # [B, N, 512]

        # Attention
        A = self.global_attn(h)  # [B, N, 1]
        A = torch.transpose(A, -2, -1)  # [B, 1, N]
        attention = torch.softmax(A, dim=-1).squeeze(1)  # [B, N]

        # Attention pooling
        h = torch.bmm(A.unsqueeze(1) if A.dim() == 2 else A,
                      self.patch_embed(h) if h.dim() == 2 else h)
        h = torch.bmm(attention.unsqueeze(1), self.patch_embed.forward(h) if hasattr(self, '_h_embed') else h)

        # 다시 계산
        h_orig = self.patch_embed(h) if h.shape[-1] == 1536 else h
        h = torch.bmm(attention.unsqueeze(1), h_orig).squeeze(1)

        # Classification
        logits = self.classifier(h)  # [B, 2]

        return logits, attention


class ABMILWithAttention(nn.Module):
    """
    Attention 출력도 포함하는 ABMIL (Triton ensemble용)
    """

    def __init__(self,
                 in_dim: int = 1536,
                 embed_dim: int = 512,
                 attn_dim: int = 384,
                 num_fc_layers: int = 1,
                 dropout: float = 0.0,
                 num_classes: int = 2):
        super().__init__()

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False
        )

        self.global_attn = GlobalGatedAttention(
            L=embed_dim,
            D=attn_dim,
            dropout=dropout,
            num_classes=1
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, h: torch.Tensor) -> tuple:
        """
        Args:
            h: [batch_size, num_patches, 1536]

        Returns:
            tuple: (logits [B, 2], attention [B, N])
        """
        # Patch embedding
        h_embed = self.patch_embed(h)  # [B, N, 512]

        # Attention
        A = self.global_attn(h_embed)  # [B, N, 1]
        A = torch.transpose(A, -2, -1)  # [B, 1, N]
        attention = torch.softmax(A, dim=-1)  # [B, 1, N]

        # Attention pooling
        h_pooled = torch.bmm(attention, h_embed).squeeze(1)  # [B, 512]

        # Classification
        logits = self.classifier(h_pooled)  # [B, 2]

        # Attention을 [B, N] 형태로
        attention_out = attention.squeeze(1)  # [B, N]

        return logits, attention_out


def load_checkpoint_weights(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    기존 ABMIL 체크포인트에서 가중치 로드
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # state_dict 키 확인
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 키 매핑 (필요시)
    new_state_dict = {}
    for key, value in state_dict.items():
        # 'model.' prefix 제거 (있는 경우)
        new_key = key.replace('model.', '') if key.startswith('model.') else key
        new_state_dict[new_key] = value

    # 로드
    model.load_state_dict(new_state_dict, strict=True)
    print(f"[✓] Loaded weights from: {checkpoint_path}")

    # Config 정보 출력
    if 'config' in checkpoint:
        print(f"[INFO] Model config: {checkpoint['config']}")

    return model


def export_torchscript(model: nn.Module,
                       output_path: str,
                       example_input: torch.Tensor,
                       use_trace: bool = True) -> str:
    """
    모델을 TorchScript로 변환

    Args:
        model: PyTorch 모델
        output_path: 저장 경로
        example_input: 예시 입력 (tracing용)
        use_trace: True면 trace, False면 script

    Returns:
        저장된 파일 경로
    """
    model.eval()

    with torch.no_grad():
        if use_trace:
            # Tracing (더 안정적)
            traced_model = torch.jit.trace(model, example_input)
        else:
            # Scripting (동적 제어 흐름 지원)
            traced_model = torch.jit.script(model)

    # 최적화
    traced_model = torch.jit.optimize_for_inference(traced_model)

    # 저장
    traced_model.save(output_path)
    print(f"[✓] Exported TorchScript model to: {output_path}")

    return output_path


def verify_exported_model(original_model: nn.Module,
                          exported_path: str,
                          example_input: torch.Tensor,
                          rtol: float = 1e-5,
                          atol: float = 1e-5) -> bool:
    """
    변환된 모델 검증
    """
    original_model.eval()

    # 원본 출력
    with torch.no_grad():
        original_output = original_model(example_input)

    # TorchScript 모델 로드 및 출력
    loaded_model = torch.jit.load(exported_path)
    loaded_model.eval()

    with torch.no_grad():
        loaded_output = loaded_model(example_input)

    # 비교
    if isinstance(original_output, tuple):
        original_output = original_output[0]
    if isinstance(loaded_output, tuple):
        loaded_output = loaded_output[0]

    is_close = torch.allclose(original_output, loaded_output, rtol=rtol, atol=atol)

    if is_close:
        print(f"[✓] Verification passed! Outputs match within tolerance.")
    else:
        max_diff = (original_output - loaded_output).abs().max().item()
        print(f"[✗] Verification failed! Max difference: {max_diff}")

    return is_close


def main():
    parser = argparse.ArgumentParser(description='Export ABMIL model to TorchScript')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--output_dir', type=str, default='exports/abmil_torchscript',
                        help='Output directory for exported model')
    parser.add_argument('--model_name', type=str, default='thyroid_abmil',
                        help='Name for the exported model')
    parser.add_argument('--with_attention', action='store_true',
                        help='Export model with attention output')
    parser.add_argument('--num_patches', type=int, default=2000,
                        help='Example number of patches for tracing')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for export (cpu recommended)')
    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 모델 설정 (학습 시 사용된 기본값)
    model_config = {
        'in_dim': 1536,
        'embed_dim': 512,
        'attn_dim': 384,
        'num_fc_layers': 1,
        'dropout': 0.0,  # 추론 시 dropout 비활성화
        'num_classes': 2
    }

    # 모델 생성
    if args.with_attention:
        model = ABMILWithAttention(**model_config)
        print("[INFO] Creating ABMILWithAttention (outputs: logits, attention)")
    else:
        model = ABMILForExport(**model_config)
        print("[INFO] Creating ABMILForExport (outputs: logits only)")

    # 가중치 로드
    model = load_checkpoint_weights(model, args.checkpoint)
    model.to(args.device)
    model.eval()

    # 예시 입력 생성
    example_input = torch.randn(1, args.num_patches, 1536).to(args.device)
    print(f"[INFO] Example input shape: {example_input.shape}")

    # TorchScript 변환
    output_path = os.path.join(args.output_dir, f"{args.model_name}.pt")
    export_torchscript(model, output_path, example_input, use_trace=True)

    # 검증
    print("\n[INFO] Verifying exported model...")
    verify_exported_model(model, output_path, example_input)

    # 모델 정보 저장
    info_path = os.path.join(args.output_dir, f"{args.model_name}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Source checkpoint: {args.checkpoint}\n")
        f.write(f"With attention output: {args.with_attention}\n")
        f.write(f"\nModel Config:\n")
        for k, v in model_config.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nInput shape: [batch_size, num_patches, 1536]\n")
        if args.with_attention:
            f.write(f"Output shapes: logits [batch_size, 2], attention [batch_size, num_patches]\n")
        else:
            f.write(f"Output shape: [batch_size, 2]\n")
        f.write(f"\nExported with PyTorch version: {torch.__version__}\n")

    print(f"\n[✓] Export complete!")
    print(f"    Model: {output_path}")
    print(f"    Info:  {info_path}")

    # 파일 크기
    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    Size:  {model_size:.2f} MB")


if __name__ == '__main__':
    main()