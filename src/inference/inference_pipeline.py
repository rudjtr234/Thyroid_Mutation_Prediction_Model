"""
Triton Server 배포용 추론 파이프라인

전체 파이프라인:
1. 데이터 I/O: WSI 타일 이미지 또는 미리 추출된 임베딩 로드
2. 전처리: UNI2-h를 통한 feature extraction (옵션)
3. 추론: ABMIL 모델로 BRAF 변이 예측
4. 후처리: 확률 계산, attention 시각화

사용법:
    # 미리 추출된 임베딩으로 추론
    python inference_pipeline.py \
        --model_path exports/abmil_torchscript/thyroid_abmil.pt \
        --embedding_path /path/to/slide.npy \
        --output_dir results/

    # 타일 이미지에서 직접 추론 (UNI2-h 포함)
    python inference_pipeline.py \
        --model_path exports/abmil_torchscript/thyroid_abmil.pt \
        --tile_dir /path/to/tiles/ \
        --with_feature_extraction \
        --output_dir results/
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# UNI2-h feature extraction용
try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class DataIO:
    """데이터 I/O 클래스"""

    @staticmethod
    def load_embedding(npy_path: str) -> np.ndarray:
        """
        미리 추출된 임베딩 로드

        Args:
            npy_path: .npy 파일 경로

        Returns:
            embeddings: [num_patches, 1536] numpy array
        """
        embeddings = np.load(npy_path)
        print(f"[DataIO] Loaded embeddings: {embeddings.shape}")
        return embeddings

    @staticmethod
    def load_tile_images(tile_dir: str,
                         extensions: List[str] = ['.png', '.jpg', '.jpeg']) -> List[str]:
        """
        타일 이미지 경로 리스트 로드

        Args:
            tile_dir: 타일 이미지 디렉토리
            extensions: 이미지 확장자 리스트

        Returns:
            tile_paths: 이미지 경로 리스트
        """
        tile_dir = Path(tile_dir)
        tile_paths = []

        for ext in extensions:
            tile_paths.extend(sorted(tile_dir.glob(f"*{ext}")))

        tile_paths = [str(p) for p in tile_paths]
        print(f"[DataIO] Found {len(tile_paths)} tile images")
        return tile_paths

    @staticmethod
    def load_coordinates(json_path: str) -> Dict:
        """
        패치 좌표 정보 로드

        Args:
            json_path: JSON 파일 경로

        Returns:
            coord_data: 좌표 정보 딕셔너리
        """
        with open(json_path, 'r') as f:
            coord_data = json.load(f)
        return coord_data

    @staticmethod
    def save_results(results: Dict, output_path: str):
        """
        추론 결과 저장

        Args:
            results: 결과 딕셔너리
            output_path: 저장 경로
        """
        # numpy array를 리스트로 변환
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                serializable_results[k] = v.tolist()
            elif isinstance(v, torch.Tensor):
                serializable_results[k] = v.cpu().numpy().tolist()
            else:
                serializable_results[k] = v

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"[DataIO] Saved results to: {output_path}")


class Preprocessor:
    """전처리 클래스 (UNI2-h feature extraction 포함)"""

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None

    def load_uni2h_model(self):
        """UNI2-h 모델 로드"""
        if not HAS_TIMM:
            raise ImportError("timm is required for feature extraction. Install with: pip install timm")

        print("[Preprocessor] Loading UNI2-h model...")

        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }

        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            **timm_kwargs
        )
        self.model.to(self.device)
        self.model.eval()

        data_cfg = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**data_cfg)

        print(f"[Preprocessor] UNI2-h loaded on {self.device}")

    def extract_features(self,
                         tile_paths: List[str],
                         batch_size: int = 64) -> np.ndarray:
        """
        타일 이미지에서 UNI2-h 특징 추출

        Args:
            tile_paths: 타일 이미지 경로 리스트
            batch_size: 배치 크기

        Returns:
            features: [num_tiles, 1536] numpy array
        """
        if self.model is None:
            self.load_uni2h_model()

        features = []

        for i in tqdm(range(0, len(tile_paths), batch_size),
                      desc="Extracting features"):
            batch_paths = tile_paths[i:i + batch_size]

            # 이미지 로드 및 변환
            imgs = []
            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                imgs.append(self.transform(img))

            batch_tensor = torch.stack(imgs).to(self.device)

            # Feature extraction
            with torch.no_grad():
                batch_features = self.model(batch_tensor)

            features.append(batch_features.cpu().numpy())

        features = np.vstack(features)
        print(f"[Preprocessor] Extracted features: {features.shape}")
        return features

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray,
                             method: str = 'none') -> np.ndarray:
        """
        임베딩 정규화 (옵션)

        Args:
            embeddings: [num_patches, dim] 임베딩
            method: 정규화 방법 ('none', 'l2', 'zscore')

        Returns:
            normalized: 정규화된 임베딩
        """
        if method == 'none':
            return embeddings
        elif method == 'l2':
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / (norms + 1e-8)
        elif method == 'zscore':
            mean = embeddings.mean(axis=0, keepdims=True)
            std = embeddings.std(axis=0, keepdims=True)
            return (embeddings - mean) / (std + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class ABMILInference:
    """ABMIL 추론 클래스"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: TorchScript 모델 경로
            device: 추론 디바이스
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> torch.jit.ScriptModule:
        """TorchScript 모델 로드"""
        print(f"[ABMILInference] Loading model from: {model_path}")
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        print(f"[ABMILInference] Model loaded on {self.device}")
        return model

    def predict(self,
                embeddings: Union[np.ndarray, torch.Tensor],
                return_attention: bool = False) -> Dict:
        """
        BRAF 변이 예측

        Args:
            embeddings: [num_patches, 1536] 또는 [1, num_patches, 1536]
            return_attention: attention 가중치 반환 여부

        Returns:
            results: 예측 결과 딕셔너리
        """
        # numpy to tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()

        # [N, D] -> [1, N, D]
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        embeddings = embeddings.to(self.device)

        # 추론
        with torch.no_grad():
            output = self.model(embeddings)

        # 출력 처리
        if isinstance(output, tuple):
            logits, attention = output
        else:
            logits = output
            attention = None

        # Softmax 확률
        probs = F.softmax(logits, dim=-1)

        # 결과 정리
        results = {
            'logits': logits.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'predicted_class': probs.argmax(dim=-1).cpu().numpy(),
            'braf_positive_prob': probs[0, 1].cpu().item(),
            'braf_negative_prob': probs[0, 0].cpu().item(),
        }

        if attention is not None and return_attention:
            results['attention'] = attention.cpu().numpy()

        return results


class Postprocessor:
    """후처리 클래스"""

    @staticmethod
    def interpret_prediction(results: Dict,
                             threshold: float = 0.5) -> Dict:
        """
        예측 결과 해석

        Args:
            results: 추론 결과
            threshold: 분류 임계값

        Returns:
            interpretation: 해석 결과
        """
        braf_prob = results['braf_positive_prob']
        predicted = 'BRAF_POSITIVE' if braf_prob >= threshold else 'BRAF_NEGATIVE'

        interpretation = {
            'prediction': predicted,
            'confidence': max(braf_prob, 1 - braf_prob),
            'braf_positive_probability': braf_prob,
            'threshold_used': threshold,
        }

        return interpretation

    @staticmethod
    def get_top_attention_patches(attention: np.ndarray,
                                  coordinates: Optional[List[Dict]] = None,
                                  top_k: int = 10) -> List[Dict]:
        """
        높은 attention을 가진 상위 패치 반환

        Args:
            attention: [num_patches] attention 가중치
            coordinates: 패치 좌표 정보 (옵션)
            top_k: 반환할 상위 패치 수

        Returns:
            top_patches: 상위 패치 정보 리스트
        """
        if attention.ndim > 1:
            attention = attention.squeeze()

        # 상위 k개 인덱스
        top_indices = np.argsort(attention)[-top_k:][::-1]

        top_patches = []
        for idx in top_indices:
            patch_info = {
                'patch_index': int(idx),
                'attention_weight': float(attention[idx]),
            }
            if coordinates is not None and idx < len(coordinates):
                patch_info.update(coordinates[idx])
            top_patches.append(patch_info)

        return top_patches

    @staticmethod
    def create_attention_heatmap(attention: np.ndarray,
                                 coordinates: List[Dict],
                                 output_path: str,
                                 slide_shape: Tuple[int, int] = None):
        """
        Attention heatmap 생성 (시각화용)

        Args:
            attention: [num_patches] attention 가중치
            coordinates: 패치 좌표 정보
            output_path: 저장 경로
            slide_shape: 슬라이드 크기 (width, height)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            print("[Postprocessor] matplotlib required for heatmap visualization")
            return

        if attention.ndim > 1:
            attention = attention.squeeze()

        # 좌표 추출
        xs = [c.get('x', c.get('X', 0)) for c in coordinates]
        ys = [c.get('y', c.get('Y', 0)) for c in coordinates]

        # 정규화
        attention_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(xs, ys, c=attention_norm, cmap='hot', s=10, alpha=0.7)
        plt.colorbar(scatter, label='Attention Weight')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('Attention Heatmap')
        plt.gca().invert_yaxis()  # WSI 좌표계
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Postprocessor] Saved attention heatmap to: {output_path}")


class InferencePipeline:
    """전체 추론 파이프라인"""

    def __init__(self,
                 model_path: str,
                 device: str = 'cuda',
                 with_feature_extraction: bool = False):
        """
        Args:
            model_path: ABMIL TorchScript 모델 경로
            device: 추론 디바이스
            with_feature_extraction: UNI2-h feature extraction 포함 여부
        """
        self.device = device
        self.with_feature_extraction = with_feature_extraction

        # 컴포넌트 초기화
        self.data_io = DataIO()
        self.preprocessor = Preprocessor(device) if with_feature_extraction else None
        self.model = ABMILInference(model_path, device)
        self.postprocessor = Postprocessor()

    def run(self,
            input_path: str,
            output_dir: str,
            threshold: float = 0.5,
            save_attention: bool = True,
            coord_json_path: Optional[str] = None) -> Dict:
        """
        전체 파이프라인 실행

        Args:
            input_path: 입력 경로 (.npy 또는 타일 디렉토리)
            output_dir: 출력 디렉토리
            threshold: 분류 임계값
            save_attention: attention 저장 여부
            coord_json_path: 좌표 JSON 경로 (옵션)

        Returns:
            final_results: 최종 결과
        """
        os.makedirs(output_dir, exist_ok=True)

        # 슬라이드 ID 추출
        input_path = Path(input_path)
        slide_id = input_path.stem if input_path.is_file() else input_path.name

        print(f"\n{'='*60}")
        print(f"[Pipeline] Processing: {slide_id}")
        print(f"{'='*60}")

        # 1. 데이터 로드
        if input_path.suffix == '.npy':
            embeddings = self.data_io.load_embedding(str(input_path))
        elif input_path.is_dir():
            if not self.with_feature_extraction:
                raise ValueError("Feature extraction required for tile directory input")
            tile_paths = self.data_io.load_tile_images(str(input_path))
            embeddings = self.preprocessor.extract_features(tile_paths)
        else:
            raise ValueError(f"Unsupported input: {input_path}")

        # 좌표 로드 (있는 경우)
        coordinates = None
        if coord_json_path and os.path.exists(coord_json_path):
            coord_data = self.data_io.load_coordinates(coord_json_path)
            coordinates = coord_data.get('patch_coords', [])

        # 2. 추론
        print("[Pipeline] Running inference...")
        results = self.model.predict(embeddings, return_attention=save_attention)

        # 3. 후처리
        interpretation = self.postprocessor.interpret_prediction(results, threshold)

        # 최종 결과 조합
        final_results = {
            'slide_id': slide_id,
            'num_patches': embeddings.shape[0],
            **interpretation,
            'raw_probabilities': {
                'BRAF_NEGATIVE': results['braf_negative_prob'],
                'BRAF_POSITIVE': results['braf_positive_prob'],
            }
        }

        # Attention 분석
        if 'attention' in results and coordinates:
            top_patches = self.postprocessor.get_top_attention_patches(
                results['attention'], coordinates, top_k=20
            )
            final_results['top_attention_patches'] = top_patches

            # Heatmap 저장
            heatmap_path = os.path.join(output_dir, f"{slide_id}_attention_heatmap.png")
            self.postprocessor.create_attention_heatmap(
                results['attention'], coordinates, heatmap_path
            )

        # 결과 저장
        result_path = os.path.join(output_dir, f"{slide_id}_prediction.json")
        self.data_io.save_results(final_results, result_path)

        # 결과 출력
        print(f"\n[Pipeline] Results for {slide_id}:")
        print(f"  Prediction: {interpretation['prediction']}")
        print(f"  Confidence: {interpretation['confidence']:.4f}")
        print(f"  BRAF+ Probability: {interpretation['braf_positive_probability']:.4f}")

        return final_results


def main():
    parser = argparse.ArgumentParser(description='BRAF Mutation Inference Pipeline')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to TorchScript model (.pt)')
    parser.add_argument('--embedding_path', type=str, default=None,
                        help='Path to pre-extracted embeddings (.npy)')
    parser.add_argument('--tile_dir', type=str, default=None,
                        help='Path to tile image directory')
    parser.add_argument('--coord_json', type=str, default=None,
                        help='Path to coordinate JSON file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--with_feature_extraction', action='store_true',
                        help='Enable UNI2-h feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process multiple slides in batch')
    parser.add_argument('--input_list', type=str, default=None,
                        help='Text file with input paths (for batch mode)')
    args = parser.parse_args()

    # 입력 검증
    if args.embedding_path is None and args.tile_dir is None and args.input_list is None:
        parser.error("Either --embedding_path, --tile_dir, or --input_list is required")

    # 파이프라인 초기화
    pipeline = InferencePipeline(
        model_path=args.model_path,
        device=args.device,
        with_feature_extraction=args.with_feature_extraction
    )

    # 배치 모드
    if args.batch_mode and args.input_list:
        with open(args.input_list, 'r') as f:
            input_paths = [line.strip() for line in f if line.strip()]

        all_results = []
        for input_path in input_paths:
            try:
                result = pipeline.run(
                    input_path=input_path,
                    output_dir=args.output_dir,
                    threshold=args.threshold,
                    coord_json_path=args.coord_json
                )
                all_results.append(result)
            except Exception as e:
                print(f"[ERROR] Failed to process {input_path}: {e}")

        # 배치 요약 저장
        summary_path = os.path.join(args.output_dir, 'batch_summary.json')
        DataIO.save_results({'results': all_results}, summary_path)

    # 단일 슬라이드 모드
    else:
        input_path = args.embedding_path or args.tile_dir
        pipeline.run(
            input_path=input_path,
            output_dir=args.output_dir,
            threshold=args.threshold,
            coord_json_path=args.coord_json
        )


if __name__ == '__main__':
    main()
