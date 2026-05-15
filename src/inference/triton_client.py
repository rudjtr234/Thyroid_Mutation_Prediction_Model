"""
Triton Inference Server 클라이언트

Triton Server에 배포된 ABMIL 모델을 호출하는 클라이언트 코드

사용법:
    python triton_client.py \
        --server_url localhost:8001 \
        --embedding_path /path/to/slide.npy \
        --output_dir results/

의존성:
    pip install tritonclient[grpc] tritonclient[http]
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

try:
    import tritonclient.grpc as grpcclient
    import tritonclient.http as httpclient
    HAS_TRITON_CLIENT = True
except ImportError:
    HAS_TRITON_CLIENT = False
    print("[WARNING] tritonclient not installed. Install with: pip install tritonclient[grpc,http]")


class TritonABMILClient:
    """Triton Server ABMIL 클라이언트"""

    def __init__(self,
                 server_url: str = "localhost:8001",
                 model_name: str = "thyroid_abmil",
                 model_version: str = "",
                 protocol: str = "grpc",
                 verbose: bool = False):
        """
        Args:
            server_url: Triton 서버 주소 (gRPC: 8001, HTTP: 8000)
            model_name: 모델 이름
            model_version: 모델 버전 (빈 문자열이면 최신)
            protocol: 통신 프로토콜 ('grpc' or 'http')
            verbose: 상세 로그 출력
        """
        if not HAS_TRITON_CLIENT:
            raise ImportError("tritonclient is required. Install with: pip install tritonclient[grpc,http]")

        self.server_url = server_url
        self.model_name = model_name
        self.model_version = model_version
        self.protocol = protocol.lower()
        self.verbose = verbose

        self._init_client()

    def _init_client(self):
        """클라이언트 초기화"""
        if self.protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(
                url=self.server_url,
                verbose=self.verbose
            )
            self.InferInput = grpcclient.InferInput
            self.InferRequestedOutput = grpcclient.InferRequestedOutput
        else:
            self.client = httpclient.InferenceServerClient(
                url=self.server_url,
                verbose=self.verbose
            )
            self.InferInput = httpclient.InferInput
            self.InferRequestedOutput = httpclient.InferRequestedOutput

        # 서버 연결 확인
        if not self.client.is_server_live():
            raise ConnectionError(f"Triton server not live at {self.server_url}")

        # 모델 준비 확인
        if not self.client.is_model_ready(self.model_name):
            raise RuntimeError(f"Model '{self.model_name}' is not ready")

        print(f"[TritonClient] Connected to {self.server_url} ({self.protocol})")
        print(f"[TritonClient] Model '{self.model_name}' is ready")

    def get_model_metadata(self) -> Dict:
        """모델 메타데이터 조회"""
        metadata = self.client.get_model_metadata(self.model_name)
        if self.protocol == "grpc":
            return {
                'name': metadata.name,
                'versions': list(metadata.versions),
                'inputs': [
                    {'name': inp.name, 'datatype': inp.datatype, 'shape': list(inp.shape)}
                    for inp in metadata.inputs
                ],
                'outputs': [
                    {'name': out.name, 'datatype': out.datatype, 'shape': list(out.shape)}
                    for out in metadata.outputs
                ]
            }
        else:
            return metadata

    def predict(self,
                embeddings: np.ndarray,
                return_probabilities: bool = True) -> Dict:
        """
        BRAF 변이 예측

        Args:
            embeddings: [num_patches, 1536] 또는 [batch, num_patches, 1536]
            return_probabilities: softmax 확률 반환 여부

        Returns:
            results: 예측 결과 딕셔너리
        """
        # 입력 형태 확인
        if embeddings.ndim == 2:
            embeddings = embeddings[np.newaxis, ...]  # [1, N, 1536]

        batch_size = embeddings.shape[0]
        num_patches = embeddings.shape[1]

        # 입력 텐서 준비
        inputs = []
        input0 = self.InferInput("INPUT__0", embeddings.shape, "FP32")
        input0.set_data_from_numpy(embeddings.astype(np.float32))
        inputs.append(input0)

        # 출력 요청
        outputs = []
        output0 = self.InferRequestedOutput("OUTPUT__0")
        outputs.append(output0)

        # 추론 요청
        if self.protocol == "grpc":
            response = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs
            )
        else:
            response = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs
            )

        # 결과 추출
        logits = response.as_numpy("OUTPUT__0")

        # 결과 처리
        results = {
            'logits': logits,
            'batch_size': batch_size,
            'num_patches': num_patches,
        }

        if return_probabilities:
            # Softmax 계산
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            results['probabilities'] = probs
            results['braf_negative_prob'] = probs[:, 0].tolist()
            results['braf_positive_prob'] = probs[:, 1].tolist()
            results['predicted_class'] = np.argmax(probs, axis=-1).tolist()

        return results

    def predict_slide(self,
                      embedding_path: str,
                      threshold: float = 0.5) -> Dict:
        """
        단일 슬라이드 예측 (편의 함수)

        Args:
            embedding_path: .npy 임베딩 파일 경로
            threshold: 분류 임계값

        Returns:
            result: 예측 결과
        """
        # 임베딩 로드
        embeddings = np.load(embedding_path)
        slide_id = Path(embedding_path).stem

        print(f"[TritonClient] Predicting slide: {slide_id}")
        print(f"[TritonClient] Embeddings shape: {embeddings.shape}")

        # 예측
        results = self.predict(embeddings)

        # 해석
        braf_prob = results['braf_positive_prob'][0]
        prediction = 'BRAF_POSITIVE' if braf_prob >= threshold else 'BRAF_NEGATIVE'

        final_result = {
            'slide_id': slide_id,
            'prediction': prediction,
            'confidence': max(braf_prob, 1 - braf_prob),
            'braf_positive_probability': braf_prob,
            'braf_negative_probability': results['braf_negative_prob'][0],
            'threshold_used': threshold,
            'num_patches': results['num_patches'],
        }

        print(f"[TritonClient] Prediction: {prediction} (confidence: {final_result['confidence']:.4f})")

        return final_result

    def health_check(self) -> Dict:
        """서버 상태 확인"""
        return {
            'server_live': self.client.is_server_live(),
            'server_ready': self.client.is_server_ready(),
            'model_ready': self.client.is_model_ready(self.model_name),
        }


class TritonBatchClient:
    """배치 추론용 클라이언트"""

    def __init__(self, client: TritonABMILClient):
        self.client = client

    def predict_batch(self,
                      embedding_paths: List[str],
                      threshold: float = 0.5,
                      output_dir: Optional[str] = None) -> List[Dict]:
        """
        여러 슬라이드 배치 예측

        Args:
            embedding_paths: .npy 파일 경로 리스트
            threshold: 분류 임계값
            output_dir: 결과 저장 디렉토리

        Returns:
            results: 예측 결과 리스트
        """
        results = []

        for path in embedding_paths:
            try:
                result = self.client.predict_slide(path, threshold)
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Failed to predict {path}: {e}")
                results.append({
                    'slide_id': Path(path).stem,
                    'error': str(e)
                })

        # 결과 저장
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # 개별 결과 저장
            for result in results:
                if 'error' not in result:
                    result_path = os.path.join(output_dir, f"{result['slide_id']}_prediction.json")
                    with open(result_path, 'w') as f:
                        json.dump(result, f, indent=2)

            # 요약 저장
            summary = {
                'total': len(results),
                'successful': sum(1 for r in results if 'error' not in r),
                'failed': sum(1 for r in results if 'error' in r),
                'braf_positive': sum(1 for r in results if r.get('prediction') == 'BRAF_POSITIVE'),
                'braf_negative': sum(1 for r in results if r.get('prediction') == 'BRAF_NEGATIVE'),
            }

            summary_path = os.path.join(output_dir, 'batch_summary.json')
            with open(summary_path, 'w') as f:
                json.dump({'summary': summary, 'results': results}, f, indent=2)
            print(f"[BatchClient] Saved summary to: {summary_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Triton ABMIL Client')
    parser.add_argument('--server_url', type=str, default='localhost:8001',
                        help='Triton server URL')
    parser.add_argument('--model_name', type=str, default='thyroid_abmil',
                        help='Model name')
    parser.add_argument('--protocol', type=str, default='grpc',
                        choices=['grpc', 'http'], help='Communication protocol')
    parser.add_argument('--embedding_path', type=str, default=None,
                        help='Path to embedding .npy file')
    parser.add_argument('--embedding_dir', type=str, default=None,
                        help='Directory containing embedding .npy files (batch mode)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--health_check', action='store_true',
                        help='Run health check only')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    # 클라이언트 초기화
    client = TritonABMILClient(
        server_url=args.server_url,
        model_name=args.model_name,
        protocol=args.protocol,
        verbose=args.verbose
    )

    # Health check
    if args.health_check:
        health = client.health_check()
        print(json.dumps(health, indent=2))
        return

    # 모델 메타데이터 출력
    print("\n[Model Metadata]")
    metadata = client.get_model_metadata()
    print(json.dumps(metadata, indent=2))

    # 추론 실행
    if args.embedding_path:
        result = client.predict_slide(args.embedding_path, args.threshold)

        # 결과 저장
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, f"{result['slide_id']}_prediction.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n[Saved] {result_path}")

    elif args.embedding_dir:
        # 배치 모드
        embedding_dir = Path(args.embedding_dir)
        embedding_paths = sorted(str(p) for p in embedding_dir.glob("*.npy"))
        print(f"\n[BatchClient] Found {len(embedding_paths)} embedding files")

        batch_client = TritonBatchClient(client)
        results = batch_client.predict_batch(
            embedding_paths,
            threshold=args.threshold,
            output_dir=args.output_dir
        )

        # 요약 출력
        print(f"\n[Summary]")
        print(f"  Total: {len(results)}")
        print(f"  BRAF+: {sum(1 for r in results if r.get('prediction') == 'BRAF_POSITIVE')}")
        print(f"  BRAF-: {sum(1 for r in results if r.get('prediction') == 'BRAF_NEGATIVE')}")

    else:
        print("[ERROR] Either --embedding_path or --embedding_dir is required")


if __name__ == '__main__':
    main()
