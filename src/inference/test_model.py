"""
TorchScript 모델 테스트 스크립트

사용법:
    python src/inference/test_model.py
"""

import torch
import numpy as np
import glob

def main():
    # 1. 모델 로드
    print("=" * 60)
    print("TorchScript ABMIL 모델 테스트")
    print("=" * 60)

    model_path = "/data/member/jks/Thyroid_Mutation_model_v2/exports/abmil_torchscript/thyroid_abmil.pt"
    print(f"\n[1] 모델 로드: {model_path}")

    model = torch.jit.load(model_path)
    model.eval()
    print("    ✓ 모델 로드 성공!")

    # 2. 테스트 임베딩 로드
    print("\n[2] 테스트 임베딩 로드")

    meta_dir = "/data/member/jks/dataset/Thyroid_Mutation_dataset/uni2_embeddings/final_meta_dataset_v0.1.1/npy"
    nonmeta_dir = "/data/member/jks/dataset/Thyroid_Mutation_dataset/uni2_embeddings/final_nonmeta_dataset_v0.1.0/npy"

    meta_files = sorted(glob.glob(f"{meta_dir}/*.npy"))[:3]
    nonmeta_files = sorted(glob.glob(f"{nonmeta_dir}/*.npy"))[:3]

    print(f"    BRAF+ (meta): {len(meta_files)}개")
    print(f"    BRAF- (nonmeta): {len(nonmeta_files)}개")

    # 3. 추론 테스트
    print("\n[3] 추론 결과")
    print("-" * 60)
    print(f"{'슬라이드ID':<20} {'패치수':>8} {'실제':>8} {'예측':>8} {'BRAF+ 확률':>12} {'결과':>6}")
    print("-" * 60)

    correct = 0
    total = 0

    all_files = [(f, "BRAF+") for f in meta_files] + [(f, "BRAF-") for f in nonmeta_files]

    for npy_path, true_label in all_files:
        # 임베딩 로드
        embeddings = np.load(npy_path)
        slide_id = npy_path.split("/")[-1].replace(".npy", "")

        # 추론
        x = torch.from_numpy(embeddings).float().unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            braf_prob = probs[0, 1].item()
            pred_label = "BRAF+" if braf_prob > 0.5 else "BRAF-"

        # 결과
        is_correct = true_label == pred_label
        correct += int(is_correct)
        total += 1

        result = "✓" if is_correct else "✗"
        print(f"{slide_id:<20} {embeddings.shape[0]:>8} {true_label:>8} {pred_label:>8} {braf_prob:>12.4f} {result:>6}")

    print("-" * 60)
    print(f"\n정확도: {correct}/{total} ({correct/total*100:.1f}%)")
    print("\n✓ 테스트 완료!")


if __name__ == "__main__":
    main()