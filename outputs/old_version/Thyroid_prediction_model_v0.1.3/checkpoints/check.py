import torch

# 1. checkpoint 로드
checkpoint = torch.load("best_model_fold2_auc0.9700.pt", map_location="cpu")

print("🔹 Keys in checkpoint:")
for k in checkpoint.keys():
    print(f"  {k}")

# 2. 실제 모델 가중치(state_dict) 추출
state_dict = checkpoint["model_state_dict"]

print("\n🔹 Keys in model_state_dict:")
for k in state_dict.keys():
    print(f"  {k}")

# 3. 전체 파라미터 개수 출력
total_params = sum(p.numel() for p in state_dict.values())
print(f"\nTotal parameters: {total_params:,}")

