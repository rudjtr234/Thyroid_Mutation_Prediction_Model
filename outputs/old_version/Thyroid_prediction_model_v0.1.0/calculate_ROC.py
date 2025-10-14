import numpy as np
import sys

# JSON 라이브러리 선택 (json5 → fallback json)
try:
    import json5 as json
except ImportError:
    import json
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score

# ==============================
# 1. results.json 불러오기
# ==============================
json_path = "results.json"
try:
    with open(json_path) as f:
        results = json.load(f)
except Exception as e:
    print(f"[ERROR] JSON 파싱 실패: {e}")
    sys.exit(1)

# ==============================
# 2. 전체 결과 모으기
# ==============================
all_true, all_probs = [], []

if "folds" not in results:
    print("[ERROR] results.json 안에 'folds' 키가 없습니다.")
    sys.exit(1)

for fold in results["folds"]:
    if "test_results" not in fold:
        continue
    for r in fold["test_results"]:
        try:
            all_true.append(r["true_label"])
            all_probs.append(r["weighted_prob"])  # or simple_avg_prob
        except KeyError:
            continue

all_true = np.array(all_true)
all_probs = np.array(all_probs)

if len(all_true) == 0:
    print("[ERROR] test_results에서 데이터를 읽을 수 없습니다.")
    sys.exit(1)

# ==============================
# 3. ROC / Threshold 최적화 (전체)
# ==============================
fpr, tpr, thresholds = roc_curve(all_true, all_probs)
roc_auc = auc(fpr, tpr)

# F1 최적 threshold
f1s = [f1_score(all_true, (all_probs >= t).astype(int)) for t in thresholds]
best_idx = np.argmax(f1s)
best_thr = thresholds[best_idx]

# Precision/Recall at best_thr
y_pred = (all_probs >= best_thr).astype(int)
prec = precision_score(all_true, y_pred)
rec = recall_score(all_true, y_pred)

print("====== [전체 ROC / Threshold 결과] ======")
print(f"ROC AUC     : {roc_auc:.4f}")
print(f"Best Thresh : {best_thr:.3f}")
print(f"F1 Score    : {f1s[best_idx]:.4f}")
print(f"Precision   : {prec:.4f}")
print(f"Recall      : {rec:.4f}")
print("========================================\n")

# ==============================
# 4. Fold별 결과
# ==============================
print("====== [Fold별 ROC AUC / Best F1] ======")
for fold in results["folds"]:
    fold_true, fold_probs = [], []
    for r in fold.get("test_results", []):
        fold_true.append(r["true_label"])
        fold_probs.append(r["weighted_prob"])
    if len(fold_true) == 0:
        continue
    fold_true = np.array(fold_true)
    fold_probs = np.array(fold_probs)

    fpr, tpr, thresholds = roc_curve(fold_true, fold_probs)
    fold_auc = auc(fpr, tpr)

    f1s = [f1_score(fold_true, (fold_probs >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1s)
    best_thr = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    print(f"Fold {fold['fold']:>2} | AUC={fold_auc:.4f} | BestThr={best_thr:.3f} | F1={best_f1:.4f}")
print("========================================")
