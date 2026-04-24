"""
TCGA-THCA External Validation Inference Script

TCGA 임베딩(.npy) → ABMIL 모델 추론 → 슬라이드별 예측 결과 + 전체 성능 지표 + MLflow 업로드

사용법:
    python src/inference/tcga_inference.py \
        --ckpt_dir /path/to/checkpoints \
        --embedding_dir /path/to/TCGA-THCA/embedding/h-optimus-0/20x/npy \
        --label_csv /path/to/TCGA-THCA/genomic/braf_slide_labels.csv \
        --out_dir /path/to/outputs/tcga_eval \
        --model_version v0.13.9
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from models.factory import create_mil_model as create_model

os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'


def load_labels(csv_path):
    """slide_name → label (0/1) 매핑. label=-1(other_BRAF) 제외"""
    label_map = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            label = int(row['label'])
            if label == -1:
                continue
            slide_name = Path(row['filename']).stem
            label_map[slide_name] = label
    return label_map


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model_name = ckpt.get('model_name', config.get('model_name', 'abmil'))
    result = create_model(model_name, in_dim=config.get('in_dim', 1536))
    model = result[0] if isinstance(result, tuple) else result
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, model_name


def filter_low_quality(npy_files, min_patches=1000):
    valid, filtered = [], []
    for f in npy_files:
        n = np.load(f, mmap_mode='r').shape[0]
        if n >= min_patches:
            valid.append(f)
        else:
            filtered.append((f.stem, n))
    if filtered:
        print(f"[FILTER] 패치 수 {min_patches} 미만 슬라이드 {len(filtered)}개 제외:")
        for name, n in filtered:
            print(f"  {name}: {n}개")
    print(f"[FILTER] 사용 슬라이드: {len(valid)}개 / 전체: {len(valid)+len(filtered)}개")
    return valid


def infer_slide(model, npy_path, device):
    feat = np.load(npy_path)
    feat = torch.from_numpy(feat).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(feat)
    # ABMIL: (results_dict, log_dict) 또는 (logits, ...) 또는 dict
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        logits = out['logits']
    else:
        logits = out
    prob = F.softmax(logits, dim=-1)[0]
    return prob[1].item()


def compute_metrics(labels, probs, threshold=0.5):
    preds = [1 if p >= threshold else 0 for p in probs]
    tp = sum(l == 1 and p == 1 for l, p in zip(labels, preds))
    tn = sum(l == 0 and p == 0 for l, p in zip(labels, preds))
    fp = sum(l == 0 and p == 1 for l, p in zip(labels, preds))
    fn = sum(l == 1 and p == 0 for l, p in zip(labels, preds))

    acc  = (tp + tn) / len(labels)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0.0

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = None

    return {"acc": acc, "auc": auc, "sensitivity": sens, "specificity": spec,
            "ppv": ppv, "npv": npv, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def plot_confusion_matrix(tp, tn, fp, fn, out_path, title="Confusion Matrix"):
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['BRAF-', 'BRAF+'],
           yticklabels=['BRAF-', 'BRAF+'],
           xlabel='Predicted', ylabel='Actual', title=title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curve(labels, probs, auc, out_path):
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, probs)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
               title='ROC Curve (TCGA External Validation)')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] ROC curve 생성 실패: {e}")


def plot_prob_distribution(labels, probs, out_path):
    braf_pos = [p for l, p in zip(labels, probs) if l == 1]
    braf_neg = [p for l, p in zip(labels, probs) if l == 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(braf_neg, bins=30, alpha=0.6, color='steelblue', label=f'BRAF- (n={len(braf_neg)})')
    ax.hist(braf_pos, bins=30, alpha=0.6, color='tomato',    label=f'BRAF+ (n={len(braf_pos)})')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='threshold=0.5')
    ax.set(xlabel='BRAF+ Probability', ylabel='Count', title='Prediction Score Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def upload_to_mlflow(fold_out, out_dir, model_version, fold_name="", embed_model="H-optimus-0"):
    try:
        import mlflow
    except ImportError:
        print("[WARN] mlflow 미설치 — MLflow 업로드 스킵")
        return

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("braf mutation")

    run_name = f"tcga_external_val_{model_version}_{fold_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    m = fold_out["metrics"]
    out_dir = Path(out_dir)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("Description",
            f"[TCGA External Validation] {embed_model} Embedding(1536-dim) → "
            f"{fold_out['model_name'].upper()} → BRAF V600E Classification. "
            f"Internal model: {model_version}, {fold_name}, n={fold_out['num_slides']} slides.")
        mlflow.set_tag("eval_type", "external_validation")
        mlflow.set_tag("dataset", "TCGA-THCA")
        mlflow.set_tag("embedding_model", embed_model)
        mlflow.set_tag("patch_config", "20x 256x256")
        mlflow.set_tag("internal_model_version", model_version)
        mlflow.set_tag("fold", fold_name)

        mlflow.log_params({
            "model":           fold_out["model_name"],
            "model_version":   model_version,
            "fold":            fold_name,
            "num_slides":      fold_out["num_slides"],
            "embedding_model": embed_model,
        })

        mlflow.log_metrics({
            "test_auc":         m["auc"]         or 0.0,
            "test_accuracy":    m["acc"],
            "test_f1":          m["f1"],
            "test_sensitivity": m["sensitivity"],
            "test_specificity": m["specificity"],
            "test_ppv":         m["ppv"],
            "test_npv":         m["npv"],
            "test_tp":          m["tp"],
            "test_tn":          m["tn"],
            "test_fp":          m["fp"],
            "test_fn":          m["fn"],
        })

        # fold별 artifacts
        fold_json = out_dir / f"{fold_name}_results.json"
        if fold_json.exists():
            mlflow.log_artifact(str(fold_json), artifact_path="results")
        for suffix in ["confusion_matrix", "roc_curve", "prob_distribution"]:
            png = out_dir / f"{fold_name}_{suffix}.png"
            if png.exists():
                mlflow.log_artifact(str(png), artifact_path="figures")

    print(f"[✓] MLflow 업로드 완료: {run_name}")


def run_inference(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    label_map = load_labels(args.label_csv)
    print(f"[INFO] 라벨 로드: {len(label_map)}개 슬라이드 (other_BRAF 제외)")

    npy_dir = Path(args.embedding_dir)
    npy_files = sorted(npy_dir.glob("*.npy"))
    print(f"[INFO] 임베딩 파일: {len(npy_files)}개")
    npy_files = filter_low_quality(npy_files, min_patches=args.min_patches)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_files = sorted(ckpt_dir.glob("*.pt"))
    print(f"[INFO] 체크포인트: {len(ckpt_files)}개 (fold별)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []

    for ckpt_path in ckpt_files:
        fold_name = ckpt_path.stem
        print(f"\n{'='*60}")
        print(f"[Fold] {fold_name}")
        print(f"{'='*60}")

        model, model_name = load_checkpoint(str(ckpt_path), device)

        slide_preds = []
        skipped = 0

        for npy_path in tqdm(npy_files, desc=f"Inferring {fold_name}"):
            slide_id = npy_path.stem
            if slide_id not in label_map:
                skipped += 1
                continue
            label = label_map[slide_id]
            prob = infer_slide(model, str(npy_path), device)
            slide_preds.append({"slide_id": slide_id, "label": label, "prob": prob})

        print(f"[INFO] 추론: {len(slide_preds)}개 / 스킵(라벨없음): {skipped}개")

        labels = [s["label"] for s in slide_preds]
        probs  = [s["prob"]  for s in slide_preds]
        metrics = compute_metrics(labels, probs, threshold=args.threshold)

        print(f"[결과] AUC={metrics['auc']:.4f}  Acc={metrics['acc']:.4f}  "
              f"F1={metrics['f1']:.4f}  Sens={metrics['sensitivity']:.4f}  "
              f"Spec={metrics['specificity']:.4f}")

        fold_out = {
            "checkpoint": ckpt_path.name,
            "model_name": model_name,
            "num_slides": len(slide_preds),
            "metrics": metrics,
            "slide_predictions": slide_preds,
        }
        fold_results.append(fold_out)

        # 시각화 (fold별)
        plot_confusion_matrix(
            metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"],
            out_dir / f"{fold_name}_confusion_matrix.png",
            title=f"Confusion Matrix ({fold_name})"
        )
        if metrics["auc"]:
            plot_roc_curve(labels, probs, metrics["auc"],
                           out_dir / f"{fold_name}_roc_curve.png")
        plot_prob_distribution(labels, probs,
                               out_dir / f"{fold_name}_prob_distribution.png")

        fold_json = out_dir / f"{fold_name}_results.json"
        with open(fold_json, "w") as f:
            json.dump(fold_out, f, indent=2)

        if not args.no_mlflow:
            upload_to_mlflow(fold_out, out_dir, args.model_version,
                             fold_name=fold_name, embed_model=args.embed_model)

    # 전체 요약 출력
    print(f"\n{'='*60}")
    print("[전체 결과 요약]")
    print(f"{'='*60}")
    print(f"{'Fold':<45} {'AUC':>7} {'Acc':>7} {'F1':>7} {'Sens':>7} {'Spec':>7}")
    print(f"{'-'*80}")
    for fold in fold_results:
        m = fold["metrics"]
        auc_s = f"{m['auc']:.4f}" if m['auc'] else "  ?   "
        print(f"{fold['checkpoint']:<45} {auc_s:>7} {m['acc']:.4f}  {m['f1']:.4f}  "
              f"{m['sensitivity']:.4f}  {m['specificity']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",      type=str, required=True)
    parser.add_argument("--embedding_dir", type=str, required=True)
    parser.add_argument("--label_csv",     type=str, required=True)
    parser.add_argument("--out_dir",       type=str, required=True)
    parser.add_argument("--model_version", type=str, default="v0.13.9")
    parser.add_argument("--embed_model",   type=str, default="H-optimus-0")
    parser.add_argument("--threshold",     type=float, default=0.5)
    parser.add_argument("--gpu",           type=int, default=0)
    parser.add_argument("--min_patches",   type=int, default=1000)
    parser.add_argument("--no_mlflow",     action="store_true")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
