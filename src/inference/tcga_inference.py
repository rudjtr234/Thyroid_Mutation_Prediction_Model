"""
TCGA-THCA External Validation Inference Script

TCGA 임베딩(.npy) → ABMIL 모델 추론 → 슬라이드별 예측 결과 + 전체 성능 지표 + MLflow 업로드

사용법:
    python src/inference/tcga_inference.py \
        --ckpt_dir outputs/Thyroid_prediction_model_v0.13.9/checkpoints \
        --embedding_dir /path/to/TCGA-THCA/embedding/h-optimus-0/20x/npy \
        --label_csv /path/to/TCGA-THCA/genomic/braf_slide_labels.csv \
        --out_dir outputs/tcga_eval_v0.13.9 \
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


def infer_slide(model, npy_path, device, return_attention=False):
    feat = np.load(npy_path)
    feat = torch.from_numpy(feat).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(feat, return_attention=return_attention)

    if isinstance(out, tuple):
        results_dict, log_dict = out[0], out[1]
    else:
        results_dict, log_dict = out, {}

    if isinstance(results_dict, dict):
        logits = results_dict['logits']
    else:
        logits = results_dict

    prob = F.softmax(logits, dim=-1)[0]

    attn = None
    if return_attention and isinstance(log_dict, dict):
        a = log_dict.get('A')
        if a is None:
            a = log_dict.get('attention')
        if a is not None:
            attn = a.cpu().numpy().flatten()

    return prob[1].item(), attn


def load_coords(json_dir, slide_id):
    """슬라이드 좌표 JSON 로드"""
    json_path = Path(json_dir) / f"{slide_id}.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    return data.get("patch_coords", [])


def plot_attention_heatmap(attn_scores, coords, slide_id, patch_size, out_path, prob=None,
                           svs_base_dir=None, thumbnail_max_side=2048, label=None):
    """Attention heatmap — SVS overlay (가능 시) 또는 grid heatmap fallback.

    label=1 (BRAF+): 파랑→초록→노랑→빨강 (attention 높을수록 빨강)
    label=0 (BRAF-): 흰색→연파랑→짙은파랑 (attention 높을수록 짙은 파랑)
    """
    if not coords or attn_scores is None:
        return

    try:
        import cv2
        from matplotlib.colors import LinearSegmentedColormap
        if label == 0:
            # BRAF- : 흰색 → 하늘 → 짙은 파랑
            cmap = LinearSegmentedColormap.from_list('attention_neg',
                ['#FFFFFF', '#AED6F1', '#2980B9', '#1A5276', '#0B2D5E'], N=256)
        else:
            # BRAF+ (기본): 파랑 → 청록 → 초록 → 노랑 → 빨강
            cmap = LinearSegmentedColormap.from_list('attention_pos',
                ['#2E3192', '#1BFFFF', '#00FF00', '#FFFF00', '#FF0000'], N=256)

        # 좌표 → grid 변환
        x_list = sorted(set(c['x'] for c in coords))
        y_list = sorted(set(c['y'] for c in coords))
        x_to_col = {x: i for i, x in enumerate(x_list)}
        y_to_row = {y: i for i, y in enumerate(y_list)}
        n_rows, n_cols = len(y_list), len(x_list)

        attn = np.array(attn_scores[:len(coords)], dtype=np.float32)
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        for i, c in enumerate(coords[:len(attn_norm)]):
            r = y_to_row.get(c['y'], 0)
            col = x_to_col.get(c['x'], 0)
            grid[r, col] = attn_norm[i]

        # SVS overlay 시도 (TCGA는 UUID 하위 폴더 구조)
        svs_path = None
        if svs_base_dir:
            base = Path(svs_base_dir)
            # 직접 경로
            for sub in ['', 'meta_braf', 'non_braf']:
                p = base / sub / f"{slide_id}.svs"
                if p.exists():
                    svs_path = p
                    break
            # UUID 하위 폴더 탐색 (TCGA 구조)
            if svs_path is None:
                matches = list(base.rglob(f"{slide_id}.svs"))
                if matches:
                    svs_path = matches[0]

        if svs_path:
            try:
                import openslide
                slide = openslide.OpenSlide(str(svs_path))
                slide_w, slide_h = slide.dimensions
                scale = min(thumbnail_max_side / max(slide_w, slide_h), 1.0)
                thumb_w = max(1, int(slide_w * scale))
                thumb_h = max(1, int(slide_h * scale))
                thumb = np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")).astype(np.float32) / 255.0
                slide.close()

                patch_w = int(x_list[1] - x_list[0]) if len(x_list) > 1 else patch_size
                patch_h = int(y_list[1] - y_list[0]) if len(y_list) > 1 else patch_size
                x0 = max(0, int(x_list[0] * scale))
                y0 = max(0, int(y_list[0] * scale))
                x1 = min(thumb_w, int((x_list[-1] + patch_w) * scale))
                y1 = min(thumb_h, int((y_list[-1] + patch_h) * scale))

                grid_resized = cv2.resize(grid, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                score = np.clip(grid_resized, 0, 1)
                score_enh = np.power(score, 0.6)
                heat_rgb = cmap(score_enh)[..., :3]
                alpha = np.where(score > 0, (0.15 + 0.85 * score_enh) * 0.85, 0.0)[..., None]

                region = thumb[y0:y1, x0:x1]
                thumb[y0:y1, x0:x1] = region * (1 - alpha) + heat_rgb * alpha

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(np.clip(thumb * 255, 0, 255).astype(np.uint8))
                title = f"{slide_id} — Attention Overlay"
                if prob is not None:
                    title += f"  (BRAF+ prob={prob:.4f})"
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close()
                return
            except Exception:
                pass  # overlay 실패 시 fallback

        # Fallback: grid heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Attention Weight')
        title = f"{slide_id} — Attention Heatmap"
        if prob is not None:
            title += f"  (BRAF+ prob={prob:.4f})"
        ax.set_title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    except Exception as e:
        print(f"  [WARN] Heatmap 생성 실패 ({slide_id}): {e}")


def bootstrap_ci_metric(labels, probs, metric_fn, n_bootstrap=2000, alpha=0.05, seed=42):
    """단일 metric에 대한 bootstrap 95% CI"""
    rng = np.random.RandomState(seed)
    labels, probs = np.array(labels), np.array(probs)
    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(labels), size=len(labels), replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        boot_vals.append(metric_fn(labels[idx], probs[idx]))
    if not boot_vals:
        return None, None
    return float(np.percentile(boot_vals, 100 * alpha / 2)), float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))


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
        auc_ci_lo, auc_ci_hi = bootstrap_ci_metric(
            labels, probs, lambda y, p: roc_auc_score(y, p))
    except Exception:
        auc = None
        auc_ci_lo, auc_ci_hi = None, None

    # acc CI (bootstrap on binary predictions)
    def _acc(y, p): return np.mean((p >= threshold) == y)
    acc_ci_lo, acc_ci_hi = bootstrap_ci_metric(labels, probs, _acc)

    return {"acc": acc, "auc": auc, "sensitivity": sens, "specificity": spec,
            "ppv": ppv, "npv": npv, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "auc_ci": [auc_ci_lo, auc_ci_hi],
            "acc_ci": [acc_ci_lo, acc_ci_hi]}


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


def build_summary_html(fold_results, model_version, embed_model, patch_config):
    """fold별 결과를 HTML 표로 생성"""
    rows = ""
    for fold in fold_results:
        m = fold["metrics"]
        auc_ci = m.get("auc_ci", [None, None])
        acc_ci = m.get("acc_ci", [None, None])
        auc_s = f"{m['auc']:.4f}" if m["auc"] else "-"
        rows += f"""
        <tr>
          <td>{fold['checkpoint']}</td>
          <td>{auc_s}</td>
          <td>{m['acc']:.4f}</td>
          <td>{m['f1']:.4f}</td>
          <td>{m['sensitivity']:.4f}</td>
          <td>{m['specificity']:.4f}</td>
          <td>{m['ppv']:.4f}</td>
          <td>{m['npv']:.4f}</td>
          <td>{m['num_slides'] if 'num_slides' in m else fold['num_slides']}</td>
        </tr>"""

    # 평균 행 계산
    aucs  = [f["metrics"]["auc"] for f in fold_results if f["metrics"]["auc"]]
    accs  = [f["metrics"]["acc"] for f in fold_results]
    f1s   = [f["metrics"]["f1"]  for f in fold_results]
    senss = [f["metrics"]["sensitivity"] for f in fold_results]
    specs = [f["metrics"]["specificity"] for f in fold_results]
    mean_auc = f"{sum(aucs)/len(aucs):.4f}" if aucs else "-"
    mean_row = f"""
        <tr style="font-weight:bold; background:#dce8f7;">
          <td>Mean</td>
          <td>{mean_auc}</td>
          <td>{sum(accs)/len(accs):.4f}</td>
          <td>{sum(f1s)/len(f1s):.4f}</td>
          <td>{sum(senss)/len(senss):.4f}</td>
          <td>{sum(specs)/len(specs):.4f}</td>
          <td>-</td><td>-</td><td>-</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; margin: 24px; }}
  h2 {{ color: #2c5f8a; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #2c5f8a; color: white; padding: 8px 12px; text-align: center; }}
  td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: center; }}
  tr:nth-child(even) {{ background: #f4f8fc; }}
</style></head>
<body>
<h2>TCGA-THCA External Validation — {model_version}</h2>
<p><b>Embedding:</b> {embed_model} &nbsp;|&nbsp; <b>Patch:</b> {patch_config} &nbsp;|&nbsp;
   <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<table>
  <thead><tr>
    <th>Checkpoint</th><th>AUC</th>
    <th>Acc</th><th>F1</th>
    <th>Sensitivity</th><th>Specificity</th><th>PPV</th><th>NPV</th><th>N slides</th>
  </tr></thead>
  <tbody>{rows}{mean_row}</tbody>
</table>
</body></html>"""
    return html


def upload_all_to_mlflow(fold_results, out_dir, model_version, embed_model="H-optimus-0", patch_config="40x 512x512", run_id=None):
    """전체 fold 결과를 하나의 MLflow run으로 업로드 (TCGA 외부검증 전용)"""
    try:
        import mlflow
    except ImportError:
        print("[WARN] mlflow 미설치 — MLflow 업로드 스킵")
        return

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("braf mutation")

    run_name = f"[TCGA] {model_version} {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(out_dir)

    # run_id가 있으면 기존 run에 이어서, 없으면 새 run 생성
    ctx = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=run_name)
    with ctx:
        mlflow.set_tag("eval_type", "external_validation_tcga")
        mlflow.set_tag("dataset", "TCGA-THCA")
        mlflow.set_tag("embedding_model", embed_model)
        mlflow.set_tag("patch_config", patch_config)
        mlflow.set_tag("internal_model_version", model_version)
        mlflow.set_tag("Description",
            f"[TCGA External Validation] {embed_model} ({patch_config}) → "
            f"{fold_results[0]['model_name'].upper()} → BRAF V600E. "
            f"Internal model: {model_version}, {len(fold_results)} folds.")

        mlflow.log_params({
            "model":           fold_results[0]["model_name"],
            "model_version":   model_version,
            "num_folds":       len(fold_results),
            "embedding_model": embed_model,
            "patch_config":    patch_config,
        })

        # fold별 metrics (fold_0, fold_1, ...)
        for fold in fold_results:
            fname = Path(fold["checkpoint"]).stem
            m = fold["metrics"]
            prefix = f"tcga_{fname}"
            mlflow.log_metrics({
                f"{prefix}_auc":         m["auc"] or 0.0,
                f"{prefix}_acc":         m["acc"],
                f"{prefix}_f1":          m["f1"],
                f"{prefix}_sensitivity": m["sensitivity"],
                f"{prefix}_specificity": m["specificity"],
            })

        # 평균 metrics
        aucs = [f["metrics"]["auc"] for f in fold_results if f["metrics"]["auc"]]
        mlflow.log_metrics({
            "tcga_mean_auc":         sum(aucs) / len(aucs) if aucs else 0.0,
            "tcga_mean_acc":         sum(f["metrics"]["acc"] for f in fold_results) / len(fold_results),
            "tcga_mean_f1":          sum(f["metrics"]["f1"]  for f in fold_results) / len(fold_results),
            "tcga_mean_sensitivity": sum(f["metrics"]["sensitivity"] for f in fold_results) / len(fold_results),
            "tcga_mean_specificity": sum(f["metrics"]["specificity"] for f in fold_results) / len(fold_results),
        })

        # fold별 JSON / 그래프 artifacts (heatmap, confusion, ROC, prob_dist는 분리 유지)
        for fold in fold_results:
            fname = Path(fold["checkpoint"]).stem
            for suffix in ["results.json", "confusion_matrix.png", "roc_curve.png", "prob_distribution.png"]:
                p = out_dir / f"{fname}_{suffix}"
                if p.exists():
                    artifact_path = "tcga_results" if suffix.endswith(".json") else "tcga_figures"
                    mlflow.log_artifact(str(p), artifact_path=artifact_path)
            heatmap_dir = out_dir / f"{fname}_heatmaps"
            if heatmap_dir.exists():
                for hf in sorted(heatmap_dir.glob("*.png"))[:20]:
                    mlflow.log_artifact(str(hf), artifact_path="tcga_heatmaps")

    print(f"[✓] TCGA MLflow 업로드 완료: {run_name}")


def run_inference(args, mlflow_run_id=None):
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
        slide_attns = {}  # slide_id → attn array (heatmap 선별용)
        skipped = 0

        # 1차: 전체 추론 (attention 수집)
        for npy_path in tqdm(npy_files, desc=f"Inferring {fold_name}"):
            slide_id = npy_path.stem
            if slide_id not in label_map:
                skipped += 1
                continue
            label = label_map[slide_id]
            prob, attn = infer_slide(model, str(npy_path), device,
                                     return_attention=args.save_heatmap)
            slide_preds.append({"slide_id": slide_id, "label": label, "prob": prob})
            if args.save_heatmap and attn is not None:
                slide_attns[slide_id] = attn

        # 2차: 확신도 상위 pos/neg 각 10장 heatmap 생성
        heatmap_dir = out_dir / f"{fold_name}_heatmaps"
        if args.save_heatmap and slide_attns:
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            coord_dir = Path(args.coord_dir) if args.coord_dir else Path(args.embedding_dir).parent / "json"

            pos_slides = sorted(
                [s for s in slide_preds if s["label"] == 1 and s["slide_id"] in slide_attns],
                key=lambda s: abs(s["prob"] - 0.5), reverse=True
            )[:10]
            neg_slides = sorted(
                [s for s in slide_preds if s["label"] == 0 and s["slide_id"] in slide_attns],
                key=lambda s: abs(s["prob"] - 0.5), reverse=True
            )[:10]

            for group, tag in [(pos_slides, "pos"), (neg_slides, "neg")]:
                for s in group:
                    sid = s["slide_id"]
                    coords = load_coords(str(coord_dir), sid)
                    if not coords:
                        continue
                    plot_attention_heatmap(
                        slide_attns[sid], coords, sid,
                        patch_size=args.patch_size,
                        out_path=heatmap_dir / f"{tag}_{sid}_attention.png",
                        prob=s["prob"],
                        svs_base_dir=getattr(args, 'svs_base_dir', None),
                        label=s["label"],
                    )
            print(f"[✓] Heatmap 저장: BRAF+ {len(pos_slides)}장 / BRAF- {len(neg_slides)}장 → {heatmap_dir}")

        print(f"[INFO] 추론: {len(slide_preds)}개 / 스킵(라벨없음): {skipped}개")

        labels = [s["label"] for s in slide_preds]
        probs  = [s["prob"]  for s in slide_preds]
        metrics = compute_metrics(labels, probs, threshold=args.threshold)

        auc_ci = metrics.get('auc_ci', [None, None])
        ci_str = f" (95% CI {auc_ci[0]:.4f}~{auc_ci[1]:.4f})" if auc_ci[0] else ""
        print(f"[결과] AUC={metrics['auc']:.4f}{ci_str}  Acc={metrics['acc']:.4f}  "
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

    # MLflow 전체 결과 — 기존 run에 이어 붙이거나 새 run 생성
    if not args.no_mlflow and fold_results:
        upload_all_to_mlflow(
            fold_results=fold_results,
            out_dir=out_dir,
            model_version=args.model_version,
            embed_model=args.embed_model,
            patch_config="40x 512x512" if "40x" in args.embedding_dir else "20x 256x256",
            run_id=mlflow_run_id,
        )

    return fold_results


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
    parser.add_argument("--save_heatmap",  action="store_true",
                        help="Attention heatmap 생성 여부")
    parser.add_argument("--coord_dir",     type=str, default=None,
                        help="패치 좌표 JSON 디렉토리 (embedding/h-optimus-0/40x/json/)")
    parser.add_argument("--patch_size",    type=int, default=512,
                        help="패치 크기 (20x=256, 40x=512)")
    parser.add_argument("--svs_base_dir",  type=str, default=None,
                        help="SVS 원본 디렉토리 (overlay용, TCGA는 UUID 하위 폴더 구조 자동 탐색)")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
