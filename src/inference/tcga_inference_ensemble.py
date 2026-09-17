"""
TCGA-THCA External Validation — BRAF Ensemble (5 models, probability averaging)

tcga_inference.py는 체크포인트를 5-Fold CV처럼 하나씩 순회해 fold별 성능만 리포트하고
확률 평균 로직이 없다. 이 스크립트는 앙상블 5개 체크포인트의 확률을 slide_id 기준으로
정렬 집계해 평균한 뒤, 그 앙상블 확률로 성능을 계산한다.

핵심 설계:
- 체크포인트 5개의 model_id={1..5} 무결성을 로드 전에 검증한다.
- 확률은 리스트 위치가 아니라 slide_id로 정렬 집계한다 (모델별 순서가 다를 수 있음).
- tcga_ensemble_*(확률 평균 후 계산)와 tcga_mean_*(개별 fold 성능의 산술평균)는
  계산식이 다른 별개의 지표이므로 절대 혼용하지 않는다.
- attention heatmap은 1차 확률 추론으로 20장을 선정한 뒤 2차로만 재추론한다
  (508장 x 5모델 전체에 대해 attention을 뽑지 않아 메모리를 절감한다).

사용법:
    python tcga_inference_ensemble.py \
        --ckpt_dir outputs/braf_ensemble_hoptimus0_v0.1.0/checkpoints \
        --embedding_dir /path/to/TCGA-THCA/embedding/h-optimus-0/40x/npy \
        --label_csv /path/to/TCGA-THCA/genomic/braf_slide_labels.csv \
        --out_dir outputs/tcga_eval_ensemble_hoptimus0_v0.1.0 \
        --save_heatmap
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

from tcga_inference import (
    load_labels,
    load_checkpoint,
    filter_low_quality,
    infer_slide,
    compute_metrics,
    load_coords,
    plot_attention_heatmap,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_prob_distribution,
)

os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'


# =========================
# 6-1. 체크포인트 무결성 검증
# =========================
def load_and_verify_checkpoints(ckpt_dir, device):
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = sorted(ckpt_dir.glob("*.pt"))

    if len(ckpt_files) != 5:
        raise RuntimeError(
            f"앙상블 체크포인트는 정확히 5개여야 함 (발견: {len(ckpt_files)}개, {ckpt_dir}) — "
            f"재실행으로 남은 이전 버전 체크포인트가 섞여있지 않은지 확인하세요."
        )

    entries = []
    for p in ckpt_files:
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        entries.append({"path": p, "ckpt": ckpt})

    model_ids = [e["ckpt"]["model_id"] for e in entries]
    if set(model_ids) != {1, 2, 3, 4, 5}:
        raise RuntimeError(f"체크포인트 model_id 집합이 {{1..5}}가 아님: {sorted(model_ids)}")

    model_types = {e["ckpt"].get("config", {}).get("model") for e in entries}
    if len(model_types) != 1:
        raise RuntimeError(f"체크포인트 5개의 모델 구조가 서로 다름: {model_types}")

    in_dims = {e["ckpt"].get("config", {}).get("in_dim") for e in entries}
    in_dims_known = in_dims - {None}
    if len(in_dims_known) > 1:
        raise RuntimeError(f"체크포인트 5개의 in_dim이 서로 다름: {in_dims_known}")
    if None in in_dims:
        print("[WARN] 일부 체크포인트에 in_dim 필드 없음 (구버전) — in_dim 검증 스킵")

    entries.sort(key=lambda e: e["ckpt"]["model_id"])

    models = []
    for e in entries:
        model, model_name = load_checkpoint(str(e["path"]), device)
        models.append({
            "model_id": e["ckpt"]["model_id"],
            "model": model,
            "model_name": model_name,
            "val_auc": e["ckpt"].get("val_auc"),
            "checkpoint": e["path"].name,
        })

    print(f"[✓] 체크포인트 무결성 검증 통과: model_id={[m['model_id'] for m in models]}")
    return models


# =========================
# 6-2. slide_id 기준 정렬 집계 (위치 기반 평균 금지)
# =========================
def infer_all_models_probability_only(models, npy_files, label_map, device):
    """1차: 확률만 계산 (attention 없음). 반환: {model_id: {slide_id: prob}}, {slide_id: label}"""
    per_model_probs = {}
    label_by_slide = {}

    for m in models:
        model_id = m["model_id"]
        slide_probs = {}
        for npy_path in npy_files:
            slide_id = npy_path.stem
            if slide_id not in label_map:
                continue
            prob, _ = infer_slide(m["model"], str(npy_path), device, return_attention=False)
            slide_probs[slide_id] = prob
            label_by_slide.setdefault(slide_id, label_map[slide_id])
        per_model_probs[model_id] = slide_probs
        print(f"[INFO] Model {model_id} 확률 추론 완료: {len(slide_probs)} slides")

    return per_model_probs, label_by_slide


def build_ensemble_matrix(per_model_probs, label_by_slide):
    """slide_id 정렬 순서를 고정하고 (5, N) 확률 행렬을 구성. 슬라이드 집합 불일치 시 실패."""
    model_ids = sorted(per_model_probs.keys())
    slide_sets = [set(per_model_probs[mid].keys()) for mid in model_ids]

    reference = slide_sets[0]
    for mid, s in zip(model_ids, slide_sets):
        if s != reference:
            missing_here = reference - s
            extra_here = s - reference
            raise RuntimeError(
                f"Model {mid}의 slide 집합이 다른 모델과 불일치. "
                f"이 모델에 없는 slide: {sorted(missing_here)[:10]}, "
                f"이 모델에만 있는 slide: {sorted(extra_here)[:10]}"
            )

    slide_ids = sorted(reference)
    labels = np.array([label_by_slide[s] for s in slide_ids])

    prob_matrix = np.zeros((len(model_ids), len(slide_ids)), dtype=np.float64)
    for i, mid in enumerate(model_ids):
        for j, sid in enumerate(slide_ids):
            prob_matrix[i, j] = per_model_probs[mid][sid]

    return slide_ids, labels, prob_matrix, model_ids


# =========================
# 6-3. 앙상블 AUC와 개별 fold 평균 AUC — 계산식을 독립 재검증
# =========================
def compute_ensemble_and_mean_metrics(slide_ids, labels, prob_matrix, model_ids, threshold=0.5):
    from sklearn.metrics import roc_auc_score

    ensemble_probs = np.mean(prob_matrix, axis=0)
    ensemble_metrics = compute_metrics(labels.tolist(), ensemble_probs.tolist(), threshold=threshold)

    per_model_metrics = {}
    per_model_aucs = []
    for i, mid in enumerate(model_ids):
        m = compute_metrics(labels.tolist(), prob_matrix[i].tolist(), threshold=threshold)
        per_model_metrics[mid] = m
        per_model_aucs.append(m["auc"])

    mean_auc = float(np.mean(per_model_aucs))

    # 계산식 자체를 독립적으로 재확인 (단순 "다른 값인지"가 아니라 정의대로 나왔는지)
    recomputed_ensemble_auc = roc_auc_score(labels, ensemble_probs)
    assert abs(recomputed_ensemble_auc - ensemble_metrics["auc"]) < 1e-9, (
        "tcga_ensemble_auc가 roc_auc_score(labels, mean(probs))와 불일치"
    )
    recomputed_mean_auc = float(np.mean([
        roc_auc_score(labels, prob_matrix[i]) for i in range(len(model_ids))
    ]))
    assert abs(recomputed_mean_auc - mean_auc) < 1e-9, (
        "tcga_mean_auc가 개별 모델 AUC 산술평균과 불일치"
    )

    return {
        "ensemble_probs": ensemble_probs,
        "ensemble_metrics": ensemble_metrics,
        "per_model_metrics": per_model_metrics,
        "mean_auc": mean_auc,
    }


# =========================
# 6-4. heatmap — 2단계 추론 (probability-only -> 상위 20장 -> attention 재추론), skip on mismatch
# =========================
def generate_ensemble_heatmaps(models, slide_ids, labels, ensemble_probs, model_ids,
                                embedding_dir, coord_dir, out_dir, patch_size,
                                svs_base_dir, device, n_positive=10, n_negative=10):
    slide_to_label = dict(zip(slide_ids, labels))
    slide_to_prob = dict(zip(slide_ids, ensemble_probs))

    pos_slides = sorted(
        [s for s in slide_ids if slide_to_label[s] == 1],
        key=lambda s: abs(slide_to_prob[s] - 0.5), reverse=True
    )[:n_positive]
    neg_slides = sorted(
        [s for s in slide_ids if slide_to_label[s] == 0],
        key=lambda s: abs(slide_to_prob[s] - 0.5), reverse=True
    )[:n_negative]
    selected = pos_slides + neg_slides
    print(f"[INFO] heatmap 대상 선정: BRAF+ {len(pos_slides)}장, BRAF- {len(neg_slides)}장 (2차 attention 재추론)")

    models_by_id = {m["model_id"]: m for m in models}
    best_model_id = max(models, key=lambda m: m["val_auc"] or 0)["model_id"]

    ensemble_dir = Path(out_dir) / "ensemble_attention"
    individual_dir = Path(out_dir) / "individual_model_best"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    individual_dir.mkdir(parents=True, exist_ok=True)

    for slide_id in selected:
        npy_path = Path(embedding_dir) / f"{slide_id}.npy"
        if not npy_path.exists():
            print(f"  [WARN] {slide_id}: 임베딩 파일 없음 — heatmap skip")
            continue

        coords = load_coords(str(coord_dir), slide_id)
        if not coords:
            print(f"  [WARN] {slide_id}: 좌표 JSON 없음 — heatmap skip")
            continue

        # 2차: 선정된 슬라이드만 5개 모델 attention 재추론
        attn_by_model = {}
        for mid in model_ids:
            _, attn = infer_slide(models_by_id[mid]["model"], str(npy_path), device, return_attention=True)
            attn_by_model[mid] = attn

        lengths = {mid: (len(a) if a is not None else None) for mid, a in attn_by_model.items()}
        valid_lengths = {v for v in lengths.values() if v is not None}
        if len(valid_lengths) == 0 or any(v is None for v in lengths.values()) or len(valid_lengths) > 1 or len(coords) not in valid_lengths:
            print(f"  [WARN] {slide_id}: attention 길이/좌표 개수 불일치({lengths}, coords={len(coords)}) — heatmap skip")
            continue

        label = int(slide_to_label[slide_id])
        prob = float(slide_to_prob[slide_id])

        # (A) 앙상블 평균 attention
        avg_attn = np.mean([attn_by_model[mid] for mid in model_ids], axis=0)
        plot_attention_heatmap(
            avg_attn, coords, slide_id, patch_size=patch_size,
            out_path=ensemble_dir / f"{slide_id}_ensemble_attention.png",
            prob=prob, svs_base_dir=svs_base_dir, label=label,
        )

        # (B) 최고 val AUC 모델 attention
        plot_attention_heatmap(
            attn_by_model[best_model_id], coords, slide_id, patch_size=patch_size,
            out_path=individual_dir / f"{slide_id}_model{best_model_id}_attention.png",
            prob=prob, svs_base_dir=svs_base_dir, label=label,
        )

    print(f"[✓] heatmap 저장 완료: {ensemble_dir}, {individual_dir}")


# =========================
# 6-5. MLflow 연결 및 메트릭 키 분리
# =========================
def find_mlflow_run_id(ckpt_dir, explicit_run_id):
    if explicit_run_id:
        return explicit_run_id
    run_id_path = Path(ckpt_dir).parent / "mlflow_run_id.txt"
    if run_id_path.exists():
        return run_id_path.read_text().strip()
    return None


def upload_ensemble_tcga_to_mlflow(result, out_dir, embed_model, model_version, run_id=None):
    try:
        import mlflow
    except ImportError:
        print("[WARN] mlflow 미설치 — 업로드 스킵")
        return

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("braf mutation")

    from datetime import datetime
    run_name = f"[TCGA-Ensemble] {model_version} {datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with (mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=run_name)) as ctx:
        mlflow.set_tag("eval_type", "external_validation_tcga_ensemble")
        mlflow.set_tag("dataset", "TCGA-THCA")
        mlflow.set_tag("embedding_model", embed_model)
        mlflow.set_tag("internal_model_version", model_version)
        # 학습 run의 기존 "Description" 태그를 덮어쓰지 않도록 별도 키 사용
        mlflow.set_tag(
            "Description_TCGA",
            f"[TCGA External Validation - Ensemble] {embed_model} -> ABMIL x5 -> BRAF V600E. "
            f"Prediction: 5-model probability averaging. Internal model: {model_version}."
        )

        ensemble_metrics = result["ensemble_metrics"]
        mlflow.log_metrics({
            "tcga_ensemble_auc": ensemble_metrics["auc"] or 0.0,
            "tcga_ensemble_acc": ensemble_metrics["acc"],
            "tcga_ensemble_f1": ensemble_metrics["f1"],
            "tcga_ensemble_sensitivity": ensemble_metrics["sensitivity"],
            "tcga_ensemble_specificity": ensemble_metrics["specificity"],
        })

        for mid, m in result["per_model_metrics"].items():
            mlflow.log_metrics({
                f"tcga_model{mid}_auc": m["auc"] or 0.0,
                f"tcga_model{mid}_acc": m["acc"],
                f"tcga_model{mid}_f1": m["f1"],
            })

        # 참고용: 개별 fold AUC 산술평균 (앙상블 성능과 다른 지표 — 혼용 금지)
        mlflow.log_metric("tcga_mean_auc", result["mean_auc"])

        out_dir = Path(out_dir)
        results_json = out_dir / "tcga_ensemble_results.json"
        if results_json.exists():
            mlflow.log_artifact(str(results_json), artifact_path="tcga_results")

        for fname in ["tcga_ensemble_confusion_matrix.png", "tcga_ensemble_roc_curve.png",
                      "tcga_ensemble_prob_distribution.png"]:
            p = out_dir / fname
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="tcga_figures")

        for sub in ["ensemble_attention", "individual_model_best"]:
            heatmap_dir = out_dir / "heatmaps" / sub
            if heatmap_dir.exists():
                for hf in sorted(heatmap_dir.glob("*.png"))[:20]:
                    mlflow.log_artifact(str(hf), artifact_path=f"tcga_heatmaps/{sub}")

    print(f"[✓] TCGA 앙상블 MLflow 업로드 완료 (run_id={ctx.info.run_id})")


def run_inference_ensemble(args, mlflow_run_id=None):
    """
    앙상블 TCGA 외부검증 실행 (함수형 엔트리포인트).

    main.py의 run_inference(args, mlflow_run_id) 패턴과 동일하게,
    main_ensemble.py 학습 파이프라인에서 직접 import해 호출할 수 있도록 분리.
    mlflow_run_id가 주어지면 args.mlflow_run_id보다 우선한다 (학습 파이프라인이 즉시 전달하는 run_id).
    """
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    label_map = load_labels(args.label_csv)
    print(f"[INFO] 라벨 로드: {len(label_map)}개 슬라이드")

    npy_dir = Path(args.embedding_dir)
    npy_files = sorted(npy_dir.glob("*.npy"))
    npy_files = filter_low_quality(npy_files, min_patches=args.min_patches)

    # 6-1
    models = load_and_verify_checkpoints(args.ckpt_dir, device)

    # 6-2 (1차: 확률만, 전체 슬라이드)
    per_model_probs, label_by_slide = infer_all_models_probability_only(
        models, npy_files, label_map, device
    )
    slide_ids, labels, prob_matrix, model_ids = build_ensemble_matrix(per_model_probs, label_by_slide)
    print(f"[✓] slide_id 정렬 집계 완료: {len(slide_ids)} slides, {len(model_ids)} models")

    # 6-3
    result = compute_ensemble_and_mean_metrics(slide_ids, labels, prob_matrix, model_ids, args.threshold)
    print(f"[결과] Ensemble AUC={result['ensemble_metrics']['auc']:.4f}  "
          f"Acc={result['ensemble_metrics']['acc']:.4f}  F1={result['ensemble_metrics']['f1']:.4f}  "
          f"| Mean(individual) AUC={result['mean_auc']:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "description": "TCGA External Validation - BRAF Ensemble (5 models, probability averaging)",
        "threshold": args.threshold,
        "ensemble_metrics": result["ensemble_metrics"],
        "mean_auc_individual_models": result["mean_auc"],
        "per_model_metrics": result["per_model_metrics"],
        "num_slides": len(slide_ids),
    }
    with open(out_dir / "tcga_ensemble_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    plot_confusion_matrix(
        result["ensemble_metrics"]["tp"], result["ensemble_metrics"]["tn"],
        result["ensemble_metrics"]["fp"], result["ensemble_metrics"]["fn"],
        out_dir / "tcga_ensemble_confusion_matrix.png",
        title="Confusion Matrix (TCGA Ensemble)"
    )
    if result["ensemble_metrics"]["auc"]:
        plot_roc_curve(labels.tolist(), result["ensemble_probs"].tolist(),
                        result["ensemble_metrics"]["auc"], out_dir / "tcga_ensemble_roc_curve.png")
    plot_prob_distribution(labels.tolist(), result["ensemble_probs"].tolist(),
                            out_dir / "tcga_ensemble_prob_distribution.png")

    # 6-4 (2차: 선정된 20장만 attention 재추론)
    if args.save_heatmap:
        coord_dir = args.coord_dir or (Path(args.embedding_dir).parent / "json")
        generate_ensemble_heatmaps(
            models, slide_ids, labels, result["ensemble_probs"], model_ids,
            embedding_dir=args.embedding_dir, coord_dir=coord_dir,
            out_dir=out_dir / "heatmaps", patch_size=args.patch_size,
            svs_base_dir=args.svs_base_dir, device=device,
        )

    # 6-5
    if not args.no_mlflow:
        run_id = mlflow_run_id or find_mlflow_run_id(args.ckpt_dir, args.mlflow_run_id)
        upload_ensemble_tcga_to_mlflow(
            result, out_dir, embed_model=args.embed_model,
            model_version=args.model_version, run_id=run_id,
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--embedding_dir", type=str, required=True)
    parser.add_argument("--label_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_version", type=str, default="braf_ensemble_hoptimus0_v0.1.0")
    parser.add_argument("--embed_model", type=str, default="h-optimus-0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min_patches", type=int, default=1000)
    parser.add_argument("--no_mlflow", action="store_true")
    parser.add_argument("--save_heatmap", action="store_true")
    parser.add_argument("--coord_dir", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--svs_base_dir", type=str, default=None)
    parser.add_argument("--mlflow_run_id", type=str, default=None,
                        help="학습 run에 이어붙일 MLflow run_id (없으면 ckpt_dir 상위의 mlflow_run_id.txt 탐색)")
    args = parser.parse_args()

    run_inference_ensemble(args)


if __name__ == "__main__":
    main()
