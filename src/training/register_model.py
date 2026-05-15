"""
Best fold .pt checkpoint를 MLflow Registered Model에 등록.

우선순위:
1) --model_path 가 지정되면 해당 파일 직접 등록
2) 지정되지 않으면 outputs 디렉터리에서 ABMIL 기반 가장 높은 test AUC + test F1 폴드를 자동 선택

사용법:
    # checkpoint 파일 직접 지정
    python src/training/register_model.py \
        --model_path outputs/Thyroid_prediction_model_v1.0.0/checkpoints/best_model_fold3_auc0.9300.pt

    # outputs에서 자동 선택
    python src/training/register_model.py \
        --outputs_dir outputs

    # v2 스타일 태그로 등록
    python src/training/register_model.py \
        --outputs_dir outputs \
        --compat v2

환경변수:
    MLFLOW_TRACKING_URI: MLflow 서버 주소 (기본: http://localhost:5000)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

MODEL_NAME = "thyr-braf"
MODEL_ARCH = "ABMIL_Gated"
EMBEDDING = "UNI2-H (1536-dim)"
TASK = "mil"
FRAMEWORK = "torchscript"
DEFAULT_COMPAT_MODE = "v1"


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


def _extract_fold_from_filename(name: str) -> Optional[int]:
    match = re.search(r"fold(\d+)", name)
    return int(match.group(1)) if match else None


def _extract_auc_from_filename(name: str) -> Optional[float]:
    match = re.search(r"_auc([0-9]+\.[0-9]+)", name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_f1_from_metrics(test_metrics: Dict[str, Any]) -> Optional[float]:
    test_f1 = _to_float(test_metrics.get("f1"))
    if test_f1 is not None:
        return test_f1

    precision = _to_float(test_metrics.get("precision"))
    if precision is None:
        precision = _to_float(test_metrics.get("ppv"))
    recall = _to_float(test_metrics.get("recall"))
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return 2 * (precision * recall) / (precision + recall)


def _format_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "N/A"


def load_fold_metrics(pt_path: Path) -> dict:
    """results_cv_summary_optimal.json에서 해당 fold의 test 메트릭을 가져온다."""
    results_json = pt_path.parent.parent / "results_cv_summary_optimal.json"
    if not results_json.exists():
        print(f"[!] results_cv_summary_optimal.json not found: {results_json}")
        return {}

    with open(results_json) as f:
        results = json.load(f)

    match = _extract_fold_from_filename(pt_path.name)
    if match is None:
        print(f"[!] Cannot extract fold number from filename: {pt_path.name}")
        return {}

    fold_num = int(match)
    for fold_data in results.get("folds", []):
        if fold_data.get("fold") == fold_num:
            test_m = fold_data.get("test_metrics", {})
            summary = results.get("summary_statistics", {})
            return {
                "fold": fold_num,
                "test": test_m,
                "summary": summary,
            }

    print(f"[!] fold {fold_num} not found in results_cv_summary_optimal.json")
    return {}


def load_version_summary(version_dir: Path) -> Optional[dict]:
    summary_path = version_dir / "results_cv_summary_optimal.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        return json.load(f)


def find_candidate_versions(outputs_dir: Path, output_version: Optional[str]) -> List[Path]:
    """outputs에서 등록 후보 버전 디렉터리를 수집한다."""
    if output_version:
        vdir = outputs_dir / output_version
        if not vdir.exists():
            raise FileNotFoundError(f"output_version not found: {vdir}")
        return [vdir]

    candidates = []
    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in ("old_version",) or child.name.startswith("braf_ensemble_") or child.name.startswith("kpi_eval_"):
            continue
        if not child.name.startswith("Thyroid_prediction_model_"):
            continue
        if not (child / "results_cv_summary_optimal.json").exists():
            continue
        if not (child / "checkpoints").exists():
            continue
        candidates.append(child)

    if not candidates:
        raise RuntimeError(f"No single-model versions found under outputs: {outputs_dir}")
    return candidates


def pick_best_checkpoint_from_outputs(
    outputs_dir: str,
    output_version: Optional[str] = None,
    model_filter: str = "abmil",
) -> Dict[str, Any]:
    """outputs 기준으로 test AUC + test F1 점수가 가장 높은 checkpoint 선택."""
    outputs_root = Path(outputs_dir).expanduser().resolve()
    version_dirs = find_candidate_versions(outputs_root, output_version)

    candidates = []
    for version_dir in version_dirs:
        summary = load_version_summary(version_dir)
        if summary is None:
            print(f"[!] results_cv_summary_optimal.json not found: {version_dir}")
            continue

        model_name = str(_safe_get(summary, "model_name", "") or "").lower()
        if model_filter and model_name and model_name not in ("none", "unknown", "null"):
            if model_filter.lower() not in model_name:
                print(f"[i] skip version {version_dir.name}: model_name={model_name}")
                continue

        folds = {}
        for fold_data in _safe_get(summary, "folds", []):
            fold_num = fold_data.get("fold")
            if fold_num is not None:
                folds[int(fold_num)] = fold_data

        ckpt_root = version_dir / "checkpoints"
        for ckpt_path in sorted(ckpt_root.glob("*.pt")):
            fold_num = _extract_fold_from_filename(ckpt_path.name)
            if fold_num is None:
                continue

            fold_data = folds.get(fold_num, {})
            test_m = _safe_get(fold_data, "test_metrics", {})
            test_auc = _to_float(test_m.get("auc"))
            if test_auc is None:
                test_auc = _extract_auc_from_filename(ckpt_path.name)
            if test_auc is None:
                continue

            test_f1 = _extract_f1_from_metrics(test_m)
            if test_f1 is None:
                test_f1 = 0.0

            candidates.append({
                "ckpt_path": ckpt_path,
                "version_dir": version_dir,
                "fold": fold_num,
                "test_auc": test_auc,
                "test_f1": test_f1,
                "test_metrics": test_m,
                "summary": _safe_get(summary, "summary_statistics", {}),
            })

    if not candidates:
        raise RuntimeError(
            f"No candidate checkpoint found from outputs_dir={outputs_root}, "
            f"output_version={output_version}, model_filter={model_filter}"
        )

    candidates.sort(key=lambda x: (x["test_auc"], x["test_f1"]), reverse=True)

    print("\n[info] candidate checkpoints (top 10):")
    for c in candidates[:10]:
        print(
            f"  - {c['version_dir'].name} / {c['ckpt_path'].name} "
            f"(fold {c['fold']}, test_auc={c['test_auc']:.6f}, test_f1={c['test_f1']:.6f})"
        )

    best = candidates[0]
    print(f"[info] selected best: {best['ckpt_path']}\n")

    return {
        "ckpt_path": best["ckpt_path"],
        "fold": best["fold"],
        "test_metrics": best["test_metrics"],
        "summary": best["summary"],
    }


def _build_description(test_m: Dict[str, Any], mean_auc: Optional[float]) -> str:
    return (
        "ABMIL TorchScript model\n"
        f"Backbone: ABMIL + UNI2-H (1536-dim), Test AUC: {_format_float(test_m.get('auc'))}, "
        f"Acc: {_format_float(test_m.get('accuracy'))}, F1: {_format_float(test_m.get('f1'))}\n"
        f"CV Mean AUC: {_format_float(mean_auc)}"
    )


def _build_version_description(test_m: Dict[str, Any], fold_num: Any) -> str:
    return (
        f"Fold {fold_num} | Test AUC: {_format_float(test_m.get('auc'))} "
        f"| Test F1: {_format_float(test_m.get('f1'))}"
    )


def build_version_tags(
    mode: str,
    fold_num: Any,
    test_metrics: Dict[str, Any],
    summary: Dict[str, Any],
    source_version: str,
) -> Dict[str, str]:
    if mode == "v2":
        tags = {
            "embedding": EMBEDDING,
            "model_arch": MODEL_ARCH,
            "framework": FRAMEWORK,
            "task": TASK,
            "model_type": "single (5-fold CV)",
            "fold": str(fold_num),
            "source_version": source_version,
        }
        for metric_key in ("auc", "accuracy", "acc", "f1", "sensitivity", "specificity"):
            if metric_key in test_metrics:
                tags[f"test_{metric_key}"] = str(test_metrics[metric_key])

        mean_auc = _to_float(_safe_get(_safe_get(summary, "auc", {}), "mean"))
        mean_f1 = _to_float(_safe_get(_safe_get(summary, "f1", {}), "mean"))
        if mean_auc is not None:
            tags["cv_mean_auc"] = str(mean_auc)
        if mean_f1 is not None:
            tags["cv_mean_f1"] = str(mean_f1)
        return tags

    return {
        "framework": FRAMEWORK,
        "task": TASK,
    }


def register(model_path: str, alias: str = "production", compat: str = DEFAULT_COMPAT_MODE):
    pt_path = Path(model_path).resolve()
    if not pt_path.exists():
        print(f"[!] File not found: {pt_path}")
        sys.exit(1)

    metrics = load_fold_metrics(pt_path)
    test_m = metrics.get("test", {})
    summary = metrics.get("summary", {})
    fold_num = metrics.get("fold", "?")
    pt_name = pt_path.name

    if not test_m:
        test_auc = _extract_auc_from_filename(pt_name)
        test_m = {"auc": test_auc} if test_auc is not None else {}
    test_auc = _to_float(test_m.get("auc"))
    test_f1 = _extract_f1_from_metrics(test_m)
    test_acc = _to_float(test_m.get("accuracy"))
    if test_f1 is not None and "f1" not in test_m:
        test_m["f1"] = test_f1

    mean_auc = _to_float(_safe_get(summary.get("auc", {}), "mean")) if isinstance(summary, dict) else None
    mean_f1 = _to_float(_safe_get(summary.get("f1", {}), "mean")) if isinstance(summary, dict) else None

    print(f"Model      : {pt_name}")
    print(f"Fold       : {fold_num}")
    print(f"Test AUC   : {_format_float(test_m.get('auc'))}")
    print(f"Test Acc   : {_format_float(test_m.get('accuracy'))}")
    print(f"Test F1    : {_format_float(test_f1)}")
    print(f"CV Mean AUC: {_format_float(mean_auc)}")
    print(f"CV Mean F1 : {_format_float(mean_f1)}")
    print(f"Compat     : {compat}")

    mlflow.set_experiment("braf mutation")

    with mlflow.start_run(run_name=f"thyr-braf_{pt_name}"):
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "checkpoint": pt_name,
            "model_arch": MODEL_ARCH,
            "embedding": EMBEDDING,
            "framework": FRAMEWORK,
            "task": TASK,
            "fold": fold_num,
        })

        if test_auc is not None:
            mlflow.log_metric("test_auc", float(test_auc))
        if test_f1 is not None:
            mlflow.log_metric("test_f1", float(test_f1))
        if test_acc is not None:
            mlflow.log_metric("test_accuracy", float(test_acc))
        if isinstance(mean_auc, float):
            mlflow.log_metric("cv_mean_auc", float(mean_auc))
        if isinstance(mean_f1, float):
            mlflow.log_metric("cv_mean_f1", float(mean_f1))

        mlflow.log_artifact(str(pt_path), artifact_path="model")
        print(f"[✓] Artifact uploaded")

        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        source = f"runs:/{run_id}/model/{pt_name}"

        desc = _build_description(test_m, mean_auc)
        try:
            client.get_registered_model(MODEL_NAME)
            client.update_registered_model(MODEL_NAME, description=desc)
        except Exception:
            client.create_registered_model(MODEL_NAME, description=desc)

        version_desc = _build_version_description(test_m, fold_num)
        mv = client.create_model_version(
            name=MODEL_NAME,
            source=source,
            run_id=run_id,
            description=version_desc,
        )

        tags = build_version_tags(
            mode=compat,
            fold_num=fold_num,
            test_metrics=test_m,
            summary=summary if isinstance(summary, dict) else {},
            source_version=pt_path.parent.parent.name,
        )
        for tag_name, tag_value in tags.items():
            client.set_model_version_tag(MODEL_NAME, mv.version, tag_name, tag_value)

        client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
        print(f"[✓] Registered: {MODEL_NAME} version {mv.version} (alias: {alias})")


def main():
    parser = argparse.ArgumentParser(
        description=f"Register best fold .pt to '{MODEL_NAME}' Registered Model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to .pt checkpoint. If omitted, auto-select from outputs",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Root outputs directory for auto-selection",
    )
    parser.add_argument(
        "--output_version",
        type=str,
        default=None,
        help="Version folder name (e.g. Thyroid_prediction_model_v1.0.0). If omitted, search all versions",
    )
    parser.add_argument(
        "--model_filter",
        type=str,
        default="abmil",
        help="Filter by model_name containing this string (default: abmil)",
    )
    parser.add_argument(
        "--compat",
        type=str,
        default=DEFAULT_COMPAT_MODE,
        choices=["v1", "v2"],
        help="Compat mode for model version tags",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="production",
        choices=["production", "staging"],
        help="Alias to set (default: production)",
    )
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        selected = pick_best_checkpoint_from_outputs(
            outputs_dir=args.outputs_dir,
            output_version=args.output_version,
            model_filter=args.model_filter,
        )
        model_path = str(selected["ckpt_path"])

    register(model_path=model_path, alias=args.alias, compat=args.compat)


if __name__ == "__main__":
    main()
