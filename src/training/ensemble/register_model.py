"""
outputs/{version}/checkpoints/ 에서 선택한 .pt를 MLflow Registered Model에 등록.
ensemble_results.json에서 메트릭을 자동으로 읽어와 tags/description에 반영.

사용법:
    # production 등록 (기본)
    python src/training/ensemble/register_model.py \
        --model_path outputs/braf_ensemble_v1.0.0/checkpoints/model_2_auc0.9200.pt

    # staging으로 등록
    python src/training/ensemble/register_model.py \
        --model_path outputs/braf_ensemble_v1.0.0/checkpoints/model_2_auc0.9200.pt \
        --alias staging

환경변수:
    MLFLOW_TRACKING_URI: MLflow 서버 주소 (기본: http://localhost:5000)
"""

import os
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

import re
import sys
import json
import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

MODEL_NAME = "thyr-braf"


def load_model_metrics(pt_path: Path, ensemble_dir: Path = None) -> dict:
    """ensemble_results.json에서 해당 model_id의 test 메트릭을 가져옴."""
    candidates = []
    if ensemble_dir:
        candidates.append(ensemble_dir / "ensemble_results.json")
    candidates.append(pt_path.parent.parent / "ensemble_results.json")
    candidates.append(pt_path.parent / "ensemble_results.json")

    results_json = None
    for c in candidates:
        if c.exists():
            results_json = c
            break

    if results_json is None:
        print(f"[!] ensemble_results.json not found (tried: {candidates})")
        return {}

    with open(results_json) as f:
        results = json.load(f)

    match = re.search(r'model_?(\d+)', pt_path.name)
    if not match:
        print(f"[!] Cannot extract model_id from filename: {pt_path.name}")
        return {}

    mid = int(match.group(1))
    for model_data in results.get("models", []):
        if model_data.get("model_id") == mid:
            test_m = model_data.get("test_metrics", {})
            ens_m = results.get("ensemble_metrics", {})
            return {"model_id": mid, "test": test_m, "ensemble": ens_m}

    print(f"[!] model_id {mid} not found in ensemble_results.json")
    return {}


def register(model_path: str, alias: str = "production", ensemble_dir: str = None):
    pt_path = Path(model_path).resolve()
    if not pt_path.exists():
        print(f"[!] File not found: {pt_path}")
        sys.exit(1)

    ens_dir = Path(ensemble_dir).resolve() if ensemble_dir else None
    metrics = load_model_metrics(pt_path, ensemble_dir=ens_dir)
    test_m = metrics.get("test", {})
    ens_m = metrics.get("ensemble", {})
    mid = metrics.get("model_id", "?")
    pt_name = pt_path.name

    print(f"Model    : {pt_name}")
    print(f"Model ID : {mid}")
    if test_m:
        print(f"Test AUC : {test_m.get('auc', 'N/A')}")
        print(f"Test Acc : {test_m.get('accuracy', 'N/A')}")
        print(f"Test F1  : {test_m.get('f1', 'N/A')}")
    if ens_m:
        print(f"Ens AUC  : {ens_m.get('auc', 'N/A')}")

    mlflow.set_experiment("braf mutation")

    with mlflow.start_run(run_name=f"thyr-braf_{pt_name}"):
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "checkpoint": pt_name,
            "model_arch": "ABMIL_Gated",
            "embedding": "UNI2-H (1536-dim)",
        })

        for k, v in test_m.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", float(v))
        for k, v in ens_m.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"ensemble_{k}", float(v))

        mlflow.log_artifact(str(pt_path), artifact_path="model")
        print(f"[✓] Artifact uploaded")

        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        source = f"runs:/{run_id}/model/{pt_name}"

        desc = (
            f"BRAF Mutation Best Model | Backbone: ABMIL + UNI2-H (1536-dim)\n"
            f"Test AUC: {test_m.get('auc', 'N/A')}, "
            f"Acc: {test_m.get('accuracy', 'N/A')}, "
            f"F1: {test_m.get('f1', 'N/A')}\n"
            f"Ensemble AUC: {ens_m.get('auc', 'N/A')}"
        )

        try:
            client.get_registered_model(MODEL_NAME)
            client.update_registered_model(MODEL_NAME, description=desc)
        except Exception:
            client.create_registered_model(MODEL_NAME, description=desc)

        version_desc = (
            f"Model {mid} | Test AUC: {test_m.get('auc', 'N/A')} | "
            f"Ensemble AUC: {ens_m.get('auc', 'N/A')}"
        )
        mv = client.create_model_version(
            name=MODEL_NAME, source=source, run_id=run_id,
            description=version_desc
        )

        client.set_model_version_tag(MODEL_NAME, mv.version, "framework", "torchscript")
        client.set_model_version_tag(MODEL_NAME, mv.version, "task", "mil")
        client.set_model_version_tag(MODEL_NAME, mv.version, "embedding", "UNI2-H (1536-dim)")
        client.set_model_version_tag(MODEL_NAME, mv.version, "model_arch", "ABMIL_Gated")
        client.set_model_version_tag(MODEL_NAME, mv.version, "model_id", str(mid))
        for k in ("auc", "accuracy", "f1", "sensitivity", "specificity"):
            if k in test_m:
                client.set_model_version_tag(MODEL_NAME, mv.version, f"test_{k}", str(test_m[k]))
        if "auc" in ens_m:
            client.set_model_version_tag(MODEL_NAME, mv.version, "ensemble_auc", str(ens_m["auc"]))

        client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
        print(f"[✓] Registered: {MODEL_NAME} version {mv.version} (alias: {alias})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Register best .pt to '{MODEL_NAME}' Registered Model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--alias", type=str, default="production",
                        choices=["production", "staging"],
                        help="Alias to set (default: production)")
    parser.add_argument("--ensemble_dir", type=str, default=None,
                        help="Path to ensemble version dir containing ensemble_results.json")
    args = parser.parse_args()

    register(model_path=args.model_path, alias=args.alias, ensemble_dir=args.ensemble_dir)
