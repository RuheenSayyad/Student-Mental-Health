import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config():
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        config_path = PROJECT_ROOT / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def get_production_model(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    production_versions = [v for v in versions if v.current_stage == "Production"]

    if not production_versions:
        raise RuntimeError(f"No Production version found for '{model_name}'.")

    return production_versions[0]


def prepare_deployment():
    print("=" * 70)
    print("PREPARE PRODUCTION DEPLOYMENT ARTIFACTS")
    print("=" * 70)

    config = load_config()
    configure_mlflow(config)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]

    deployment_dir = PROJECT_ROOT / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    model_version = get_production_model(client, model_name)
    version = model_version.version
    run_id = model_version.run_id
    model_source = model_version.source

    print(f"[INFO] Production Model Name : {model_name}")
    print(f"[INFO] Production Version    : {version}")
    print(f"[INFO] MLflow Run ID         : {run_id}")

    print("[INFO] Loading Production model...")
    model_uri = f"models:/{model_name}/{version}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback to local model if MLflow loading fails in local dev
        local_model_path = PROJECT_ROOT / config["paths"]["model"]
        print(f"[WARN] Loading from local model path {local_model_path}")
        model = joblib.load(local_model_path)

    model_path = deployment_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"[SUCCESS] Model artifact saved to: {model_path}")

    print("[INFO] Resolving matching preprocessor...")
    final_preprocessor_path = deployment_dir / "preprocessor.pkl"

    try:
        downloaded = client.download_artifacts(
            run_id,
            "preprocessing_pipeline/preprocessor.pkl",
            dst_path=str(deployment_dir),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path != final_preprocessor_path and downloaded_path.exists():
            shutil.move(downloaded_path, final_preprocessor_path)
    except Exception as exc:
        print(f"[WARN] Could not download preprocessor from MLflow: {exc}")
        local_pre = PROJECT_ROOT / config["paths"]["preprocessor"]
        if local_pre.exists():
            shutil.copy2(local_pre, final_preprocessor_path)
            print(f"[INFO] Copied preprocessor from {local_pre}")

    # Verify artifacts
    loaded_model = joblib.load(model_path)
    loaded_preprocessor = joblib.load(final_preprocessor_path)
    assert loaded_model is not None, "Model verification failed"
    assert loaded_preprocessor is not None, "Preprocessor verification failed"

    metadata = {
        "model_name": model_name,
        "model_version": str(version),
        "model_stage": model_version.current_stage,
        "run_id": run_id,
        "model_source": model_source,
        "source": "MLflow",
    }

    metadata_path = deployment_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Save artifact report
    report_path = PROJECT_ROOT / "artifacts" / "production_model_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Metadata saved to: {metadata_path}")
    print("[SUCCESS] Production deployment bundle ready in deployment/")
    return True


if __name__ == "__main__":
    try:
        passed = prepare_deployment()
        sys.exit(0 if passed else 1)
    except Exception as exc:
        print(f"[ERROR] Deployment preparation failed: {exc}")
        sys.exit(1)
