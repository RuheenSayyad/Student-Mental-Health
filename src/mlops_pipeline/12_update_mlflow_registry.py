import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient

MODEL_V2_PATH = Path("models/retrained_model_v2.pkl")
PREPROCESSOR_V2_PATH = Path("models/retrained_preprocessor_v2.pkl")
RETRAINING_REPORT_PATH = Path("outputs/retraining/retraining_report.json")
OUTPUT_REPORT_PATH = Path("outputs/retraining/registry_update_report.json")


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def main():
    print("=" * 70)
    print("REGISTER RETRAINED MODEL V2 IN MLFLOW REGISTRY (LAB 12)")
    print("=" * 70)

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join("configs", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    metric_to_optimize = config["registry"]["decision_metric"]

    if not MODEL_V2_PATH.exists():
        raise FileNotFoundError(f"{MODEL_V2_PATH} not found. Run 12_retrain_model.py first.")

    model_v2 = joblib.load(MODEL_V2_PATH)

    with mlflow.start_run(run_name="Register_Retrained_V2") as run:
        mlflow.log_param("retrained", True)
        if PREPROCESSOR_V2_PATH.exists():
            mlflow.log_artifact(str(PREPROCESSOR_V2_PATH), artifact_path="preprocessing_pipeline")

        print("[INFO] Logging model V2 to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model_v2,
            name="random_forest_model",
            registered_model_name=model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    # Fetch latest registered version
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_v2 = versions[0] if versions else None

    if latest_v2:
        print(f"[SUCCESS] Registered new version: Version {latest_v2.version}")
        # Transition new version to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=latest_v2.version,
            stage="Staging",
            archive_existing_versions=False,
        )

        # Champion-Challenger Faceoff
        prod_models = [v for v in versions if v.current_stage == "Production"]
        current_champ = prod_models[0] if prod_models else None

        promoted = False
        if not current_champ:
            client.transition_model_version_stage(
                name=model_name, version=latest_v2.version, stage="Production"
            )
            promoted = True
            print(f"[INFO] Version {latest_v2.version} promoted to Production (no incumbent).")
        else:
            # Promote challenger
            client.transition_model_version_stage(
                name=model_name, version=latest_v2.version, stage="Production"
            )
            client.transition_model_version_stage(
                name=model_name, version=current_champ.version, stage="Archived"
            )
            promoted = True
            print(f"[SUCCESS] Version {latest_v2.version} promoted! Version {current_champ.version} archived.")

        # Update local deployment bundle
        deployment_dir = Path("deployment")
        deployment_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_v2, deployment_dir / "model.pkl")
        if PREPROCESSOR_V2_PATH.exists():
            joblib.dump(joblib.load(PREPROCESSOR_V2_PATH), deployment_dir / "preprocessor.pkl")

        metadata = {
            "model_name": model_name,
            "model_version": str(latest_v2.version),
            "model_stage": "Production",
            "run_id": latest_v2.run_id,
            "source": "MLflow Registry (Retrained V2)",
        }
        with open(deployment_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        report = {
            "status": "success",
            "model_name": model_name,
            "new_version": str(latest_v2.version),
            "promoted": promoted,
            "stage": "Production",
        }
        OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        print(f"[SUCCESS] Registry update report saved to: {OUTPUT_REPORT_PATH}")
        print("=" * 70)


if __name__ == "__main__":
    main()
