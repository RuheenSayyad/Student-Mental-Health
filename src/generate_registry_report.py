import os
import sys
import json
import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils import TRACKING_URI, REGISTERED_MODEL_NAME


def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    model_name = REGISTERED_MODEL_NAME

    try:
        # Search for the model specifically in Production
        versions = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in versions if mv.current_stage == "Production"), None)

        if not champion:
            print("[ERROR] No model found in Production stage!")
            sys.exit(1)

        # Get the run details to extract metrics and parameters
        run = client.get_run(champion.run_id)

        # Build the deployment-ready artifact dictionary
        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source,
            },
            "performance_metrics": {
                "recall": run.data.metrics.get("recall"),
                "f1_score": run.data.metrics.get("f1_score"),
                "roc_auc": run.data.metrics.get("roc_auc"),
                "accuracy": run.data.metrics.get("accuracy"),
                "precision": run.data.metrics.get("precision"),
            },
            "hyperparameters": run.data.params,
            "preprocessing_dependency": "models/preprocessor.pkl",
        }

        os.makedirs("artifacts", exist_ok=True)
        report_path = "artifacts/production_model_report.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"[SUCCESS] Report successfully generated for Version {champion.version}")
        print(f"[INFO] Saved to: {report_path}")

    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    generate_registry_report()
