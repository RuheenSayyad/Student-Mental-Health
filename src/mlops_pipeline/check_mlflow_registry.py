import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "Student_Depression_Production_Model"


def check_registry():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", TRACKING_URI))
    client = MlflowClient()

    print(f"[INFO] Inspecting Model Registry for '{MODEL_NAME}'...")
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            print("[INFO] No model versions found.")
            return

        print(f"{'Version':<10} {'Stage':<15} {'Status':<15} {'Run ID':<35}")
        print("-" * 75)
        for v in versions:
            print(f"{v.version:<10} {v.current_stage:<15} {v.status:<15} {v.run_id:<35}")
    except Exception as exc:
        print(f"[ERROR] Registry inspection failed: {exc}")


if __name__ == "__main__":
    check_registry()
