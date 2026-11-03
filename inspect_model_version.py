import os
import mlflow
from mlflow.tracking import MlflowClient

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

model_name = "Student_Depression_Production_Model"
print(f"Querying Model Registry for '{model_name}'...")
try:
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        print(f"Version: {v.version} | Stage: {v.current_stage} | Run ID: {v.run_id} | Source: {v.source}")
except Exception as e:
    print(f"Error querying model registry: {e}")
