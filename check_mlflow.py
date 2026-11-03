import os
import mlflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
experiments = mlflow.search_experiments()
print(f"Found {len(experiments)} experiment(s):")
for exp in experiments:
    print(f" - ID: {exp.experiment_id} | Name: {exp.name} | Artifact Location: {exp.artifact_location}")
