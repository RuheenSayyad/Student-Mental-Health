import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

exp = client.get_experiment_by_name("Student_Depression_Prediction")
if not exp:
    print("Experiment 'Student_Depression_Prediction' not found.")
    sys.exit(0)

runs = client.search_runs(exp.experiment_id, max_results=5)
print(f"Found {len(runs)} recent run(s):")
for r in runs:
    print(f"\nRun ID: {r.info.run_id} | Status: {r.info.status}")
    print("Metrics:", r.data.metrics)
    print("Parameters:", r.data.params)
