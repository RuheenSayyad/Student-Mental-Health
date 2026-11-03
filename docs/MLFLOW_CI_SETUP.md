# MLflow CI/CD Setup Guide

## Tracking URI Configuration
- Local Environment: SQLite backend (`sqlite:///mlflow.db`)
- CI / Remote Environment: Configured via GitHub Repository Secret `MLFLOW_TRACKING_URI` (e.g., DagsHub, AWS EC2, or managed MLflow server).

## Environment Variable Precedence
All pipeline scripts check:
```python
tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
```
This guarantees zero code edits when switching between local SQLite and cloud hosted MLflow servers.
