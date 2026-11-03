# Operations Runbook - Student Mental Health Prediction

## 1. Pipeline Execution
To execute the master automated pipeline locally:
```bash
python src/mlops_pipeline/student_mental_health_pipeline.py
```

## 2. Model Registry Management
To inspect current models, stages, and metrics:
```bash
python src/mlops_pipeline/check_mlflow_registry.py
```

To run Champion-Challenger tournament manually:
```bash
python src/mlops_pipeline/08_automate_lifecycle.py
```

## 3. Serving API Locally
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Test endpoints:
- Root: `http://localhost:8000/`
- Health: `http://localhost:8000/health`
- Metadata: `http://localhost:8000/metadata`
- Swagger Docs: `http://localhost:8000/docs`

## 4. Docker & Kubernetes Deployment
Build and run Docker container:
```bash
docker build -t student-mental-health-api:1.0 .
docker run -p 8000:8000 student-mental-health-api:1.0
```
Deploy to Kubernetes cluster:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get svc student-mental-health-api-service
```

## 5. Drift Monitoring & Retraining
Simulate drift and execute detection:
```bash
python src/mlops_pipeline/12_create_evolved_dataset.py
python src/mlops_pipeline/12_create_concept_drift_dataset.py
python src/mlops_pipeline/12_data_drift.py
python src/mlops_pipeline/12_prediction_drift.py
python src/mlops_pipeline/12_concept_drift.py
python src/mlops_pipeline/12_drift_analysis.py
```
Trigger automated retraining:
```bash
python src/mlops_pipeline/12_monitor_trigger.py
```
