# Student Mental Health - Production MLOps Engineering Platform (Labs 1 - 13)

A comprehensive, production-grade Machine Learning Operations (MLOps) platform built for student depression risk screening and early mental health intervention. This project implements the full lifecycle from raw survey ingestion, Pandera schema validation, and MLflow tracking/registry to Docker containerization, Kubernetes serving, drift monitoring, and closed-loop automated retraining.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [MLOps Architecture & Directory Structure](#mlops-architecture--directory-structure)
3. [Lab-by-Lab Roadmap (Labs 1 - 13)](#lab-by-lab-roadmap-labs-1---13)
4. [Installation & Environment Setup](#installation--environment-setup)
5. [Executing the Master Pipeline](#executing-the-master-pipeline)
6. [API Serving & Endpoints (Lab 10 & 11)](#api-serving--endpoints-lab-10--11)
7. [Docker & Kubernetes Deployment (Lab 11)](#docker--kubernetes-deployment-lab-11)
8. [Continuous Monitoring & Drift Detection (Lab 12)](#continuous-monitoring--drift-detection-lab-12)
9. [Automated Retraining & Closed-Loop CI/CD (Lab 13)](#automated-retraining--closed-loop-cicd-lab-13)
10. [Resilience Testing & Failure Injection (Lab 9)](#resilience-testing--failure-injection-lab-9)
11. [Testing Suite](#testing-suite)

---

## Project Overview
University students face academic pressures and lifestyle challenges that can impact mental health. This platform ingests survey responses (demographics, course load, CGPA, anxiety, panic attacks, specialist treatments) and calculates depression probability and risk levels (`Low`, `Medium`, `High`) while adhering strictly to production engineering and clinical safety practices.

- **Primary Goal**: High-Recall screening (minimizing false negatives to catch all students needing support).
- **Core Algorithms**: Class-Balanced Random Forest Classifiers and Scikit-Learn Pipelines.
- **Tracking & Governance**: SQLite-backed MLflow experiment tracking and Model Registry.
- **Serving Architecture**: FastAPI microservice packaged via Docker and Kubernetes.

---

## MLOps Architecture & Directory Structure

```text
Student_Mental_Health_ML_Project/
├── .github/
│   └── workflows/
│       ├── lab8_ci.yml                   # Lab 8: Continuous Integration workflow
│       ├── lab9_failure_tests.yml        # Lab 9: Resilience & fault injection tests
│       ├── lab13_monitoring.yml          # Lab 13: Scheduled drift monitoring trigger
│       └── lab13_runner_test.yml         # Lab 13: Runner healthcheck
├── api/
│   ├── __init__.py
│   ├── main.py                           # FastAPI application endpoints
│   ├── model_loader.py                   # Verified deployment bundle loader
│   ├── monitoring_logger.py              # Real-time JSONL telemetry logger
│   ├── preprocessing.py                  # Serving inference preprocessor
│   ├── schemas.py                        # Pydantic input/output contracts
│   ├── stage9_flow.txt
│   └── stages.txt
├── artifacts/                            # Quality gates, plots, & validation reports
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── baseline_validation.csv
│   ├── preprocessing_summary_report.json
│   ├── api_performance_report.json
│   └── failure_simulation_report.json
├── data/
│   ├── monitoring/
│   │   └── drift/
│   │       ├── evolved_data.csv          # Lab 12: Covariate shift dataset
│   │       ├── evolved_labeled_data.csv  # Lab 12: Concept drift dataset
│   │       └── post_drift_training_data.csv
│   ├── processed/
│   │   ├── dataset_metadata.json
│   │   ├── test_student_ids.csv          # Fixed held-out test cohort
│   │   ├── updated_training_data.csv     # Combined post-drift dataset
│   │   ├── X_train_final.npy, X_test_final.npy
│   │   ├── X_train_raw.csv, X_test_raw.csv
│   │   └── y_train.npy, y_test.npy
│   └── raw/
│       ├── Student_Mental_Health.csv     # Master survey dataset
│       ├── new_student_mental_health.csv # Incoming batch
│       └── archive/                      # Versioned lineage snapshots
├── deployment/
│   ├── deploy_production.py              # Automated Docker & K8s rollout
│   ├── model.pkl                         # Current Production model
│   ├── preprocessor.pkl                  # Matching fitted pipeline
│   └── model_metadata.json               # Lineage & MLflow tracking metadata
├── docs/
│   ├── ARCHITECTURE.md                   # Full system architecture diagram
│   ├── RUNBOOK.md                        # Production operations manual
│   ├── LAB7_PIPELINE.md                  # Lab 7 pipeline documentation
│   ├── LAB8_CI_REPORT.md                 # Lab 8 CI verification report
│   ├── LAB9_FAILURE_HANDLING_REPORT.md   # Lab 9 failure recovery report
│   ├── lab11_containerization_report.md  # Lab 11 container report
│   ├── lab11_deployment_debugging_report.md
│   ├── LAB13_END_TO_END.md               # Lab 13 continuous retraining guide
│   └── MLFLOW_CI_SETUP.md                # Remote tracking URI instructions
├── k8s/
│   ├── deployment.yaml                   # Kubernetes Deployment manifest
│   └── service.yaml                      # Kubernetes NodePort Service manifest
├── lab11_commands/
│   ├── lab11.13-kubernetes.txt
│   └── lab11_docker_kubernetes_reference.txt
├── logs/
│   ├── inference/
│   │   └── inference_log.jsonl           # Production request telemetry
│   └── student_mental_health_pipeline_run_log.json
├── models/
│   ├── preprocessor.pkl
│   ├── student_model.pkl
│   ├── retrained_model_v2.pkl
│   └── retrained_preprocessor_v2.pkl
├── outputs/
│   ├── drift/                            # Drift analysis outputs
│   ├── monitoring/                       # Monitor trigger outputs
│   └── retraining/                       # Model comparison outputs
├── reports/
│   ├── evaluation_report.json
│   ├── quality_gate_report.json
│   ├── lifecycle_report.json
│   ├── failure_report.json
│   └── deployment_automation_report.json
├── src/
│   ├── mlops_pipeline/
│   │   ├── 01_load_data.py               # Stage 1: Lineage & Ingestion
│   │   ├── 02_validate_data.py           # Stage 2: Schema Validation
│   │   ├── 03_preprocessing.py           # Stage 3: Feature Engineering
│   │   ├── 04_validate_outputs.py        # Stage 4: Matrix Sanity Checks
│   │   ├── 05_train_registry.py          # Stage 5: Training & MLflow Log
│   │   ├── 06_model_evaluation.py        # Stage 6: Evaluation & Plots
│   │   ├── 07_quality_gate.py            # Stage 7: Quality Threshold Gate
│   │   ├── 08_automate_lifecycle.py      # Stage 8: Champion-Challenger
│   │   ├── 09_inference.py               # Stage 9: Offline Batch Predict
│   │   ├── 10_failure_injection.py       # Stage 10: Fault Injection
│   │   ├── 11_prepare_deployment.py      # Stage 11: Deployment Bundler
│   │   ├── 12_*.py                       # Stage 12: Drift & Monitoring
│   │   ├── student_mental_health_pipeline.py # Master Pipeline Orchestrator
│   │   ├── deploy_if_promoted.py
│   │   └── validation.py
│   └── utils.py
├── tests/
│   ├── test_api.py                       # FastAPI endpoint test suite
│   ├── test_local_vs_container.py        # Model serving parity tests
│   ├── test_performance.py              # Latency & throughput benchmarks
│   └── test_pipeline.py                  # Core logic unit tests
├── Dockerfile
├── .dockerignore
├── .gitignore
├── config.yaml
├── requirements.txt
├── requirements-api.txt
└── start_lab_13.txt
```

---

## Lab-by-Lab Roadmap (Labs 1 - 13)

| Lab | Title | Purpose & Core Components |
|---|---|---|
| **Lab 1 & 2** | Git & Environment Setup | Repository initialization, Python venv, and dataset inspection. |
| **Lab 3** | Baseline Model | Baseline Random Forest classifier and confusion matrix. |
| **Lab 4** | Experiment Tracking | MLflow run tracking with hyperparameter tuning & metrics. |
| **Lab 5** | Preprocessing Pipeline | Leak-free Scikit-Learn `ColumnTransformer` with imputation & encoding. |
| **Lab 6** | Model Registry | Storing, versioning, and managing model stages in MLflow Registry. |
| **Lab 7** | Reproducible Pipeline | Modular `src/mlops_pipeline/` orchestrating stages 01 through 06. |
| **Lab 8** | CI & Quality Gates | GitHub Actions CI verifying `recall >= 0.60` quality gate. |
| **Lab 9** | Failure Handling & Recovery | 5 resilience scenarios (missing data, corrupted types, auto-recovery). |
| **Lab 10** | Production Serving Preparation | Packaging verified model, preprocessor, and metadata in `deployment/`. |
| **Lab 11** | Containerization & K8s | FastAPI service, Docker containerization, and Kubernetes manifests. |
| **Lab 12** | Continuous Monitoring & Drift | KS-test data drift, JS-divergence, prediction drift, and concept drift. |
| **Lab 13** | Autonomous Closed-Loop Retraining | Event/cron monitor trigger detecting drift and executing retraining. |

---

## Installation & Environment Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt
```

---

## Executing the Master Pipeline

Run the automated end-to-end pipeline covering ingestion, validation, preprocessing, training, evaluation, quality gate, lifecycle tournament, and deployment bundling:

```bash
python src/mlops_pipeline/student_mental_health_pipeline.py
```

Inspect run status:
```bash
cat logs/student_mental_health_pipeline_run_log.json
```

---

## API Serving & Endpoints (Lab 10 & 11)

Start the high-performance FastAPI prediction service:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Healthcheck**: `GET http://localhost:8000/health`
- **Metadata**: `GET http://localhost:8000/metadata`
- **Single Prediction**: `POST http://localhost:8000/predict`
  ```json
  {
    "student_id": "STU102",
    "gender": "Female",
    "age": 21,
    "course": "BCS",
    "year_of_study": "year 3",
    "cgpa_range": "3.50 - 4.00",
    "marital_status": "No",
    "anxiety": "Yes",
    "panic_attack": "Yes",
    "specialist_treatment": "No"
  }
  ```
- **Response**:
  ```json
  {
    "prediction": 1,
    "depression_probability": 0.8245,
    "risk_level": "High"
  }
  ```
- **Batch Predictions**: `POST http://localhost:8000/predict/batch`
- **Real-Time Logs**: `GET http://localhost:8000/logs?limit=10`

---

## Docker & Kubernetes Deployment (Lab 11)

### Docker Build & Run
```bash
docker build -t student-mental-health-api:1.0 .
docker run -d -p 8000:8000 --name smh-api student-mental-health-api:1.0
```

### Kubernetes Orchestration
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods -l app=student-mental-health-api
```

---

## Continuous Monitoring & Drift Detection (Lab 12)

Execute full multi-dimensional drift diagnostics:
```bash
# 1. Generate realistic covariate and concept drift data
python src/mlops_pipeline/12_create_evolved_dataset.py
python src/mlops_pipeline/12_create_concept_drift_dataset.py

# 2. Run statistical drift evaluations
python src/mlops_pipeline/12_data_drift.py
python src/mlops_pipeline/12_prediction_drift.py
python src/mlops_pipeline/12_concept_drift.py
python src/mlops_pipeline/12_drift_analysis.py

# 3. Model comparison & telemetry monitoring
python src/mlops_pipeline/12_api_performance_monitoring.py
python src/mlops_pipeline/12_prediction_monitoring.py
python src/mlops_pipeline/12_performance_trend.py
```

---

## Automated Retraining & Closed-Loop CI/CD (Lab 13)

The automated monitor trigger closes the MLOps loop:
```bash
python src/mlops_pipeline/12_monitor_trigger.py
```
If drift is detected and new data exists, it automatically:
1. Ingests the new student survey cohort.
2. Executes full pipeline retraining.
3. Tests the Quality Gate (`recall >= 0.60`).
4. Executes Champion-Challenger promotion.
5. Updates the Production deployment bundle in `deployment/`.

---

## Resilience Testing & Failure Injection (Lab 9)

Run the comprehensive fault-injection test harness:
```bash
python src/mlops_pipeline/10_failure_injection.py
```
Simulate schema corruptions:
```bash
python src/simulate_data_corruption.py
```

---

## Testing Suite

Run all unit, integration, and performance tests:
```bash
pytest tests/test_pipeline.py
pytest tests/test_api.py
pytest tests/test_local_vs_container.py
pytest tests/test_performance.py
```
