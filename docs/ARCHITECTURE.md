# Student Mental Health MLOps Architecture

## System Overview
The Student Mental Health Prediction system is an end-to-end production MLOps platform designed to assess student depression risk based on academic, psychological, and lifestyle indicators.

```
       [Raw Survey Data]
              │
              ▼
    [01_load_data.py] ──> Lineage & Archive (student_vXX.csv)
              │
              ▼
    [02_validate_data.py] ──> Pandera Schema Validation
              │
              ▼
    [03_preprocessing.py] ──> Feature Engineering & Imputation
              │
              ▼
    [04_validate_outputs.py] ──> Dimensionality & NaN Checks
              │
              ▼
    [05_train_registry.py] ──> MLflow Tracking & Registry
              │
              ▼
    [06_model_evaluation.py] ──> Confusion Matrix & Metrics
              │
              ▼
    [07_quality_gate.py] ──> Recall & F1 Verification
              │
              ▼
    [08_automate_lifecycle.py] ──> Champion-Challenger Tournament
              │
              ▼
    [11_prepare_deployment.py] ──> Deployment Artifacts Bundle
              │
              ▼
       [FastAPI Service] ──> Docker / Kubernetes NodePort 8000
              │
              ▼
    [Continuous Monitoring] ──> Data, Prediction & Concept Drift
              │
              ▼
   [Automated Retraining] ──> Closes the MLOps Loop
```

## Features & Modeling
- **Numeric Features**:
  - `age`: Student age (imputed with median)
  - `year_of_study`: Academic year (1 through 4)
  - `cgpa_numeric`: Numeric midpoint of CGPA grade band
  - `symptom_burden_score`: Additive score of anxiety + panic attacks
  - `senior_high_achiever`: High performer flag (Year >= 3 and CGPA >= 3.5)
- **Categorical Features**:
  - `gender`: Male / Female
  - `marital_status`: Yes / No
  - `anxiety`: Yes / No
  - `panic_attack`: Yes / No
  - `specialist_treatment`: Yes / No
  - `faculty_group`: Course mapped to faculty cluster (Computing_IT, Engineering, Business_Economics, etc.)
- **Primary Optimization Metric**: Recall (captures positive depression cases to prioritize student intervention).
