# Lab 7: Reproducible End-to-End MLOps Pipeline

## Objective
Establish a deterministic, version-controlled, multi-stage machine learning pipeline for student mental health prediction that spans raw data ingestion to production-ready artifacts.

## Architecture Stages
1. **01_load_data.py**: Automated batch ingestion with SHA256 hashing and archiving into `data/raw/archive/`.
2. **02_validate_data.py**: Pandera schema validation catching missing fields, illegal categories, and corrupted numbers.
3. **03_preprocessing.py**: Domain feature engineering (symptom burden, CGPA midpoint, course faculty categorization) and sklearn pipeline serialization.
4. **04_validate_outputs.py**: Validates transformed matrices for NaN leakage and dimensionality consistency.
5. **05_train_registry.py**: Trains balanced RandomForestClassifier and registers candidate in MLflow Model Registry.
6. **06_model_evaluation.py**: Calculates test set confusion matrix and ROC curves.
7. **09_inference.py**: Validates offline production inference execution.

## Verification
Run the pipeline:
```bash
python src/mlops_pipeline/student_mental_health_pipeline.py
```
Output log is stored at `logs/student_mental_health_pipeline_run_log.json`.
