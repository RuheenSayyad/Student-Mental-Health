import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

UPDATED_DATASET_PATH = Path("data/processed/updated_training_data.csv")
TEST_FEATURES_PATH = Path("data/processed/X_test_raw.csv")
TEST_LABELS_PATH = Path("data/processed/y_test.npy")

MODEL_V2_PATH = Path("models/retrained_model_v2.pkl")
PREPROCESSOR_V2_PATH = Path("models/retrained_preprocessor_v2.pkl")
REPORT_PATH = Path("outputs/retraining/retraining_report.json")

NUMERIC_FEATURES = [
    "age", "year_of_study", "cgpa_numeric",
    "symptom_burden_score", "senior_high_achiever",
]

CATEGORICAL_FEATURES = [
    "gender", "marital_status", "anxiety", "panic_attack",
    "specialist_treatment", "faculty_group",
]


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def build_preprocessor():
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline(steps=[
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def main():
    print("=" * 70)
    print("RETRAIN MODEL ON UPDATED POST-DRIFT DATASET (LAB 12)")
    print("=" * 70)

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join("configs", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    if not UPDATED_DATASET_PATH.exists():
        raise FileNotFoundError(f"{UPDATED_DATASET_PATH} not found. Run 12_create_updated_dataset.py first.")

    updated_df = pd.read_csv(UPDATED_DATASET_PATH)
    y_train = updated_df["depression"].to_numpy(dtype=int)
    X_train = updated_df.drop(columns=["depression"])

    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH).astype(int)

    preprocessor = build_preprocessor()
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    # Challenger V2 hyperparameter configuration
    params = {
        "n_estimators": 200,
        "max_depth": 7,
        "random_state": 42,
        "class_weight": "balanced",
    }

    with mlflow.start_run(run_name="RandomForest_Retrained_V2") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest_V2")
        mlflow.log_param("retraining_trigger", "drift_detected")

        rf_v2 = RandomForestClassifier(**params)
        rf_v2.fit(X_train_final, y_train)

        y_pred = rf_v2.predict(X_test_final)
        y_proba = rf_v2.predict_proba(X_test_final)[:, 1]

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        }
        mlflow.log_metrics(metrics)

        MODEL_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf_v2, MODEL_V2_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_V2_PATH)

        mlflow.log_artifact(str(PREPROCESSOR_V2_PATH), artifact_path="preprocessing_pipeline")

        print("[SUCCESS] Model V2 successfully retrained!")
        print(f"Metrics: Recall={metrics['recall']} | F1={metrics['f1']} | ROC-AUC={metrics['roc_auc']}")

        report = {
            "status": "success",
            "model_version": "v2",
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "run_id": run.info.run_id,
            "metrics": metrics,
            "model_path": str(MODEL_V2_PATH),
            "preprocessor_path": str(PREPROCESSOR_V2_PATH),
        }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        print(f"Retraining report saved to: {REPORT_PATH}")
        print("=" * 70)


if __name__ == "__main__":
    main()
