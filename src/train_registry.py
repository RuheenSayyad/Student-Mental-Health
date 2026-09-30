import os
import sys
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils import EXPERIMENT_NAME, TRACKING_URI, REGISTERED_MODEL_NAME


def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")

    # 1. Set up MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2. Load processed data (produced by the Lab 5 pipeline)
    try:
        X_train = np.load("data/processed/X_train_final.npy")
        X_test = np.load("data/processed/X_test_final.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run the Lab 5 pipeline first.")
        sys.exit(1)

    if not os.path.exists("models/preprocessor.pkl"):
        print("[ERROR] models/preprocessor.pkl not found. Please run the Lab 5 pipeline first.")
        sys.exit(1)

    # 3. Define model parameters
    # V1 (first registered version)
    # params = {
    #     "n_estimators": 100,
    #     "max_depth": 5,
    #     "random_state": 42,
    #     "class_weight": "balanced",
    # }
    # V2 (challenger)
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "random_state": 42,
        "class_weight": "balanced",
    }

    # with mlflow.start_run(run_name="RandomForest_Registry_V1") as run:
    with mlflow.start_run(run_name="RandomForest_Registry_V2") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")

        # 4. Train model
        print("[INFO] Training RandomForest model...")
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)

        # 5. Evaluate
        print("[INFO] Evaluating model...")
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        # 6. Log preprocessor pipeline (lineage)
        print("[INFO] Attaching preprocessing pipeline to model artifacts...")
        mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessing_pipeline")

        # 7. Log and register the model
        print("[INFO] Pushing model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            # artifact_path is deprecated in newer MLflow versions
            name="random_forest_model",
            registered_model_name=REGISTERED_MODEL_NAME,  # creates the registry entry
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"[SUCCESS] Metrics logged: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")
        print(f"[SUCCESS] Model successfully registered under name: '{REGISTERED_MODEL_NAME}'")

    # Local backup, consistent with models/preprocessor.pkl (used for offline inference)
    joblib.dump(rf_model, "models/random_forest_registry.pkl")


if __name__ == "__main__":
    train_and_register_model()
