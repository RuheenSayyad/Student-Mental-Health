import os
import sys
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils import EXPERIMENT_NAME, TRACKING_URI


def train_and_track(run_name="RandomForest_Baseline", params=None,
                    model_class=RandomForestClassifier, model_family="RandomForest",
                    local_backup="models/random_forest_model.pkl"):
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
            "class_weight": "balanced",
        }

    print(f"\n--- Starting MLflow Run: {run_name} ---")

    # 1. Load data
    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Set experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        # 2. Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", model_family)

        # 3. Train model
        model = model_class(**params)
        model.fit(X_train, y_train)

        # 4. Evaluate predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        # 5. Log metrics to MLflow
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")

        # 6. Generate and log diagnostic plots
        os.makedirs("artifacts", exist_ok=True)

        # Confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax_cm, cmap="Blues",
            display_labels=["No Depression", "Depression"],
        )
        ax_cm.set_title(f"Confusion Matrix - {run_name}")
        cm_path = "artifacts/confusion_matrix.png"
        fig_cm.savefig(cm_path, bbox_inches="tight")
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # ROC curve
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc)
        ax_roc.set_title(f"ROC Curve - {run_name}")
        roc_path = "artifacts/roc_curve.png"
        fig_roc.savefig(roc_path, bbox_inches="tight")
        plt.close(fig_roc)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # 7. Log preprocessing metadata for lineage
        if os.path.exists("data/processed/dataset_metadata.json"):
            mlflow.log_artifact("data/processed/dataset_metadata.json", artifact_path="metadata")

        # 8. Log model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Save local backup
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, local_backup)
        print(f"Run '{run_name}' successfully tracked!")


if __name__ == "__main__":
    # Run 1: baseline
    train_and_track(run_name="RandomForest")

    # Run 2: shallower, larger forest (less overfitting on a small dataset)
    tuned_params = {
        "n_estimators": 200,
        "max_depth": 3,
        "random_state": 42,
        "class_weight": "balanced",
    }
    train_and_track(
        run_name="RandomForest_Shallow", params=tuned_params,
        local_backup="models/random_forest_shallow_model.pkl",
    )

    # Run 3: Logistic Regression baseline
    lr_params = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}
    train_and_track(
        run_name="LogisticRegression_Baseline", params=lr_params,
        model_class=LogisticRegression, model_family="LogisticRegression",
        local_backup="models/logistic_regression_model.pkl",
    )
