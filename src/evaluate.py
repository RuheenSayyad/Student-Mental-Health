import os
import sys
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils import load_and_split


def run_evaluation():
    print("Starting Model Evaluation...")
    os.makedirs("outputs", exist_ok=True)

    # 1. Load test data and the trained model
    X_test_final = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")
    model = joblib.load("models/random_forest_baseline.pkl")

    # 2. Make predictions
    y_pred = model.predict(X_test_final)

    # 3. Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-------------------------------\n")

    # 4. Error analysis export
    # Re-create the same split on the cleaned raw data to get the original student rows back
    _, X_test_raw, _, _ = load_and_split(impute_missing=True)

    errors_df = X_test_raw.copy()
    errors_df["Actual_Depression"] = y_test
    errors_df["Predicted_Depression"] = y_pred

    false_negatives = errors_df[
        (errors_df["Actual_Depression"] == 1) & (errors_df["Predicted_Depression"] == 0)
    ]
    false_positives = errors_df[
        (errors_df["Actual_Depression"] == 0) & (errors_df["Predicted_Depression"] == 1)
    ]

    false_negatives.to_csv("outputs/false_negatives.csv", index_label="row_id")
    false_positives.to_csv("outputs/false_positives.csv", index_label="row_id")

    print(f"False negatives (missed depression cases): {len(false_negatives)}")
    print(f"False positives (over-flagged students)  : {len(false_positives)}")
    print("Evaluation complete and error analysis files saved!")


if __name__ == "__main__":
    run_evaluation()
