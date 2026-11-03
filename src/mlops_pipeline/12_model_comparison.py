import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

TEST_FEATURES_PATH = Path("data/processed/X_test_raw.csv")
TEST_LABELS_PATH = Path("data/processed/y_test.npy")

MODEL_V1_PATH = Path("models/student_model.pkl")
PREPROCESSOR_V1_PATH = Path("models/preprocessor.pkl")

MODEL_V2_PATH = Path("models/retrained_model_v2.pkl")
PREPROCESSOR_V2_PATH = Path("models/retrained_preprocessor_v2.pkl")

OUTPUT_PATH = Path("outputs/retraining/model_comparison_report.csv")


def evaluate(model, preprocessor, X_raw, y):
    X_clean = preprocessor.transform(X_raw)
    preds = model.predict(X_clean)
    probs = model.predict_proba(X_clean)[:, 1]

    return {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, probs)), 4),
    }


def main():
    print("=" * 70)
    print("MODEL COMPARISON: BASELINE (V1) VS RETRAINED (V2) (LAB 12)")
    print("=" * 70)

    if not TEST_FEATURES_PATH.exists() or not TEST_LABELS_PATH.exists():
        raise FileNotFoundError("Test data not found.")

    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH).astype(int)

    # Load V1
    m1 = joblib.load(MODEL_V1_PATH if MODEL_V1_PATH.exists() else "deployment/model.pkl")
    p1 = joblib.load(PREPROCESSOR_V1_PATH if PREPROCESSOR_V1_PATH.exists() else "deployment/preprocessor.pkl")

    # Load V2
    m2 = joblib.load(MODEL_V2_PATH if MODEL_V2_PATH.exists() else m1)
    p2 = joblib.load(PREPROCESSOR_V2_PATH if PREPROCESSOR_V2_PATH.exists() else p1)

    v1_metrics = evaluate(m1, p1, X_test, y_test)
    v2_metrics = evaluate(m2, p2, X_test, y_test)

    rows = []
    for metric in v1_metrics:
        val1 = v1_metrics[metric]
        val2 = v2_metrics[metric]
        rows.append({
            "metric": metric,
            "baseline_v1": val1,
            "retrained_v2": val2,
            "delta": round(float(val2 - val1), 4),
            "improvement": "YES" if val2 >= val1 else "NO",
        })

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(df.to_string(index=False))
    print(f"\n[SUCCESS] Model comparison report saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
