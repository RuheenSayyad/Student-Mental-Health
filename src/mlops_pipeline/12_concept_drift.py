import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data

BASELINE_X_PATH = Path("data/processed/X_test_raw.csv")
BASELINE_Y_PATH = Path("data/processed/y_test.npy")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")
REPORT_PATH = Path("outputs/drift/concept_drift_report.csv")


def evaluate(model_loader, X, y):
    clean_data = clean_prediction_data(X)
    transformed = model_loader.preprocessor.transform(clean_data)
    preds = model_loader.model.predict(transformed)
    probs = model_loader.model.predict_proba(transformed)[:, 1]

    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, probs),
    }


def main():
    print("=" * 70)
    print("CONCEPT DRIFT ANALYSIS (LAB 12)")
    print("=" * 70)

    if not BASELINE_X_PATH.exists() or not EVOLVED_PATH.exists():
        raise FileNotFoundError("Baseline or evolved dataset missing.")

    baseline_X = pd.read_csv(BASELINE_X_PATH)
    baseline_y = np.load(BASELINE_Y_PATH)

    evolved = pd.read_csv(EVOLVED_PATH)
    evolved_X = evolved.drop(columns=["depression"])
    evolved_y = evolved["depression"].to_numpy(dtype=int)

    model_loader = ModelLoader()
    model_loader.load()

    base_metrics = evaluate(model_loader, baseline_X, baseline_y)
    evol_metrics = evaluate(model_loader, evolved_X, evolved_y)

    results = []
    for metric in base_metrics:
        b_val = base_metrics[metric]
        e_val = evol_metrics[metric]
        results.append({
            "metric": metric,
            "baseline": round(float(b_val), 4),
            "evolved": round(float(e_val), 4),
            "change": round(float(e_val - b_val), 4),
        })

    report = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print(report.to_string(index=False))
    print(f"\n[INFO] Concept drift report saved to: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
