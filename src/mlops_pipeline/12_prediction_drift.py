import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data

BASELINE_PATH = Path("data/processed/X_test_raw.csv")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_data.csv")
REPORT_PATH = Path("outputs/drift/prediction_drift_report.csv")

KS_THRESHOLD = 0.10


def load_data():
    if not BASELINE_PATH.exists() or not EVOLVED_PATH.exists():
        raise FileNotFoundError("Baseline or evolved dataset not found.")

    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)
    return baseline, evolved


def predict(model_loader, data):
    clean_data = clean_prediction_data(data)
    transformed = model_loader.preprocessor.transform(clean_data)
    predictions = model_loader.model.predict(transformed)
    probabilities = model_loader.model.predict_proba(transformed)[:, 1]
    return predictions, probabilities


def main():
    print("=" * 70)
    print("PREDICTION DRIFT ANALYSIS (LAB 12)")
    print("=" * 70)

    baseline, evolved = load_data()

    model_loader = ModelLoader()
    model_loader.load()

    base_preds, base_probs = predict(model_loader, baseline)
    evol_preds, evol_probs = predict(model_loader, evolved)

    pred_ks, pred_p = ks_2samp(base_preds, evol_preds)
    prob_ks, prob_p = ks_2samp(base_probs, evol_probs)

    pred_drift = pred_ks >= KS_THRESHOLD
    prob_drift = prob_ks >= KS_THRESHOLD

    results = {
        "prediction_distribution_ks": float(pred_ks),
        "prediction_distribution_p_value": float(pred_p),
        "prediction_distribution_status": "DRIFT" if pred_drift else "OK",
        "probability_distribution_ks": float(prob_ks),
        "probability_distribution_p_value": float(prob_p),
        "probability_distribution_status": "DRIFT" if prob_drift else "OK",
    }

    report = pd.DataFrame([results])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print(f"Prediction KS  : {pred_ks:.4f} -> {results['prediction_distribution_status']}")
    print(f"Probability KS : {prob_ks:.4f} -> {results['probability_distribution_status']}")
    print(f"Saved Report   : {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
