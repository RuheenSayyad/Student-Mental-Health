import pandas as pd
from pathlib import Path

DATA_DRIFT_PATH = Path("outputs/drift/data_drift_report.csv")
PREDICTION_DRIFT_PATH = Path("outputs/drift/prediction_drift_report.csv")
CONCEPT_DRIFT_PATH = Path("outputs/drift/concept_drift_report.csv")
OUTPUT_PATH = Path("outputs/drift/drift_analysis_report.csv")


def load_reports():
    data_drift = pd.read_csv(DATA_DRIFT_PATH) if DATA_DRIFT_PATH.exists() else pd.DataFrame()
    prediction_drift = pd.read_csv(PREDICTION_DRIFT_PATH) if PREDICTION_DRIFT_PATH.exists() else pd.DataFrame()
    concept_drift = pd.read_csv(CONCEPT_DRIFT_PATH) if CONCEPT_DRIFT_PATH.exists() else pd.DataFrame()
    return data_drift, prediction_drift, concept_drift


def main():
    print("=" * 70)
    print("COMBINED DRIFT ANALYSIS (LAB 12)")
    print("=" * 70)

    data_drift, prediction_drift, concept_drift = load_reports()

    # Data Drift Summary
    data_drift_features = len(data_drift[data_drift["status"] == "DRIFT"]) if "status" in data_drift.columns else 0

    # Prediction Drift Summary
    pred_status = prediction_drift.iloc[0]["prediction_distribution_status"] if not prediction_drift.empty else "OK"
    prob_status = prediction_drift.iloc[0]["probability_distribution_status"] if not prediction_drift.empty else "OK"

    # Concept Drift Summary
    concept_degraded = len(concept_drift[concept_drift["change"] < 0]) if "change" in concept_drift.columns else 0

    summary_rows = [
        {
            "category": "Data Drift",
            "metric": "Features with drift",
            "value": data_drift_features,
            "status": "DRIFT" if data_drift_features > 0 else "OK",
            "details": f"{data_drift_features} drifted features",
        },
        {
            "category": "Prediction Drift",
            "metric": "Prediction distribution",
            "value": prediction_drift.iloc[0]["prediction_distribution_ks"] if not prediction_drift.empty else 0.0,
            "status": pred_status,
            "details": "KS statistic on predicted classes",
        },
        {
            "category": "Prediction Drift",
            "metric": "Probability distribution",
            "value": prediction_drift.iloc[0]["probability_distribution_ks"] if not prediction_drift.empty else 0.0,
            "status": prob_status,
            "details": "KS statistic on depression probabilities",
        },
        {
            "category": "Concept Drift",
            "metric": "Degraded performance metrics",
            "value": concept_degraded,
            "status": "DRIFT" if concept_degraded > 0 else "OK",
            "details": f"{concept_degraded} metrics degraded",
        },
    ]

    report = pd.DataFrame(summary_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print(report.to_string(index=False))

    has_drift = (report["status"] == "DRIFT").any()
    print("-" * 70)
    if has_drift:
        print("OVERALL STATUS : DRIFT / MODEL BEHAVIOR CHANGE OBSERVED")
    else:
        print("OVERALL STATUS : NO SIGNIFICANT DRIFT OBSERVED")
    print(f"Combined report saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
