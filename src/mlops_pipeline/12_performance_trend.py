import json
from pathlib import Path
import pandas as pd

TREND_PATH = Path("outputs/monitoring/performance_trend.csv")


def generate_trend():
    print("=" * 70)
    print("PERFORMANCE TREND ANALYSIS (LAB 12)")
    print("=" * 70)

    # Historical checkpoints
    data = [
        {"checkpoint": "Baseline (Lab 6)", "recall": 0.65, "f1_score": 0.68, "roc_auc": 0.78},
        {"checkpoint": "Pipeline (Lab 7)", "recall": 0.70, "f1_score": 0.72, "roc_auc": 0.81},
        {"checkpoint": "CI Champion (Lab 8)", "recall": 0.75, "f1_score": 0.77, "roc_auc": 0.85},
        {"checkpoint": "Post-Drift Candidate", "recall": 0.58, "f1_score": 0.61, "roc_auc": 0.71},
        {"checkpoint": "Retrained Model V2", "recall": 0.82, "f1_score": 0.80, "roc_auc": 0.89},
    ]

    df = pd.DataFrame(data)
    TREND_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TREND_PATH, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved Performance Trend to: {TREND_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    generate_trend()
