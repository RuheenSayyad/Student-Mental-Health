import numpy as np
import pandas as pd
from pathlib import Path

FEATURES_PATH = Path("data/monitoring/drift/evolved_data.csv")
LABELS_PATH = Path("data/processed/y_test.npy")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")
RANDOM_SEED = 42


def create_concept_drift_dataset():
    if not FEATURES_PATH.exists():
        print(f"[ERROR] Evolved features not found at {FEATURES_PATH}")
        return

    if not LABELS_PATH.exists():
        print(f"[ERROR] Test labels not found at {LABELS_PATH}")
        return

    rng = np.random.default_rng(RANDOM_SEED)

    X = pd.read_csv(FEATURES_PATH)
    y = np.load(LABELS_PATH).astype(int)

    if len(X) != len(y):
        raise ValueError(f"Feature/label mismatch: X={len(X)}, y={len(y)}")

    y_evolved = y.copy()

    # Students with high symptom burden become more likely to develop depression
    if "symptom_burden_score" in X.columns:
        high_symptoms = np.where(X["symptom_burden_score"] >= 1)[0]
        flip_count = max(1, int(len(high_symptoms) * 0.35))
        selected = rng.choice(high_symptoms, size=flip_count, replace=False)
        y_evolved[selected] = 1

    # Students with low CGPA (< 2.75) under pressure show elevated depression rate
    if "cgpa_numeric" in X.columns:
        low_cgpa = np.where(X["cgpa_numeric"] < 2.75)[0]
        if len(low_cgpa) > 0:
            flip_count = max(1, int(len(low_cgpa) * 0.25))
            selected = rng.choice(low_cgpa, size=flip_count, replace=False)
            y_evolved[selected] = 1

    result = X.copy()
    result["depression"] = y_evolved

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("CONCEPT DRIFT DATASET CREATED (LAB 12)")
    print("=" * 70)
    print(f"Feature Source : {FEATURES_PATH}")
    print(f"Label Source   : {LABELS_PATH}")
    print(f"Output         : {OUTPUT_PATH}")
    print(f"Rows           : {len(result)}")
    print(f"Changed labels : {(y != y_evolved).sum()} out of {len(y)}")


if __name__ == "__main__":
    create_concept_drift_dataset()
