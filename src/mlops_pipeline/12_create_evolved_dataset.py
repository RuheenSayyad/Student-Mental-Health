import os
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE_PATH = Path("data/processed/X_test_raw.csv")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_data.csv")
RANDOM_SEED = 42


def create_evolved_dataset():
    if not SOURCE_PATH.exists():
        print(f"[ERROR] Source test set not found at {SOURCE_PATH}")
        return

    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.read_csv(SOURCE_PATH).copy()

    # 1. Numerical drift: Age (older graduate students cohort)
    if "age" in df.columns:
        df["age"] = (df["age"] + rng.normal(1.5, 1.0, len(df))).clip(18.0, 35.0).round(1)

    # 2. Numerical drift: CGPA (grade compression / distribution shift)
    if "cgpa_numeric" in df.columns:
        df["cgpa_numeric"] = (df["cgpa_numeric"] * rng.normal(0.92, 0.05, len(df))).clip(1.0, 4.0).round(2)

    # 3. Categorical & symptom drift: Anxiety and panic attack surge during semester peak
    if "anxiety" in df.columns:
        df["anxiety"] = rng.choice(["Yes", "No"], size=len(df), p=[0.65, 0.35])

    if "panic_attack" in df.columns:
        df["panic_attack"] = rng.choice(["Yes", "No"], size=len(df), p=[0.55, 0.45])

    # Recalculate symptom burden score
    if "anxiety" in df.columns and "panic_attack" in df.columns:
        df["symptom_burden_score"] = (
            (df["anxiety"] == "Yes").astype(int) + (df["panic_attack"] == "Yes").astype(int)
        )

    # 4. Faculty shift: surge in computing and engineering enrollments
    if "faculty_group" in df.columns:
        df["faculty_group"] = rng.choice(
            ["Computing_IT", "Engineering", "Business_Economics", "Health_Sciences", "Arts_Humanities"],
            size=len(df),
            p=[0.40, 0.30, 0.15, 0.10, 0.05],
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("EVOLVED DATASET CREATED (LAB 12)")
    print("=" * 70)
    print(f"Source : {SOURCE_PATH}")
    print(f"Output : {OUTPUT_PATH}")
    print(f"Rows   : {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Modified features: age, cgpa_numeric, anxiety, panic_attack, symptom_burden_score, faculty_group")


if __name__ == "__main__":
    create_evolved_dataset()
