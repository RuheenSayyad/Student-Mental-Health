import numpy as np
import pandas as pd
from pathlib import Path

TRAIN_FEATURES_PATH = Path("data/processed/X_train_raw.csv")
TRAIN_LABELS_PATH = Path("data/processed/y_train.npy")

NEW_BATCH_PATH = Path("data/monitoring/drift/post_drift_training_data.csv")
UPDATED_DATASET_PATH = Path("data/processed/updated_training_data.csv")
SUMMARY_PATH = Path("outputs/retraining/updated_dataset_summary.csv")

RANDOM_SEED = 42
NEW_BATCH_SIZE = 50


def load_original_training_data():
    X_train = pd.read_csv(TRAIN_FEATURES_PATH)
    y_train = np.load(TRAIN_LABELS_PATH).astype(int)
    if len(X_train) != len(y_train):
        raise ValueError("Training feature/label length mismatch")
    return X_train, y_train


def create_post_drift_batch(X_train, y_train):
    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = rng.choice(len(X_train), size=NEW_BATCH_SIZE, replace=True)

    X_new = X_train.iloc[sample_indices].copy().reset_index(drop=True)
    y_new = y_train[sample_indices].copy()

    # Apply realistic post-drift patterns to new cohort
    if "age" in X_new.columns:
        X_new["age"] = (X_new["age"] + rng.normal(1.2, 0.8, len(X_new))).clip(18.0, 35.0).round(1)

    if "cgpa_numeric" in X_new.columns:
        X_new["cgpa_numeric"] = (X_new["cgpa_numeric"] * rng.normal(0.95, 0.05, len(X_new))).clip(1.0, 4.0).round(2)

    if "anxiety" in X_new.columns:
        X_new["anxiety"] = rng.choice(["Yes", "No"], size=len(X_new), p=[0.60, 0.40])

    if "panic_attack" in X_new.columns:
        X_new["panic_attack"] = rng.choice(["Yes", "No"], size=len(X_new), p=[0.50, 0.50])

    if "anxiety" in X_new.columns and "panic_attack" in X_new.columns:
        X_new["symptom_burden_score"] = (
            (X_new["anxiety"] == "Yes").astype(int) + (X_new["panic_attack"] == "Yes").astype(int)
        )

    # Label adaptation under higher stress
    high_burden = np.where(X_new["symptom_burden_score"] >= 1)[0]
    flip_count = max(1, int(len(high_burden) * 0.30))
    selected = rng.choice(high_burden, size=flip_count, replace=False)
    y_new[selected] = 1

    post_drift_df = X_new.copy()
    post_drift_df["depression"] = y_new

    NEW_BATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    post_drift_df.to_csv(NEW_BATCH_PATH, index=False)

    return X_new, y_new


def main():
    print("=" * 70)
    print("CREATE UPDATED RETRAINING DATASET (LAB 12)")
    print("=" * 70)

    X_train, y_train = load_original_training_data()
    X_new, y_new = create_post_drift_batch(X_train, y_train)

    train_baseline_df = X_train.copy()
    train_baseline_df["depression"] = y_train

    post_drift_df = X_new.copy()
    post_drift_df["depression"] = y_new

    updated_df = pd.concat([train_baseline_df, post_drift_df], ignore_index=True)

    UPDATED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated_df.to_csv(UPDATED_DATASET_PATH, index=False)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([
        {"dataset": "Baseline Training", "rows": len(train_baseline_df), "depression_rate": round(float(y_train.mean()), 4)},
        {"dataset": "Post-Drift Batch", "rows": len(post_drift_df), "depression_rate": round(float(y_new.mean()), 4)},
        {"dataset": "Updated Training Combined", "rows": len(updated_df), "depression_rate": round(float(updated_df['depression'].mean()), 4)},
    ])
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"Original Training rows : {len(train_baseline_df)}")
    print(f"Post-Drift Batch rows  : {len(post_drift_df)}")
    print(f"Total Combined rows    : {len(updated_df)}")
    print(f"Saved Updated Data     : {UPDATED_DATASET_PATH}")
    print(f"Saved Summary Report   : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
