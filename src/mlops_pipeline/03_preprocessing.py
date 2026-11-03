import os
import sys
import json
import joblib
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

CGPA_MIDPOINTS = {
    "0 - 1.99": 1.00,
    "2.00 - 2.49": 2.25,
    "2.50 - 2.99": 2.75,
    "3.00 - 3.49": 3.25,
    "3.50 - 4.00": 3.75,
}

NUMERIC_FEATURES = [
    "age", "year_of_study", "cgpa_numeric",
    "symptom_burden_score", "senior_high_achiever",
]

CATEGORICAL_FEATURES = [
    "gender", "marital_status", "anxiety", "panic_attack",
    "specialist_treatment", "faculty_group",
]


def categorize_course(course_name):
    """Group the free-text course names into faculty clusters."""
    c = str(course_name).lower().strip()
    tokens = set(c.replace("-", " ").split())
    if c.startswith("engin") or c == "koe":
        return "Engineering"
    if tokens & {"bcs", "bit", "it", "cts"} or "computer" in c:
        return "Computing_IT"
    if any(k in c for k in ["islam", "pendidikan", "fiqh", "irkhs", "kirkhs", "usuluddin", "taasl"]):
        return "Islamic_Studies"
    if any(k in c for k in ["econ", "accounting", "bank", "kenms", "enm", "human resource", "business"]):
        return "Business_Economics"
    if any(k in c for k in ["bio", "medic", "nurs", "radio", "marine", "mhsc"]):
        return "Health_Sciences"
    if c in {"law", "laws"}:
        return "Law"
    if "psych" in c:
        return "Psychology"
    return "Arts_Humanities"


def clean_dataframe(df):
    """Apply deterministic dataset cleaning shared by training and inference."""
    d = df.copy()

    # Normalize column names if raw headers are present
    col_mapping = {
        "Choose your gender": "gender",
        "Age": "age",
        "What is your course?": "course",
        "Your current year of Study": "year_of_study",
        "What is your CGPA?": "cgpa_range",
        "Marital status": "marital_status",
        "Do you have Depression?": "depression",
        "Do you have Anxiety?": "anxiety",
        "Do you have Panic attack?": "panic_attack",
        "Did you seek any specialist for a treatment?": "specialist_treatment",
    }
    for old_col, new_col in col_mapping.items():
        if old_col in d.columns:
            d[new_col] = d[old_col]

    if "Timestamp" in d.columns:
        d = d.drop(columns=["Timestamp"])
    if "timestamp" in d.columns:
        d = d.drop(columns=["timestamp"])

    text_cols = [
        "gender", "course", "year_of_study", "cgpa_range", "marital_status",
        "anxiety", "panic_attack", "specialist_treatment",
    ]
    for col in text_cols:
        if col in d.columns:
            d[col] = d[col].astype(str).str.strip()

    if "age" in d.columns:
        d["age"] = pd.to_numeric(d["age"], errors="coerce")
    if "year_of_study" in d.columns:
        d["year_of_study"] = d["year_of_study"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    if "cgpa_range" in d.columns:
        d["cgpa_numeric"] = d["cgpa_range"].map(CGPA_MIDPOINTS)

    if "anxiety" in d.columns and "panic_attack" in d.columns:
        d["symptom_burden_score"] = (
            (d["anxiety"] == "Yes").astype(int) + (d["panic_attack"] == "Yes").astype(int)
        )
    else:
        d["symptom_burden_score"] = 0

    if "year_of_study" in d.columns and "cgpa_numeric" in d.columns:
        d["senior_high_achiever"] = (
            (d["year_of_study"] >= 3) & (d["cgpa_numeric"] >= 3.5)
        ).astype(int)
    else:
        d["senior_high_achiever"] = 0

    if "course" in d.columns:
        d["faculty_group"] = d["course"].apply(categorize_course)

    if "depression" in d.columns:
        d["depression"] = d["depression"].apply(lambda x: 1 if str(x).strip().lower() in {"1", "yes"} else 0).astype(int)

    feature_cols = [col for col in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if col in d.columns]
    if "depression" in d.columns:
        feature_cols.append("depression")

    return d[feature_cols]


def create_initial_test_ids(df, config, processed_dir):
    """Create and save fixed test student IDs."""
    id_column = config["data"]["id_column"]
    test_ids_path = os.path.join(processed_dir, "test_student_ids.csv")

    if os.path.exists(test_ids_path):
        test_ids = pd.read_csv(test_ids_path)
        return test_ids[id_column].astype(str).tolist()

    print("[INFO] test_student_ids.csv not found.")
    print("[INFO] Creating fixed stratified test split...")

    working_df = df.copy()
    if id_column not in working_df.columns:
        working_df[id_column] = [f"STU{i+1:03d}" for i in range(len(working_df))]

    cleaned = clean_dataframe(working_df)
    y = cleaned["depression"]
    X = working_df

    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )

    test_ids = X_test[id_column].astype(str).tolist()
    pd.DataFrame({id_column: test_ids}).to_csv(test_ids_path, index=False)

    print(f"[SUCCESS] Original test-set IDs saved to {test_ids_path}")
    print(f"[INFO] Fixed test set size: {len(test_ids)}")

    return test_ids


def split_raw_data(df, config, processed_dir):
    """Keep the original test set fixed while using all other records for training."""
    id_column = config["data"]["id_column"]
    test_ids_path = os.path.join(processed_dir, "test_student_ids.csv")

    working_df = df.copy()
    if id_column not in working_df.columns:
        working_df[id_column] = [f"STU{i+1:03d}" for i in range(len(working_df))]

    if os.path.exists(test_ids_path):
        test_ids_df = pd.read_csv(test_ids_path)
        test_ids = set(test_ids_df[id_column].astype(str))
        print(f"[INFO] Using fixed test set from {test_ids_path}")
    else:
        test_ids = set(create_initial_test_ids(working_df, config, processed_dir))
        print("[INFO] Original test set has been permanently fixed.")

    test_mask = working_df[id_column].astype(str).isin(test_ids)
    test_df = working_df.loc[test_mask].copy()
    train_df = working_df.loc[~test_mask].copy()

    clean_test = clean_dataframe(test_df)
    clean_train = clean_dataframe(train_df)

    y_test = clean_test["depression"].astype(int)
    y_train = clean_train["depression"].astype(int)

    X_test = clean_test.drop(columns=["depression"])
    X_train = clean_train.drop(columns=["depression"])

    print(f"[INFO] Training records: {len(X_train)}")
    print(f"[INFO] Fixed test records: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def build_preprocessor():
    """Create modular scikit-learn preprocessing pipeline."""
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline(steps=[
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def run_preprocessing():
    print("[INFO] Starting Preprocessing Pipeline...")

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join("configs", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    raw_df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = split_raw_data(raw_df, config, processed_dir)

    preprocessor = build_preprocessor()

    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)

    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    np.save(os.path.join(processed_dir, "X_train_final.npy"), X_train_final)
    np.save(os.path.join(processed_dir, "X_test_final.npy"), X_test_final)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train.to_numpy(dtype=np.int64))
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test.to_numpy(dtype=np.int64))

    X_train.to_csv(os.path.join(processed_dir, "X_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test_raw.csv"), index=False)

    preprocessor_path = config["paths"]["preprocessor"]
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "dataset_name": config["dataset"]["name"],
        "target_column": config["data"]["target_column"],
        "train_shape_raw": list(X_train.shape),
        "test_shape_raw": list(X_test.shape),
        "train_shape_transformed": list(X_train_final.shape),
        "test_shape_transformed": list(X_test_final.shape),
        "numerical_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "split_test_size": config["split"]["test_size"],
        "split_random_state": config["split"]["random_state"],
        "fixed_test_set": True,
        "test_student_ids_path": os.path.join(processed_dir, "test_student_ids.csv"),
        "raw_train_path": os.path.join(processed_dir, "X_train_raw.csv"),
        "raw_test_path": os.path.join(processed_dir, "X_test_raw.csv"),
    }

    with open(os.path.join(processed_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[SUCCESS] Preprocessing complete. Pipeline saved as {preprocessor_path}.")
    return True


if __name__ == "__main__":
    passed = run_preprocessing()
    sys.exit(0 if passed else 1)
