"""Shared constants and data-cleaning helpers used by every lab script.

Kept in one place so the raw-data cleaning, the train/test split and the
column lists are identical across preprocess.py, preprocess_pipeline.py,
evaluate.py, the MLflow scripts and the tests.
"""
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()

RAW_PATH = CFG["dataset"]["raw_path"]
DATASET_NAME = CFG["dataset"]["name"]
TARGET = CFG["dataset"]["target"]
TEST_SIZE = CFG["dataset"]["test_size"]
RANDOM_STATE = CFG["dataset"]["random_state"]

TRACKING_URI = CFG["mlflow"]["tracking_uri"]
EXPERIMENT_NAME = CFG["mlflow"]["experiment_name"]
REGISTERED_MODEL_NAME = CFG["mlflow"]["registered_model_name"]
DECISION_METRIC = CFG["mlflow"]["decision_metric"]

RAW_COLUMNS = [
    "timestamp", "gender", "age", "course", "year_of_study", "cgpa_range",
    "marital_status", "depression", "anxiety", "panic_attack",
    "specialist_treatment",
]

CGPA_MIDPOINTS = {
    "0 - 1.99": 1.00,
    "2.00 - 2.49": 2.25,
    "2.50 - 2.99": 2.75,
    "3.00 - 3.49": 3.25,
    "3.50 - 4.00": 3.75,
}

# Explicit column lists (more robust than select_dtypes across pandas versions)
NUMERIC_FEATURES = [
    "age", "year_of_study", "cgpa_numeric",
    "symptom_burden_score", "senior_high_achiever",
]
CATEGORICAL_FEATURES = [
    "gender", "marital_status", "anxiety", "panic_attack",
    "specialist_treatment", "faculty_group",
]


def setup_mlflow():
    """Point MLflow at the local SQLite backend (needed for the registry)."""
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow


def ensure_dirs():
    for d in ("data/processed", "models", "outputs", "artifacts", "reports", "logs"):
        os.makedirs(d, exist_ok=True)


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


def clean_dataframe(df, impute_missing=True):
    """Clean the raw survey frame into model-ready columns.

    impute_missing=True  -> fills missing age / CGPA with the median
                            (baseline, Lab 3/4 behaviour, like churn's TotalCharges fill)
    impute_missing=False -> leaves NaNs so the sklearn Pipeline imputer handles
                            them after the split (Lab 5+, no leakage).
    The target column ``depression`` is returned as an int (Yes -> 1, No -> 0).
    """
    d = df.copy()
    d.columns = RAW_COLUMNS
    d = d.drop(columns=["timestamp"])

    text_cols = [
        "gender", "course", "year_of_study", "cgpa_range", "marital_status",
        "depression", "anxiety", "panic_attack", "specialist_treatment",
    ]
    for col in text_cols:
        d[col] = d[col].astype(str).str.strip()

    d["age"] = pd.to_numeric(d["age"], errors="coerce")
    d["year_of_study"] = d["year_of_study"].str.extract(r"(\d+)")[0].astype(int)
    d["cgpa_numeric"] = d["cgpa_range"].map(CGPA_MIDPOINTS)

    d["symptom_burden_score"] = (
        (d["anxiety"] == "Yes").astype(int) + (d["panic_attack"] == "Yes").astype(int)
    )
    d["senior_high_achiever"] = (
        (d["year_of_study"] >= 3) & (d["cgpa_numeric"] >= 3.5)
    ).astype(int)
    d["faculty_group"] = d["course"].apply(categorize_course)

    d[TARGET] = d[TARGET].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0).astype(int)

    if impute_missing:
        d["age"] = d["age"].fillna(d["age"].median())
        d["cgpa_numeric"] = d["cgpa_numeric"].fillna(d["cgpa_numeric"].median())

    return d[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]


def load_and_split(impute_missing=True):
    """Load the raw CSV, clean it and return the stratified split (DataFrames)."""
    raw = pd.read_csv(RAW_PATH)
    df = clean_dataframe(raw, impute_missing=impute_missing)
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
