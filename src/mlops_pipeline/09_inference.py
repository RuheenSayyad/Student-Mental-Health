import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils import clean_dataframe, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def run_inference_demo():
    print("=" * 70)
    print("PRODUCTION INFERENCE")
    print("=" * 70)

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join("configs", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    model_name = config["registry"]["model_name"]
    model_uri = f"models:/{model_name}/Production"

    model = None
    try:
        print(f"[INFO] Loading Production model from MLflow: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        print(f"[WARN] Could not load from MLflow registry URI: {exc}")
        # Try deployment/model.pkl or models/student_model.pkl
        for fallback in ["deployment/model.pkl", config["paths"]["model"]]:
            if os.path.exists(fallback):
                print(f"[INFO] Loading fallback model from {fallback}")
                model = joblib.load(fallback)
                break

    if model is None:
        print("[ERROR] No model found for inference.")
        return False

    preprocessor_path = config["paths"]["preprocessor"]
    if not os.path.exists(preprocessor_path):
        preprocessor_path = "deployment/preprocessor.pkl"

    raw_data_path = config["data"]["raw_path"]

    if not os.path.exists(preprocessor_path):
        print(f"[ERROR] Preprocessor not found at {preprocessor_path}")
        return False

    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Raw dataset not found at {raw_data_path}")
        return False

    preprocessor = joblib.load(preprocessor_path)
    df = pd.read_csv(raw_data_path)

    sample_size = min(5, len(df))
    sample = df.sample(n=sample_size, random_state=42)

    cleaned_sample = clean_dataframe(sample, impute_missing=False)
    if "depression" in cleaned_sample.columns:
        cleaned_sample = cleaned_sample.drop(columns=["depression"])

    X_sample = preprocessor.transform(cleaned_sample)

    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)[:, 1]

    result = sample.copy()
    result["Predicted_Depression"] = predictions
    result["Depression_Probability"] = np.round(probabilities, 4)

    print("\n--- Production Inference Results ---")
    display_cols = [c for c in ["What is your course?", "Your current year of Study", "What is your CGPA?", "Predicted_Depression", "Depression_Probability"] if c in result.columns]
    print(result[display_cols].to_string(index=False))
    print("------------------------------------\n")

    os.makedirs("outputs", exist_ok=True)
    output_path = config["paths"]["inference_output"]
    result.to_csv(output_path, index=False)

    print(f"[SUCCESS] Predictions saved to {output_path}")
    return True


if __name__ == "__main__":
    passed = run_inference_demo()
    sys.exit(0 if passed else 1)
