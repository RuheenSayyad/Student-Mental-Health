import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config():
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        config_path = PROJECT_ROOT / "configs" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def run_script(script_path):
    script = PROJECT_ROOT / script_path
    result = subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT)
    return result.returncode


def restore_file(backup_path, original_path):
    if os.path.exists(original_path):
        os.remove(original_path)
    os.replace(backup_path, original_path)


def test_missing_data(config):
    print("\n[LAB 9] Scenario 1: Missing raw dataset")
    data_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup_path = Path(str(data_path) + ".lab9_backup")
    os.replace(data_path, backup_path)

    try:
        exit_code = run_script("src/mlops_pipeline/02_validate_data.py")
        passed = exit_code != 0
        return {
            "scenario": "missing_data",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed,
        }
    finally:
        restore_file(backup_path, data_path)


def test_invalid_datatype(config):
    print("\n[LAB 9] Scenario 2: Invalid datatype injection")
    data_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup_path = Path(str(data_path) + ".lab9_backup")
    shutil.copy2(data_path, backup_path)

    try:
        df = pd.read_csv(data_path)
        df["Age"] = df["Age"].astype(object)
        df.loc[0, "Age"] = "INVALID_AGE"
        df.to_csv(data_path, index=False)

        exit_code = run_script("src/mlops_pipeline/02_validate_data.py")
        passed = exit_code != 0
        return {
            "scenario": "invalid_datatype",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed,
        }
    finally:
        restore_file(backup_path, data_path)


def test_schema_mismatch(config):
    print("\n[LAB 9] Scenario 3: Schema mismatch (missing column)")
    data_path = PROJECT_ROOT / config["data"]["raw_path"]
    backup_path = Path(str(data_path) + ".lab9_backup")
    shutil.copy2(data_path, backup_path)

    try:
        df = pd.read_csv(data_path)
        col_to_drop = "What is your CGPA?" if "What is your CGPA?" in df.columns else df.columns[4]
        df = df.drop(columns=[col_to_drop])
        df.to_csv(data_path, index=False)

        exit_code = run_script("src/mlops_pipeline/02_validate_data.py")
        passed = exit_code != 0
        return {
            "scenario": "schema_mismatch",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed,
        }
    finally:
        restore_file(backup_path, data_path)


def test_preprocessing_inconsistency(config):
    print("\n[LAB 9] Scenario 4: Preprocessing output corruption (NaNs injected)")
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    path = processed_dir / "X_train_final.npy"
    backup_path = Path(str(path) + ".lab9_backup")

    if not path.exists():
        return {
            "scenario": "preprocessing_inconsistency",
            "expected": "HALT",
            "actual": "NOT_TESTED",
            "passed": False,
        }

    shutil.copy2(path, backup_path)
    try:
        X_train = np.load(path)
        X_train[0, 0] = np.nan
        np.save(path, X_train)

        exit_code = run_script("src/mlops_pipeline/04_validate_outputs.py")
        passed = exit_code != 0
        return {
            "scenario": "preprocessing_inconsistency",
            "expected": "HALT",
            "actual": "HALT" if passed else "CONTINUED",
            "passed": passed,
        }
    finally:
        restore_file(backup_path, path)


def test_missing_model_with_recovery(config):
    print("\n[LAB 9] Scenario 5: Missing local model artifact + recovery from MLflow")
    model_path = PROJECT_ROOT / config["paths"]["model"]
    model_name = config["registry"]["model_name"]
    backup_path = Path(str(model_path) + ".lab9_backup")

    if not model_path.exists():
        return {
            "scenario": "missing_model_artifact",
            "expected": "RECOVER",
            "actual": "NOT_TESTED",
            "passed": False,
        }

    shutil.copy2(model_path, backup_path)
    os.remove(model_path)

    try:
        configure_mlflow(config)
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest = versions[0] if versions else None

        if latest:
            recovered_model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest.version}")
            joblib.dump(recovered_model, model_path)
            recovered = model_path.exists()
        else:
            recovered = False

        return {
            "scenario": "missing_model_artifact",
            "expected": "RECOVER",
            "actual": "RECOVER" if recovered else "FAILED",
            "passed": recovered,
        }
    finally:
        if not model_path.exists() and backup_path.exists():
            restore_file(backup_path, model_path)
        elif backup_path.exists():
            os.remove(backup_path)


def main():
    print("=" * 70)
    print("LAB 9: SYSTEM RESILIENCE & FAILURE INJECTION TESTS")
    print("=" * 70)

    config = load_config()
    scenarios = [
        test_missing_data(config),
        test_invalid_datatype(config),
        test_schema_mismatch(config),
        test_preprocessing_inconsistency(config),
        test_missing_model_with_recovery(config),
    ]

    all_passed = all(s["passed"] for s in scenarios)

    report = {
        "lab": "Lab 9 - Failure Injection & Recovery",
        "total_tests": len(scenarios),
        "passed_tests": sum(1 for s in scenarios if s["passed"]),
        "status": "PASSED" if all_passed else "FAILED",
        "scenarios": scenarios,
    }

    # Save report in reports/ and artifacts/
    for out_path in [
        PROJECT_ROOT / "reports" / "failure_report.json",
        PROJECT_ROOT / "artifacts" / "failure_simulation_report.json",
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

    print("\n" + "=" * 70)
    print("FAILURE INJECTION RESULTS SUMMARY")
    print("=" * 70)
    for s in scenarios:
        status = "PASSED" if s["passed"] else "FAILED"
        print(f"[{status}] Scenario: {s['scenario']:30} Expected: {s['expected']:8} Actual: {s['actual']}")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
