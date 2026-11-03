import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DRIFT_SCRIPT = Path("src/mlops_pipeline/12_drift_analysis.py")
DRIFT_REPORT = Path("outputs/drift/drift_analysis_report.csv")
NEW_DATA_PATH = Path("data/raw/new_student_mental_health.csv")
ALT_NEW_DATA = Path("data/raw/new_students.csv")
PIPELINE_SCRIPT = Path("src/mlops_pipeline/student_mental_health_pipeline.py")
TRIGGER_REPORT = Path("outputs/monitoring/monitor_trigger_report.json")
DATA_LOAD_REPORT = Path("artifacts/data_load_report.json")


def run_drift_analysis():
    print("[INFO] Running drift analysis...")
    if DRIFT_SCRIPT.exists():
        result = subprocess.run([sys.executable, str(DRIFT_SCRIPT)])
        return result.returncode == 0
    return False


def detect_drift():
    if not DRIFT_REPORT.exists():
        return False
    df = pd.read_csv(DRIFT_REPORT)
    for col in df.columns:
        if df[col].astype(str).str.strip().str.upper().eq("DRIFT").any():
            return True
    return False


def get_data_load_status():
    if not DATA_LOAD_REPORT.exists():
        return None
    try:
        with DATA_LOAD_REPORT.open("r", encoding="utf-8") as f:
            report = json.load(f)
        return report.get("status")
    except Exception:
        return None


def save_trigger_report(report):
    TRIGGER_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with TRIGGER_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=str)


def run_monitor_trigger():
    print("=" * 70)
    print("LAB 13: CONTINUOUS MONITORING & RETRAINING TRIGGER")
    print("=" * 70)

    drift_analysis_passed = run_drift_analysis()
    new_data_present = NEW_DATA_PATH.exists() or ALT_NEW_DATA.exists()

    if not drift_analysis_passed:
        report = {
            "status": "drift_analysis_failed",
            "drift_detected": False,
            "new_data_present": new_data_present,
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_trigger_report(report)
        print("[ERROR] Drift analysis failed.")
        return False

    drift_detected = detect_drift()
    print(f"[INFO] Drift Detected    : {drift_detected}")
    print(f"[INFO] New Batch Present : {new_data_present}")

    if not drift_detected:
        report = {
            "status": "no_drift",
            "drift_detected": False,
            "new_data_present": new_data_present,
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_trigger_report(report)
        print("[INFO] System healthy. No significant drift detected.")
        return True

    if not new_data_present:
        report = {
            "status": "drift_waiting_for_data",
            "drift_detected": True,
            "new_data_present": False,
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_trigger_report(report)
        print("[INFO] Drift detected, waiting for new survey dataset batch.")
        return True

    print("[ALERT] Drift confirmed and new dataset detected!")
    print("[INFO] Triggering automated Student Mental Health MLOps pipeline...")

    pipeline_result = subprocess.run([sys.executable, str(PIPELINE_SCRIPT)])

    data_load_status = get_data_load_status()
    new_data_ingested = (data_load_status == "ingested")
    retraining_executed = (new_data_ingested and pipeline_result.returncode == 0)

    status = "retraining_completed" if retraining_executed else "no_new_data_retraining_skipped"

    report = {
        "status": status,
        "drift_detected": True,
        "new_data_present": True,
        "pipeline_called": True,
        "pipeline_exit_code": pipeline_result.returncode,
        "pipeline_status": "passed" if pipeline_result.returncode == 0 else "failed",
        "data_load_status": data_load_status,
        "new_data_ingested": new_data_ingested,
        "retraining_executed": retraining_executed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_trigger_report(report)

    if pipeline_result.returncode != 0:
        print("[ERROR] Automated MLOps pipeline failed.")
        return False

    print("[SUCCESS] Automated continuous retraining lifecycle finished successfully.")
    return True


if __name__ == "__main__":
    passed = run_monitor_trigger()
    sys.exit(0 if passed else 1)
