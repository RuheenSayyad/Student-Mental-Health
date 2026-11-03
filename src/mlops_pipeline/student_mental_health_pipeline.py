import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

STAGES = [
    ("Data Loading", "src/mlops_pipeline/01_load_data.py"),
    ("Raw Data Validation", "src/mlops_pipeline/02_validate_data.py"),
    ("Preprocessing", "src/mlops_pipeline/03_preprocessing.py"),
    ("Processed Output Validation", "src/mlops_pipeline/04_validate_outputs.py"),
    ("Training + MLflow + Model Registry", "src/mlops_pipeline/05_train_registry.py"),
    ("Model Evaluation", "src/mlops_pipeline/06_model_evaluation.py"),
    ("Model Quality Gate", "src/mlops_pipeline/07_quality_gate.py"),
    ("Automated Model Lifecycle", "src/mlops_pipeline/08_automate_lifecycle.py"),
    ("Deployment Update", "src/mlops_pipeline/deploy_if_promoted.py"),
    ("Production Inference", "src/mlops_pipeline/09_inference.py"),
]


def load_data_load_report():
    report_path = "artifacts/data_load_report.json"
    if not os.path.exists(report_path):
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(force=True):
    print("=" * 70)
    print("STUDENT MENTAL HEALTH MLOPS END-TO-END PIPELINE")
    print("=" * 70)

    run_log = {
        "pipeline": "Student Mental Health MLOps",
        "stages": [],
        "force": force,
    }

    for stage_index, (stage_name, script_path) in enumerate(STAGES):
        print("\n" + "=" * 70)
        print(f"STAGE {stage_index + 1}: {stage_name}")
        print("=" * 70)

        result = subprocess.run([sys.executable, script_path])
        status = "PASSED" if result.returncode == 0 else "FAILED"

        run_log["stages"].append({
            "stage": stage_name,
            "status": status,
            "exit_code": result.returncode,
        })

        print(f"--> {stage_name}: {status}")

        if result.returncode != 0:
            print(f"[HALT] Pipeline stopped because {stage_name} failed.")
            run_log["overall_status"] = "FAILED"
            break

        if stage_index == 0 and not force:
            report = load_data_load_report()
            if report:
                data_status = report.get("status")
                if data_status in {"already_ingested", "no_new_data"}:
                    print(f"[INFO] Data load status: {data_status}. Skipping retraining.")
                    run_log["overall_status"] = "NO_NEW_DATA"
                    break

    all_passed = len(run_log["stages"]) == len(STAGES) and all(
        s["status"] == "PASSED" for s in run_log["stages"]
    )
    if all_passed:
        run_log["overall_status"] = "PASSED"

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/student_mental_health_pipeline_run_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)

    print("\n" + "=" * 70)
    if run_log.get("overall_status") == "PASSED":
        print("[SUCCESS] Student Mental Health MLOps Pipeline completed successfully.")
    elif run_log.get("overall_status") == "NO_NEW_DATA":
        print("[INFO] Pipeline finished cleanly with no new data.")
    else:
        print("[ERROR] Student Mental Health MLOps Pipeline failed.")

    print(f"[INFO] Pipeline run log: {log_path}")
    print("=" * 70)
    return run_log.get("overall_status") in {"PASSED", "NO_NEW_DATA"}


def main():
    parser = argparse.ArgumentParser(description="Student Mental Health MLOps Pipeline")
    parser.add_argument("--force", action="store_true", default=True, help="Force execute all stages even if no new data was ingested")
    args = parser.parse_args()

    passed = run_pipeline(force=args.force)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
