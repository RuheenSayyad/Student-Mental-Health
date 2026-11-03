import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_REPORT = PROJECT_ROOT / "reports" / "lifecycle_report.json"
PREPARE_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "11_prepare_deployment.py"
DEPLOY_SCRIPT = PROJECT_ROOT / "deployment" / "deploy_production.py"
REPORT_PATH = PROJECT_ROOT / "reports" / "deployment_automation_report.json"


def load_lifecycle_report():
    if not LIFECYCLE_REPORT.exists():
        raise FileNotFoundError(f"Lifecycle report not found: {LIFECYCLE_REPORT}")

    with LIFECYCLE_REPORT.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_script(script_path):
    result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
    return result.returncode == 0


def main():
    print("=" * 70)
    print("DEPLOY IF MODEL WAS PROMOTED")
    print("=" * 70)

    report = load_lifecycle_report()

    promoted = report.get("promoted", False)
    candidate_version = report.get("candidate_version")
    production_version = report.get("production_version_after")

    print(f"[INFO] Candidate version : {candidate_version}")
    print(f"[INFO] Promoted          : {promoted}")
    print(f"[INFO] Production version: {production_version}")

    if not promoted:
        print("[INFO] Model was not promoted. Deployment skipped.")
        return True

    print(f"[INFO] Model Version {candidate_version} was promoted.")
    print("[INFO] Preparing Production deployment bundle...")

    if not run_script(PREPARE_SCRIPT):
        print("[ERROR] Deployment preparation failed.")
        return False

    print("[SUCCESS] Deployment bundle prepared.")

    # Check if deploy_production.py exists
    if DEPLOY_SCRIPT.exists():
        print("[INFO] Attempting production deployment execution...")
        result = subprocess.run([sys.executable, str(DEPLOY_SCRIPT)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print("[WARN] Docker/Kubernetes rollout did not finish (environment may not have Docker active). Bundle remains prepared.")
    else:
        print("[INFO] Deployment script not found; bundle remains staged in deployment/.")

    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
