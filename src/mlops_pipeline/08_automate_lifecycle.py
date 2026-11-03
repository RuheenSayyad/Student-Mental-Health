import json
import os
import sys

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def save_lifecycle_report(report):
    os.makedirs("reports", exist_ok=True)
    with open("reports/lifecycle_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)


def automate_model_lifecycle():
    print("=" * 70)
    print("AUTOMATED MODEL LIFECYCLE (CHAMPION-CHALLENGER)")
    print("=" * 70)

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join("configs", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)
    client = MlflowClient()
    model_name = config["registry"]["model_name"]
    metric_to_optimize = config["registry"]["decision_metric"]

    report = {
        "status": "success",
        "model_name": model_name,
        "decision_metric": metric_to_optimize,
        "candidate_version": None,
        "candidate_run_id": None,
        "candidate_metric": None,
        "production_version_before": None,
        "production_metric_before": None,
        "promoted": False,
        "decision": None,
        "production_version_after": None,
    }

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        new_models = [mv for mv in all_versions if mv.current_stage == "None"]

        for mv in new_models:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            print(f"[INFO] Version {mv.version} moved to Staging.")

        all_versions = client.search_model_versions(f"name='{model_name}'")
        staging_models = [mv for mv in all_versions if mv.current_stage == "Staging"]

        if not staging_models:
            print("[INFO] No models in Staging to evaluate.")
            report["decision"] = "no_staging_candidate"
            save_lifecycle_report(report)
            return True

        best_challenger = None
        best_challenger_score = -1.0

        for mv in staging_models:
            run = client.get_run(mv.run_id)
            score = run.data.metrics.get(metric_to_optimize, 0.0)
            print(f"[INFO] Staging Candidate: Version {mv.version} | {metric_to_optimize}: {score:.4f}")

            if score > best_challenger_score:
                best_challenger_score = score
                best_challenger = mv

        report["candidate_version"] = str(best_challenger.version)
        report["candidate_run_id"] = best_challenger.run_id
        report["candidate_metric"] = round(float(best_challenger_score), 4)

        production_models = [mv for mv in all_versions if mv.current_stage == "Production"]
        current_champion = production_models[0] if production_models else None

        promote_challenger = False

        if not current_champion:
            print("[INFO] No model in Production. Challenger promoted by default.")
            promote_challenger = True
            report["decision"] = "initial_promotion"
        else:
            champ_run = client.get_run(current_champion.run_id)
            champ_score = champ_run.data.metrics.get(metric_to_optimize, 0.0)
            report["production_version_before"] = str(current_champion.version)
            report["production_metric_before"] = round(float(champ_score), 4)

            print(f"[INFO] Current Champion: Version {current_champion.version} | {metric_to_optimize}: {champ_score:.4f}")

            if best_challenger_score > champ_score:
                print("[SUCCESS] Challenger beat the Champion! Promoting...")
                promote_challenger = True
                report["decision"] = "challenger_beat_champion"
            else:
                print("[INFO] Challenger failed to beat Champion. Retaining Champion.")
                report["decision"] = "champion_retained"
                report["production_version_after"] = str(current_champion.version)

        if promote_challenger:
            client.transition_model_version_stage(
                name=model_name,
                version=best_challenger.version,
                stage="Production",
                archive_existing_versions=False,
            )
            report["promoted"] = True
            report["production_version_after"] = str(best_challenger.version)

            if current_champion:
                client.transition_model_version_stage(
                    name=model_name,
                    version=current_champion.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
                print(f"[INFO] Archived defeated Champion (Version {current_champion.version}).")

        # Archive remaining staging models
        final_versions = client.search_model_versions(f"name='{model_name}'")
        remaining_staging = [mv for mv in final_versions if mv.current_stage == "Staging"]
        for mv in remaining_staging:
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage="Archived",
                archive_existing_versions=False,
            )
            print(f"[INFO] Archived losing candidate Version {mv.version}.")

        save_lifecycle_report(report)
        print("[SUCCESS] Model lifecycle automation completed.")
        return True

    except Exception as exc:
        print(f"[ERROR] Lifecycle automation failed: {exc}")
        report["status"] = "failed"
        report["error"] = str(exc)
        save_lifecycle_report(report)
        return False


if __name__ == "__main__":
    passed = automate_model_lifecycle()
    sys.exit(0 if passed else 1)
