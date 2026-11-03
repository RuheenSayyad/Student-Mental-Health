import json
from pathlib import Path
import pandas as pd

LOG_PATH = Path("logs/inference/inference_log.jsonl")
REPORT_PATH = Path("outputs/monitoring/prediction_monitoring_report.json")


def monitor_predictions():
    print("=" * 70)
    print("LIVE PREDICTION MONITORING (LAB 12)")
    print("=" * 70)

    predictions = []
    probabilities = []

    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get("prediction") is not None:
                            predictions.append(record["prediction"])
                        if record.get("depression_probability") is not None:
                            probabilities.append(record["depression_probability"])
                    except Exception:
                        pass

    count = len(predictions)
    dep_count = sum(1 for p in predictions if p == 1)
    dep_rate = round(dep_count / max(count, 1), 4)

    report = {
        "total_inferences_logged": count,
        "predicted_depression_count": dep_count,
        "predicted_no_depression_count": count - dep_count,
        "predicted_depression_rate": dep_rate,
        "alert_threshold": 0.65,
        "alert_triggered": dep_rate > 0.65,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(json.dumps(report, indent=4))
    print(f"Report saved to {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    monitor_predictions()
