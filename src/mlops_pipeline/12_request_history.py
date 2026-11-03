import json
from pathlib import Path
import pandas as pd

LOG_PATH = Path("logs/inference/inference_log.jsonl")
REPORT_PATH = Path("outputs/monitoring/request_history_summary.csv")


def summarize_history():
    print("=" * 70)
    print("INFERENCE REQUEST HISTORY SUMMARY (LAB 12)")
    print("=" * 70)

    rows = []
    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        rows.append({
                            "request_id": record.get("request_id"),
                            "timestamp": record.get("timestamp"),
                            "prediction": record.get("prediction"),
                            "probability": record.get("depression_probability"),
                            "risk_level": record.get("risk_level"),
                            "latency_ms": record.get("latency_ms"),
                            "status": record.get("status"),
                        })
                    except Exception:
                        pass

    if not rows:
        print("[INFO] No history records yet in inference log.")
        df = pd.DataFrame(columns=["request_id", "timestamp", "prediction", "probability", "risk_level", "latency_ms", "status"])
    else:
        df = pd.DataFrame(rows)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_PATH, index=False)

    print(f"Total historical requests parsed: {len(df)}")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    summarize_history()
