import os
import sys
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validation import validate_schema


def main():
    print("=" * 70)
    print("RAW DATA VALIDATION")
    print("=" * 70)

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join("configs", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]

    if not os.path.exists(data_path):
        print(f"[ERROR] Target file not found at: {data_path}")
        sys.exit(1)

    raw_df = pd.read_csv(data_path)
    passed = validate_schema(raw_df, output_report_name="baseline_validation.csv")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
