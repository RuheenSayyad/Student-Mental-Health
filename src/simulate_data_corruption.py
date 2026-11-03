import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.mlops_pipeline.validation import validate_schema


def simulate():
    print("=" * 70)
    print("SIMULATE DATA CORRUPTION")
    print("=" * 70)

    raw_path = "data/raw/Student_Mental_Health.csv"
    if not os.path.exists(raw_path):
        print(f"[ERROR] {raw_path} not found.")
        return

    df = pd.read_csv(raw_path)
    corrupted_df = df.copy()

    # Inject corruptions:
    # 1. Invalid gender
    corrupted_df.loc[0, "Choose your gender"] = "Alien"
    # 2. Out-of-range Age
    corrupted_df.loc[1, "Age"] = 99.0
    # 3. Invalid CGPA string format
    corrupted_df.loc[2, "What is your CGPA?"] = "9.99 - 10.00"
    # 4. Invalid Year format
    corrupted_df.loc[3, "Your current year of Study"] = "Semester 8"

    print("[INFO] Injected 4 distinct data anomalies into survey sample.")
    print("[INFO] Running schema validation...")

    is_clean = validate_schema(corrupted_df, output_report_name="corrupted_simulation_errors.csv")

    if not is_clean:
        print("[SUCCESS] Corruptions were successfully intercepted by Pandera!")
        print("[INFO] Detailed errors written to artifacts/corrupted_simulation_errors.csv")
    else:
        print("[FAIL] Corruptions were not caught!")


if __name__ == "__main__":
    simulate()
