import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

RAW_PATH = "data/raw/Student_Mental_Health.csv"

YES_NO = ["Yes", "No"]

# Raw CGPA cells occasionally carry a trailing space (e.g. '3.50 - 4.00 '), so the
# check tolerates surrounding whitespace; the cleaning step strips it.
CGPA_REGEX = r"^\s*(0 - 1\.99|2\.00 - 2\.49|2\.50 - 2\.99|3\.00 - 3\.49|3\.50 - 4\.00)\s*$"
YEAR_REGEX = r"(?i)^\s*year\s*[1-4]\s*$"


def get_student_schema():
    """Defines the strict schema specification for the Student Mental Health survey."""
    return pa.DataFrameSchema({
        "Timestamp": Column(pa.String),
        "Choose your gender": Column(pa.String, Check.isin(["Male", "Female"])),
        "Age": Column(pa.Float, Check.in_range(16, 40), nullable=True),
        "What is your course?": Column(pa.String),
        "Your current year of Study": Column(pa.String, Check.str_matches(YEAR_REGEX)),
        "What is your CGPA?": Column(pa.String, Check.str_matches(CGPA_REGEX)),
        "Marital status": Column(pa.String, Check.isin(YES_NO)),
        "Do you have Depression?": Column(pa.String, Check.isin(YES_NO)),
        "Do you have Anxiety?": Column(pa.String, Check.isin(YES_NO)),
        "Do you have Panic attack?": Column(pa.String, Check.isin(YES_NO)),
        "Did you seek any specialist for a treatment?": Column(pa.String, Check.isin(YES_NO)),
    }, strict=True)


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    """Validates the input DataFrame against the defined schema."""
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    schema = get_student_schema()

    try:
        schema.validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED. Dataset is clean.")
        return True

    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED. Corruptions detected.")

        failures = err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]
        print(failures.to_string())

        os.makedirs("artifacts", exist_ok=True)
        report_path = os.path.join("artifacts", output_report_name)
        failures.to_csv(report_path, index=False)
        print(f"[INFO] Detailed failure report saved to '{report_path}'.")
        return False


if __name__ == "__main__":
    # When run directly, it only validates the raw survey data
    if os.path.exists(RAW_PATH):
        raw_df = pd.read_csv(RAW_PATH)
        ok = validate_schema(raw_df, output_report_name="baseline_validation.csv")
        raise SystemExit(0 if ok else 1)
    else:
        print(f"[ERROR] Target file not found at: {RAW_PATH}")
        raise SystemExit(1)
