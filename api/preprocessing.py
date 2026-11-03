import pandas as pd

CGPA_MIDPOINTS = {
    "0 - 1.99": 1.00,
    "2.00 - 2.49": 2.25,
    "2.50 - 2.99": 2.75,
    "3.00 - 3.49": 3.25,
    "3.50 - 4.00": 3.75,
}

NUMERIC_FEATURES = [
    "age", "year_of_study", "cgpa_numeric",
    "symptom_burden_score", "senior_high_achiever",
]

CATEGORICAL_FEATURES = [
    "gender", "marital_status", "anxiety", "panic_attack",
    "specialist_treatment", "faculty_group",
]


def categorize_course(course_name: str) -> str:
    c = str(course_name).lower().strip()
    tokens = set(c.replace("-", " ").split())
    if c.startswith("engin") or c == "koe":
        return "Engineering"
    if tokens & {"bcs", "bit", "it", "cts"} or "computer" in c:
        return "Computing_IT"
    if any(k in c for k in ["islam", "pendidikan", "fiqh", "irkhs", "kirkhs", "usuluddin", "taasl"]):
        return "Islamic_Studies"
    if any(k in c for k in ["econ", "accounting", "bank", "kenms", "enm", "human resource", "business"]):
        return "Business_Economics"
    if any(k in c for k in ["bio", "medic", "nurs", "radio", "marine", "mhsc"]):
        return "Health_Sciences"
    if c in {"law", "laws"}:
        return "Law"
    if "psych" in c:
        return "Psychology"
    return "Arts_Humanities"


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw student survey inputs and engineer model features.
    """
    d = df.copy()

    # Normalize column names if raw survey headers are passed
    col_mapping = {
        "Choose your gender": "gender",
        "Age": "age",
        "What is your course?": "course",
        "Your current year of Study": "year_of_study",
        "What is your CGPA?": "cgpa_range",
        "Marital status": "marital_status",
        "Do you have Depression?": "depression",
        "Do you have Anxiety?": "anxiety",
        "Do you have Panic attack?": "panic_attack",
        "Did you seek any specialist for a treatment?": "specialist_treatment",
    }
    for old_c, new_c in col_mapping.items():
        if old_c in d.columns:
            d[new_c] = d[old_c]

    # Remove non-features
    for drop_col in ["student_id", "Timestamp", "timestamp", "depression", "Predicted_Depression", "Depression_Probability"]:
        if drop_col in d.columns:
            d = d.drop(columns=[drop_col])

    # Standardize string fields
    for col in ["gender", "course", "year_of_study", "cgpa_range", "marital_status", "anxiety", "panic_attack", "specialist_treatment", "faculty_group"]:
        if col in d.columns:
            d[col] = d[col].astype(str).str.strip()

    # Engineer features
    if "age" in d.columns:
        d["age"] = pd.to_numeric(d["age"], errors="coerce")
    else:
        d["age"] = 21.0

    # Extract year number (e.g. 'year 2' -> 2)
    if "year_of_study" in d.columns:
        extracted_year = d["year_of_study"].astype(str).str.extract(r"(\d+)")
        d["year_of_study"] = pd.to_numeric(extracted_year[0], errors="coerce").fillna(1).astype(float)
    else:
        d["year_of_study"] = 1.0

    # Map CGPA range to midpoint if present
    if "cgpa_range" in d.columns:
        d["cgpa_numeric"] = d["cgpa_range"].map(CGPA_MIDPOINTS)
    if "cgpa_numeric" not in d.columns:
        d["cgpa_numeric"] = 3.25
    d["cgpa_numeric"] = pd.to_numeric(d["cgpa_numeric"], errors="coerce").fillna(3.25)

    # Symptom burden
    if "symptom_burden_score" not in d.columns:
        anxiety_val = (d["anxiety"].astype(str).str.lower() == "yes").astype(int) if "anxiety" in d.columns else 0
        panic_val = (d["panic_attack"].astype(str).str.lower() == "yes").astype(int) if "panic_attack" in d.columns else 0
        d["symptom_burden_score"] = anxiety_val + panic_val
    else:
        d["symptom_burden_score"] = pd.to_numeric(d["symptom_burden_score"], errors="coerce").fillna(0).astype(int)

    # Senior high achiever
    if "senior_high_achiever" not in d.columns:
        d["senior_high_achiever"] = ((d["year_of_study"] >= 3) & (d["cgpa_numeric"] >= 3.5)).astype(int)
    else:
        d["senior_high_achiever"] = pd.to_numeric(d["senior_high_achiever"], errors="coerce").fillna(0).astype(int)

    # Course faculty clustering
    if "faculty_group" not in d.columns:
        d["faculty_group"] = d["course"].apply(categorize_course) if "course" in d.columns else "Arts_Humanities"

    ordered_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for c in ordered_cols:
        if c not in d.columns:
            d[c] = "No" if c in CATEGORICAL_FEATURES else 0

    return d[ordered_cols]
