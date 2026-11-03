# Lab 9: Resilience Testing & Failure Injection

## Overview
Lab 9 verifies that the MLOps pipeline fails gracefully when confronted with corrupted inputs and automatically recovers when artifacts are compromised.

## Injected Scenarios & Outcomes
1. **Missing Data**:
   - Injection: Raw survey dataset removed.
   - Outcome: `02_validate_data.py` stops pipeline execution (HALT).
2. **Invalid Datatype**:
   - Injection: String value written into `Age` column.
   - Outcome: Pandera schema validator detects type violation and halts pipeline (HALT).
3. **Schema Mismatch**:
   - Injection: Critical column dropped from survey.
   - Outcome: Schema validator reports missing column (HALT).
4. **Preprocessing Inconsistency**:
   - Injection: `np.nan` injected into `X_train_final.npy`.
   - Outcome: `04_validate_outputs.py` detects NaN values and prevents training (HALT).
5. **Missing Model Artifact & Auto-Recovery**:
   - Injection: Local model artifact deleted from disk.
   - Outcome: System automatically retrieves Production model version from MLflow Registry and restores artifact (RECOVER).

Reports generated:
- `reports/failure_report.json`
- `artifacts/failure_simulation_report.json`
- `artifacts/corrupted_simulation_errors.csv`
