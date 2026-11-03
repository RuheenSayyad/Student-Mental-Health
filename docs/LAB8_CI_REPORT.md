# Lab 8: Continuous Integration (CI) & Quality Gates

## Overview
Lab 8 automates pipeline verification on every Git push and pull request using GitHub Actions.

## Quality Gate Implementation
- Script: `src/mlops_pipeline/07_quality_gate.py`
- Metric Threshold: Recall >= 0.60
- Outcome: Prevents sub-par models from progressing to the model lifecycle promotion stage.
- Evidence Report: `reports/quality_gate_report.json`

## CI Pipeline Workflow
- Workflow File: `.github/workflows/lab8_ci.yml`
- Triggers: Code push, Pull Request, Workflow Dispatch
- Actions: Environment setup, dependency installation, end-to-end pipeline execution, and artifact archiving.
