# Git Workflow & Release Management

## Branching Strategy
- `main` / `master`: Production-ready releases. Protected branch.
- `feature/*`: Development of individual labs, stages, and enhancements.
- `hotfix/*`: Urgent fixes to production serving or pipeline logic.

## GitHub Actions Triggers
- Any push or pull request to `main` touching `src/mlops_pipeline/**` or `config.yaml` automatically triggers the **Lab 8 CI** workflow.
- Automated hourly or scheduled cron jobs run the **Lab 13 Monitoring Trigger** workflow.

## Version Tagging
- Tags follow semantic versioning:
  - `v1.0.0`: Initial baseline and Lab 7 pipeline
  - `v1.1.0`: Lab 8 CI & Lab 9 Failure handling
  - `v1.2.0`: Lab 10 serving & Lab 11 containerization
  - `v1.3.0`: Lab 12 drift detection & Lab 13 continuous retraining loop
