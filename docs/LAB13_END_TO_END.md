# Lab 13: Continuous Monitoring & Automated Retraining Loop

## The Closed MLOps Loop
Lab 13 bridges real-time monitoring and production model evolution:
1. **Continuous Telemetry**: Inference requests to the FastAPI service log inputs, predictions, probabilities, and latencies to `logs/inference/inference_log.jsonl`.
2. **Drift Detection**: Periodic evaluation calculates:
   - Data Drift (KS-test on numeric, JS-divergence on categorical)
   - Prediction Drift (KS-test on output distributions)
   - Concept Drift (Performance metric degradation against ground truth)
3. **Automated Trigger**: `12_monitor_trigger.py` detects when drift crosses threshold and new student cohorts arrive.
4. **Autonomous Retraining**: Automatically launches pipeline, trains Challenger V2, evaluates against Champion, and promotes V2 to Production.
5. **Continuous Deployment**: Triggers container rollout in Kubernetes without service interruption.
