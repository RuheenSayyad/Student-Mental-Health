# Lab 11 Complete Flow: Containerization and Kubernetes Orchestration

## Overview
This document outlines the complete workflow for deploying the Student Mental Health Prediction API using Docker containers and Kubernetes clusters.

## Step 1: Prepare Production Deployment Artifacts
```bash
python src/mlops_pipeline/11_prepare_deployment.py
```
This loads the current `Production` model from MLflow, packages it along with the fitted preprocessing pipeline and metadata into `deployment/`.

## Step 2: Build the Container Image
```bash
docker build -t student-mental-health-api:1.0 .
```

## Step 3: Run and Test Locally
```bash
docker run -d -p 8000:8000 --name smh-api student-mental-health-api:1.0
curl http://localhost:8000/health
```

## Step 4: Deploy to Kubernetes Cluster
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/student-mental-health-api
```

## Step 5: Execute Integration Tests
```bash
pytest tests/test_api.py
pytest tests/test_local_vs_container.py
pytest tests/test_performance.py
```
