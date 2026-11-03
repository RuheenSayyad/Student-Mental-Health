# Lab 11: Containerization & Kubernetes Deployment Report

## Container Specifications
- Base Image: `python:3.11-slim`
- Server Engine: `uvicorn` / `FastAPI`
- Production Dependencies: Isolated via `requirements-api.txt`
- Build Size: Highly optimized (< 250MB)

## Kubernetes Orchestration
- Deployment Manifest: `k8s/deployment.yaml`
- Service Manifest: `k8s/service.yaml` (NodePort 8000)
- Replicas: 1 (horizontally scalable)

## Health & Parity Verification
- Tested via `tests/test_api.py` and `tests/test_local_vs_container.py`
- Mathematical parity between local preprocessor pipeline and containerized API verified with zero divergence.
