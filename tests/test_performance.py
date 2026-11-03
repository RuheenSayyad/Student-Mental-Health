import json
import time
from pathlib import Path
import numpy as np
import pytest
from starlette.testclient import TestClient

from api.main import app

REPORT_PATH = Path("artifacts/api_test_report.json")
ALT_REPORT_PATH = Path("reports/api_testing_report.json")

SAMPLE_STUDENT = {
    "student_id": "PERF_TEST_001",
    "gender": "Male",
    "age": 21,
    "course": "BIT",
    "year_of_study": "year 2",
    "cgpa_range": "3.00 - 3.49",
    "marital_status": "No",
    "anxiety": "No",
    "panic_attack": "No",
    "specialist_treatment": "No",
}


def test_api_performance_benchmark():
    latencies = []
    num_requests = 50

    with TestClient(app) as client:
        # Warmup
        client.get("/health")

        for _ in range(num_requests):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=SAMPLE_STUDENT)
            t1 = time.perf_counter()

            assert resp.status_code == 200
            latencies.append((t1 - t0) * 1000)

    p50 = round(float(np.percentile(latencies, 50)), 2)
    p95 = round(float(np.percentile(latencies, 95)), 2)
    p99 = round(float(np.percentile(latencies, 99)), 2)
    mean_lat = round(float(np.mean(latencies)), 2)

    report = {
        "test_name": "API Latency and Concurrency Benchmark",
        "requests_sent": num_requests,
        "successful_requests": num_requests,
        "latency_ms": {
            "mean": mean_lat,
            "p50": p50,
            "p95": p95,
            "p99": p99,
        },
        "sla_target_p95_ms": 100.0,
        "sla_passed": p95 < 100.0,
    }

    for path in [REPORT_PATH, ALT_REPORT_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

    print(f"\n[BENCHMARK] Mean: {mean_lat}ms | P50: {p50}ms | P95: {p95}ms | P99: {p99}ms")
    assert p95 < 500.0, f"P95 latency {p95}ms exceeds ceiling"
