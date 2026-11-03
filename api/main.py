import time
import uuid
from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_loader import ModelLoader
from api.monitoring_logger import log_inference
from api.preprocessing import clean_prediction_data
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    StudentInput,
)

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Starting Student Mental Health API...")
    try:
        model_loader.load()
    except Exception as exc:
        print(f"[WARN] Startup model loading: {exc}")
    yield
    print("[INFO] Student Mental Health API shutting down...")


app = FastAPI(
    title="Student Mental Health Prediction API",
    description="Lab 10 & 11 - ML Model Serving and Production API Engineering",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = uuid.uuid4().hex[:12]
    errors = []

    for error in exc.errors():
        locations = [str(loc) for loc in error["loc"] if loc != "body"]
        field = ".".join(locations)
        errors.append({
            "field": field,
            "message": error["msg"],
        })

    log_inference(
        input_data={},
        latency_ms=None,
        status="failure",
        error="Request validation failed",
        model_name=model_loader.model_name,
        model_version=model_loader.model_version,
        run_id=model_loader.run_id,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Request validation failed",
            "details": errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": "Prediction service error",
            "details": [{"field": "prediction", "message": str(exc.detail)}],
        },
    )


@app.get("/")
def root():
    return {
        "message": "Student Mental Health Prediction API is running",
        "lab": "Lab 10 & 11",
        "status": "success",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_ready": model_loader.is_ready(),
    }


@app.get("/metadata")
def metadata():
    return {
        "service": "Student Mental Health Prediction API",
        "version": "1.0.0",
        "model_name": model_loader.model_name,
        "model_version": model_loader.model_version,
        "model_uri": model_loader.model_uri,
        "run_id": model_loader.run_id,
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
    }


def compute_risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    return "Low"


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput):
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    start_time = time.perf_counter()
    input_dict = student.model_dump()

    try:
        raw_df = pd.DataFrame([input_dict])
        clean_df = clean_prediction_data(raw_df)
        X_trans = model_loader.preprocessor.transform(clean_df)

        pred = int(model_loader.model.predict(X_trans)[0])
        prob = float(model_loader.model.predict_proba(X_trans)[0, 1])
        risk = compute_risk_level(prob)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        log_inference(
            input_data=input_dict,
            prediction=pred,
            depression_probability=round(prob, 4),
            risk_level=risk,
            latency_ms=latency_ms,
            status="success",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
        )

        return PredictionResponse(
            prediction=pred,
            depression_probability=round(prob, 4),
            risk_level=risk,
        )

    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        log_inference(
            input_data=input_dict,
            latency_ms=latency_ms,
            status="failure",
            error=str(exc),
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
        )
        raise HTTPException(status_code=500, detail=f"Inference failure: {exc}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPredictionRequest):
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    results = []
    for item in batch.students:
        pred_res = predict(item)
        results.append(pred_res)

    return BatchPredictionResponse(
        count=len(results),
        predictions=results,
    )


@app.get("/logs")
def get_logs(limit: int = 10):
    from pathlib import Path
    log_path = Path("logs/inference/inference_log.jsonl")
    if not log_path.exists():
        return {"logs": [], "count": 0}

    lines = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    latest_lines = lines[-limit:]
    import json
    parsed = [json.loads(l) for l in latest_lines]
    return {"count": len(parsed), "logs": parsed}
