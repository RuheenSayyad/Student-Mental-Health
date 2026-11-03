import pytest
import pandas as pd
from starlette.testclient import TestClient

from api.main import app
from api.model_loader import ModelLoader
from api.preprocessing import clean_prediction_data

SAMPLE_STUDENT = {
    "student_id": "TEST_PARITY_001",
    "gender": "Female",
    "age": 21,
    "course": "Engineering",
    "year_of_study": "year 2",
    "cgpa_range": "3.50 - 4.00",
    "marital_status": "No",
    "anxiety": "No",
    "panic_attack": "No",
    "specialist_treatment": "No",
}


def test_parity_local_vs_api():
    # 1. Local direct prediction
    loader = ModelLoader()
    loader.load()

    raw_df = pd.DataFrame([SAMPLE_STUDENT])
    clean_df = clean_prediction_data(raw_df)
    transformed = loader.preprocessor.transform(clean_df)

    local_pred = int(loader.model.predict(transformed)[0])
    local_prob = float(loader.model.predict_proba(transformed)[0, 1])

    # 2. Prediction through API
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE_STUDENT)
        assert response.status_code == 200
        api_data = response.json()

    # 3. Assert full mathematical parity
    assert local_pred == api_data["prediction"]
    assert round(local_prob, 4) == api_data["depression_probability"]
    print(f"\n[PARITY PASSED] Local pred: {local_pred} == API pred: {api_data['prediction']}")
