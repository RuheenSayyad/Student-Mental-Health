import pytest
from starlette.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


VALID_STUDENT = {
    "student_id": "TEST_STU_001",
    "gender": "Female",
    "age": 20,
    "course": "Engineering",
    "year_of_study": "year 2",
    "cgpa_range": "3.50 - 4.00",
    "marital_status": "No",
    "anxiety": "No",
    "panic_attack": "No",
    "specialist_treatment": "No",
}

DEPRESSED_STUDENT = {
    "student_id": "TEST_STU_002",
    "gender": "Female",
    "age": 22,
    "course": "BCS",
    "year_of_study": "year 3",
    "cgpa_range": "2.50 - 2.99",
    "marital_status": "Yes",
    "anxiety": "Yes",
    "panic_attack": "Yes",
    "specialist_treatment": "Yes",
}


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_metadata_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "version" in data


def test_predict_single_student(client):
    response = client.post("/predict", json=VALID_STUDENT)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "depression_probability" in data
    assert 0.0 <= data["depression_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_depressed_student(client):
    response = client.post("/predict", json=DEPRESSED_STUDENT)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "risk_level" in data


def test_predict_invalid_gender(client):
    invalid = VALID_STUDENT.copy()
    invalid["gender"] = "UnknownGender"
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"


def test_predict_invalid_age(client):
    invalid = VALID_STUDENT.copy()
    invalid["age"] = 120  # Out of range (ge=15, le=65)
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_batch_prediction(client):
    payload = {"students": [VALID_STUDENT, DEPRESSED_STUDENT]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_logs_endpoint(client):
    response = client.get("/logs?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "logs" in data
    assert "count" in data
