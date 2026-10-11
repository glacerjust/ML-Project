import pytest
from fastapi.testclient import TestClient

from src.serving.main import app


@pytest.fixture(scope="module")
def client():
    # Use context manager so FastAPI's lifespan startup hook executes
    with TestClient(app) as test_client:
        yield test_client


def test_dashboard_endpoint(client: TestClient):
    """Verify that root dashboard returns 200 and renders HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Telco Churn AI Platform" in response.text


def test_health_endpoint(client: TestClient):
    """Verify that the health check returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_metrics_endpoint(client: TestClient):
    """Verify that operational metrics endpoint reports counters and distributions."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "request_count" in data
    assert "error_count" in data
    assert "avg_latency_ms" in data
    assert "p95_latency_ms" in data
    assert "predictions_distribution" in data


def test_predict_endpoint_valid_payload(client: TestClient):
    """Verify that a valid customer payload receives a structured prediction."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 70.35,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert "threshold" in data
    assert "model_version" in data


def test_predict_endpoint_invalid_payload_returns_422(client: TestClient):
    """Verify that malformed data is rejected by Pydantic schema validation."""
    malformed_payload = {
        "gender": "Female",
        "MonthlyCharges": "invalid_string_instead_of_float",
    }
    response = client.post("/predict", json=malformed_payload)
    assert response.status_code == 422
