"""Tests for the FastAPI service."""

import pytest
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)

SAMPLE = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Female",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000,
}


def test_root_ok():
    assert client.get("/").status_code == 200


@pytest.mark.skipif(app_module.model is None, reason="model.joblib not trained yet")
def test_predict_ok():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    assert response.json()["prediction"] in (0, 1)


def test_predict_invalid_input():
    bad = dict(SAMPLE, Geography="Mars")
    assert client.post("/predict", json=bad).status_code == 422