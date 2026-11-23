"""FastAPI service exposing the churn model prediction."""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_pipeline import MODEL_PATH, load_model, predict

app = FastAPI(title="Churn prediction API")

try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH.name}")
except FileNotFoundError:
    model = None


class ChurnFeatures(BaseModel):
    """Input features of one customer."""

    CreditScore: int
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Female", "Male"]
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get("/")
def root() -> dict[str, str]:
    """Health check."""
    return {"status": "ok", "docs": "/docs"}


@app.post("/predict")
def predict_endpoint(features: ChurnFeatures) -> dict[str, int]:
    """Return the churn prediction (1 = customer leaves, 0 = stays)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Train it first.")
    try:
        return {"prediction": predict(model, features.model_dump())}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc