"""Reusable data preparation, training, persistence, and evaluation functions."""

from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump, load
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def prepare_data(
    test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Iris and return reproducible stratified train/test partitions."""
    dataset = load_iris()
    return train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Fit a small deterministic random-forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model.fit(X_train, y_train)


def save_model(model: Any, path: str | Path = "model.joblib") -> None:
    """Serialize a fitted model to the specified path."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)


def load_model(path: str | Path = "model.joblib") -> Any:
    """Load a serialized model from disk."""
    return load(Path(path))


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Return classification accuracy for a held-out dataset."""
    predictions = model.predict(X_test)
    return {"accuracy": float(accuracy_score(y_test, predictions))}
