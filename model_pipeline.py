"""Reusable data preparation, training, persistence, and evaluation functions."""

from pathlib import Path
from typing import Any

import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "data" / "Churn_Modelling.csv"
TARGET = "Exited"
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
CATEGORICAL_COLUMNS = ["Geography", "Gender"]


def prepare_data(
    test_size: float = 0.2, random_state: int = 42, path: str | Path = DATA_PATH
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the churn dataset, encode it, and return stratified train/test partitions."""
    df = pd.read_csv(path)
    df = df.drop(columns=ID_COLUMNS)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Fit a deterministic random-forest classifier."""
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    return model.fit(X_train, y_train)


def save_model(model: Any, path: str | Path = "model.joblib") -> None:
    """Serialize a fitted model to the specified path."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)


def load_model(path: str | Path = "model.joblib") -> Any:
    """Load a serialized model from disk."""
    return load(Path(path))


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Return accuracy and F1 score (churn classes are imbalanced) on held-out data."""
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
    }
