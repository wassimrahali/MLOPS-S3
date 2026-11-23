"""Unit tests for the churn data preparation, model and prediction."""

from model_pipeline import evaluate_model, predict, prepare_data, train_model

SAMPLE = {
    "CreditScore": 600,
    "Geography": "France",
    "Gender": "Female",
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0,
}


def test_prepare_data_shapes():
    X_train, X_test, y_train, y_test = prepare_data()
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert "Exited" not in X_train.columns
    assert set(y_train.unique()) <= {0, 1}


def test_model_beats_trivial_baseline():
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["accuracy"] > 0.80


def test_predict_returns_binary_label():
    X_train, _, y_train, _ = prepare_data()
    model = train_model(X_train, y_train)
    assert predict(model, SAMPLE) in (0, 1)