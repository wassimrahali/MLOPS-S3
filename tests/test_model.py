"""Unit tests for the churn data preparation and model."""

from model_pipeline import prepare_data, train_model, evaluate_model


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