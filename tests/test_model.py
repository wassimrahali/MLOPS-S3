"""Fast unit tests for the sample model pipeline."""

import numpy as np

from model_pipeline import prepare_data, train_model


def test_prepare_data_returns_valid_split() -> None:
    """The data split should have matching rows and all three Iris classes."""
    X_train, X_test, y_train, y_test = prepare_data()

    assert X_train.shape == (120, 4)
    assert X_test.shape == (30, 4)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert set(np.unique(y_train)) == {0, 1, 2}


def test_train_model_fits_classifier() -> None:
    """Training should produce a classifier that predicts each test row."""
    X_train, X_test, y_train, _ = prepare_data()
    model = train_model(X_train, y_train)

    assert model.predict(X_test).shape == (30,)
