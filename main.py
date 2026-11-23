"""Command-line entry point for training and evaluating the churn model."""

from model_pipeline import evaluate_model, prepare_data, save_model, train_model


def main() -> dict[str, float]:
    """Train the churn classifier, save it, and report test metrics."""
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_model(X_train, y_train)
    save_model(model)
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Evaluation metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    main()
