"""Register and serve the workshop Prefect deployments."""

from prefect import serve

from pipeline_prefect import (
    all_flow,
    api_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    train_flow,
)


def main() -> None:
    """Create six named deployments and serve them."""
    deployments = [
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        evaluate_flow.to_deployment(name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
        code_flow.to_deployment(name="ml-pipeline-code", tags=["quality", "mlops"]),
        api_flow.to_deployment(name="ml-pipeline-api", tags=["api", "mlops"]),
    ]
    serve(*deployments)


if __name__ == "__main__":
    main()