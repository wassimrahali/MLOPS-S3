"""Prefect tasks and flows for the workshop machine-learning pipeline."""

import argparse
import os
import subprocess  # nosec B404 - subprocess is required for the project tool commands.
import sys
from pathlib import Path
from typing import Sequence

from prefect import flow, task

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.joblib"
CLONE_PATH = PROJECT_DIR / "project_clone"
REPO_URL = os.environ.get("REPO_URL")
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run(cmd: Sequence[str], cwd: Path = PROJECT_DIR) -> None:
    """Run a command and fail immediately if it exits unsuccessfully."""
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(list(cmd), check=True, cwd=cwd)  # nosec B603 - args are passed without a shell.


@task(name="fetch-project")
def fetch_project(enabled: bool = False) -> None:
    """Optionally clone REPO_URL or update the existing project_clone checkout."""
    if not enabled:
        print("Remote project fetch disabled.")
        return
    if not REPO_URL:
        raise ValueError("Set the REPO_URL environment variable to enable project fetching.")

    if CLONE_PATH.exists():
        run(["git", "pull"], cwd=CLONE_PATH)
    else:
        run(["git", "clone", REPO_URL, str(CLONE_PATH)])


@task(name="install-dependencies")
def install_dependencies() -> None:
    """Install project dependencies using the active Python interpreter."""
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="format-code")
def format_code() -> None:
    """Format only the project source files."""
    run([sys.executable, "-m", "black", "--line-length=100", *FILES])


@task(name="code-quality")
def code_quality() -> None:
    """Lint only the project source files."""
    run([sys.executable, "-m", "flake8", "--max-line-length=100", *FILES])


@task(name="code-security")
def code_security() -> None:
    """Run Bandit against only the project source files."""
    run([sys.executable, "-m", "bandit", "-q", *FILES])


@task(name="unit-tests")
def unit_tests() -> None:
    """Run the project test suite."""
    run([sys.executable, "-m", "pytest", "tests/"])


@task(name="prepare-data")
def prepare_data_task():
    """Load and split the model dataset."""
    from model_pipeline import prepare_data

    return prepare_data()


@task(name="train-model")
def train_model_task(data):
    """Fit the classifier on the training partition."""
    from model_pipeline import train_model

    X_train, _, y_train, _ = data
    return train_model(X_train, y_train)


@task(name="save-model")
def save_model_task(model) -> str:
    """Persist the fitted classifier."""
    from model_pipeline import save_model

    save_model(model, MODEL_PATH)
    return str(MODEL_PATH)


@task(name="load-model")
def load_model_task():
    """Load the persisted classifier."""
    from model_pipeline import load_model

    return load_model(MODEL_PATH)


@task(name="evaluate-model")
def evaluate_model_task(model) -> dict[str, float]:
    """Evaluate the classifier against the held-out test partition."""
    from model_pipeline import evaluate_model, prepare_data

    _, X_test, _, y_test = prepare_data()
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Evaluation metrics: {metrics}")
    return metrics


@flow(name="install")
def install_flow() -> None:
    """Install project dependencies."""
    install_dependencies()


@flow(name="code")
def code_flow(fetch_remote: bool = False) -> None:
    """Fetch (optionally), install, format, lint, scan, and test the project."""
    fetch_project(fetch_remote)
    install_dependencies()
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train")
def train_flow():
    """Prepare data, train a model, and save it."""
    data = prepare_data_task()
    model = train_model_task(data)
    return save_model_task(model)


@flow(name="evaluate")
def evaluate_flow() -> dict[str, float]:
    """Load the saved model and evaluate it."""
    model = load_model_task()
    return evaluate_model_task(model)


@flow(name="all")
def all_flow(fetch_remote: bool = False) -> dict[str, float]:
    """Run code checks first, then train and evaluate the model."""
    code_flow(fetch_remote=fetch_remote)
    data = prepare_data_task()
    model = train_model_task(data)
    save_model_task(model)
    return evaluate_model_task(model)


def main() -> None:
    """Parse CLI options and run the requested flow."""
    parser = argparse.ArgumentParser(description="Run the Prefect MLOps workshop pipeline.")
    parser.add_argument(
        "--flow",
        required=True,
        choices=["all", "train", "entrainement", "evaluate", "code", "install"],
        help="Flow to execute (entrainement is an alias for train).",
    )
    parser.add_argument(
        "--fetch-project",
        action="store_true",
        help="Enable cloning/updating the repository configured in REPO_URL.",
    )
    args = parser.parse_args()

    if args.flow == "all":
        all_flow(fetch_remote=args.fetch_project)
    elif args.flow == "train" or args.flow == "entrainement":
        train_flow()
    elif args.flow == "evaluate":
        evaluate_flow()
    elif args.flow == "code":
        code_flow(fetch_remote=args.fetch_project)
    else:
        install_flow()


if __name__ == "__main__":
    main()
