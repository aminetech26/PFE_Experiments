from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import dagshub
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from dotenv import load_dotenv
from loguru import logger


def setup_mlflow(experiment_name: str) -> None:
    load_dotenv()

    username = os.environ.get("DAGSHUB_USERNAME")
    repo = os.environ.get("DAGSHUB_REPO")
    token = os.environ.get("DAGSHUB_USER_TOKEN")

    if not all([username, repo, token]):
        raise OSError(
            "Missing DagsHub credentials. "
            "Set DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_USER_TOKEN "
            "in .env (local) or Colab Secrets."
        )

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    dagshub.init(repo_owner=username, repo_name=repo, mlflow=True)

    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow -> DagsHub | experiment: '{experiment_name}'")


def log_experiment(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    model=None,
    artifacts: list[str] | None = None,
) -> str:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if tags:
            mlflow.set_tags(tags)

        if model is not None:
            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception:
                try:
                    mlflow.pytorch.log_model(model, "model")
                except Exception:
                    logger.warning("Could not auto-log model - save manually if needed")

        if artifacts:
            for path in artifacts:
                if Path(path).exists():
                    mlflow.log_artifact(path)

        run_id = mlflow.active_run().info.run_id
        logger.success(f"Run logged: {run_name} [{run_id}]")
        return run_id
