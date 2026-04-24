"""Shared MLflow initialization for active thesis tasks."""

from __future__ import annotations

from src.modeling.common.experiment_tracker import setup_mlflow

EXPERIMENTS = {
    "anomaly": "Task_A_Anomaly",
    "classification": "Task_B_Classification",
}


def init_tracking(task: str | None = None) -> str:
    """Initialize MLflow -> DagsHub and set the active experiment."""
    if task is None:
        task = "classification"

    experiment_name = EXPERIMENTS[task]
    setup_mlflow(experiment_name)
    return experiment_name
