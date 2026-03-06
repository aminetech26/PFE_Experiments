"""
Shared MLflow initialisation.
Call once at the top of any script or notebook before logging anything.

Usage (local or Colab):
    from src.mlflow_setup import init_tracking
    init_tracking()  # sets up all 3 experiments; returns the experiment names
"""
from __future__ import annotations

from src.training.experiment_tracker import setup_mlflow

EXPERIMENTS = {
    "anomaly": "Task_A_Anomaly",
    "classification": "Task_B_Classification",
    "forecasting": "Task_C_Forecasting",
}


def init_tracking(task: str | None = None) -> str:
    """
    Initialise MLflow → DagsHub and set the active experiment.

    Args:
        task: one of 'anomaly', 'classification', 'forecasting'.
              If None, defaults to 'classification'.

    Returns:
        The MLflow experiment name that was set.
    """
    if task is None:
        task = "classification"

    experiment_name = EXPERIMENTS[task]
    setup_mlflow(experiment_name)
    return experiment_name
