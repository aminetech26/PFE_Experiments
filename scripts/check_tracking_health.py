from __future__ import annotations

import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import optuna


def _init_tracking(task: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.mlflow_setup import init_tracking

    init_tracking(task)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise OSError(f"Missing required env var: {name}")
    return value


def _check_mlflow(task: str) -> str:
    _init_tracking(task)
    run_name = f"tracking_health_{task}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("health_check", "true")
        mlflow.log_param("check", "mlflow_dagshub")
        mlflow.log_param("health_task", task)
        mlflow.log_metric("health_metric", 1.0)
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else ""
    return run_id


def _check_optuna(storage_url: str | None) -> str:
    if storage_url:
        study = optuna.create_study(
            study_name="tracking_health_optuna",
            direction="maximize",
            storage=storage_url,
            load_if_exists=True,
        )
    else:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name.replace("\\", "/")
            sqlite_url = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="tracking_health_optuna",
            direction="maximize",
            storage=sqlite_url,
            load_if_exists=True,
        )

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        return 1.0 - abs(x - 0.5)

    study.optimize(objective, n_trials=1)
    return study.study_name


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    _require_env("DAGSHUB_USERNAME")
    _require_env("DAGSHUB_REPO")
    _require_env("DAGSHUB_USER_TOKEN")

    storage_url = os.environ.get("OPTUNA_STORAGE_URL", "").strip() or None

    cls_run_id = _check_mlflow("classification")
    anom_run_id = _check_mlflow("anomaly")
    study_name = _check_optuna(storage_url)

    print("Tracking health check passed.")
    print(f"MLflow classification run_id: {cls_run_id}")
    print(f"MLflow anomaly run_id: {anom_run_id}")
    print(f"Optuna study: {study_name}")
    if storage_url:
        print(f"Optuna storage: {storage_url}")
    else:
        print("Optuna storage: temporary sqlite (set OPTUNA_STORAGE_URL to test persistent storage)")


if __name__ == "__main__":
    main()
