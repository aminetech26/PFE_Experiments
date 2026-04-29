from __future__ import annotations

import os
from pathlib import Path


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise OSError(f"Missing required env var: {name}")
    return value


def main() -> None:
    base_dir = Path("/content/drive/MyDrive/PV-FDD/optuna")

    _require_env("DAGSHUB_USERNAME")
    _require_env("DAGSHUB_REPO")
    _require_env("DAGSHUB_USER_TOKEN")

    targets = {
        "db": base_dir / "db",
        "artifacts": base_dir / "artifacts",
        "ml_models": base_dir / "artifacts" / "ml_models",
        "dl_models": base_dir / "artifacts" / "dl_models",
        "metrics": base_dir / "artifacts" / "metrics",
        "plots": base_dir / "artifacts" / "plots",
        "logs": base_dir / "artifacts" / "logs",
        "checkpoints": base_dir / "artifacts" / "checkpoints",
    }

    for path in targets.values():
        path.mkdir(parents=True, exist_ok=True)

    ml_db = targets["db"] / "optuna_ml.db"
    dl_db = targets["db"] / "optuna_dl.db"

    print("Tracking bootstrap complete.")
    print(f"Base directory: {base_dir}")
    print("Use these model_config.yaml values:")
    print(
        "classification.ml.hpo.storage_url: "
        f'"sqlite:////{ml_db.as_posix().lstrip("/")}"'
    )
    print(
        "classification.dl.hpo.storage_url: "
        f'"sqlite:////{dl_db.as_posix().lstrip("/")}"'
    )


if __name__ == "__main__":
    main()
