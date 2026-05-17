from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_experiments_root() -> Path:
    """Return the root for all experiment outputs.

    On Colab, set PFE_ARTIFACTS_ROOT=/content/drive/MyDrive/PV-FDD/optuna/artifacts
    to redirect all writes to Google Drive. Unset locally — falls back to
    PROJECT_ROOT/experiments.
    """
    env = os.environ.get("PFE_ARTIFACTS_ROOT")
    if env:
        return Path(env)
    return PROJECT_ROOT / "experiments"
