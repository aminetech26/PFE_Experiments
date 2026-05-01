from __future__ import annotations

import argparse
from pathlib import Path

BASE_DIR = Path("/content/drive/MyDrive/PV-FDD/optuna")
ARTIFACTS_ROOT = BASE_DIR / "artifacts"


def ensure_colab_tracking_dirs() -> dict[str, Path]:
    targets = {
        "db": BASE_DIR / "db",
        "artifacts": ARTIFACTS_ROOT,
        "ml_models": ARTIFACTS_ROOT / "ml_models",
        "dl_models": ARTIFACTS_ROOT / "dl_models",
        "metrics": ARTIFACTS_ROOT / "metrics",
        "plots": ARTIFACTS_ROOT / "plots",
        "logs": ARTIFACTS_ROOT / "logs",
        "checkpoints": ARTIFACTS_ROOT / "checkpoints",
        "anomaly_ocsvm": ARTIFACTS_ROOT / "anomaly" / "one_class_svm",
        "anomaly_iforest": ARTIFACTS_ROOT / "anomaly" / "isolation_forest",
    }
    for path in targets.values():
        path.mkdir(parents=True, exist_ok=True)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Colab Drive tracking folders")
    parser.add_argument("--init", action="store_true", help="Create required Drive folders")
    args = parser.parse_args()

    if not args.init:
        raise SystemExit("Use --init to create required Drive folders.")

    ensure_colab_tracking_dirs()
    print(f"Initialized tracking folders under: {BASE_DIR}")
    print("\nAdd to your notebook to redirect all writes to Drive:")
    print(f"  import os; os.environ['PFE_ARTIFACTS_ROOT'] = '{ARTIFACTS_ROOT}'")


if __name__ == "__main__":
    main()
