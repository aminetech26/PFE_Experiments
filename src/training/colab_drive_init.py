from __future__ import annotations

import argparse
from pathlib import Path

BASE_DIR = Path("/content/drive/MyDrive/PV-FDD/optuna")


def ensure_colab_tracking_dirs() -> dict[str, Path]:
    targets = {
        "db": BASE_DIR / "db",
        "artifacts": BASE_DIR / "artifacts",
        "ml_models": BASE_DIR / "artifacts" / "ml_models",
        "dl_models": BASE_DIR / "artifacts" / "dl_models",
        "metrics": BASE_DIR / "artifacts" / "metrics",
        "plots": BASE_DIR / "artifacts" / "plots",
        "logs": BASE_DIR / "artifacts" / "logs",
        "checkpoints": BASE_DIR / "artifacts" / "checkpoints",
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


if __name__ == "__main__":
    main()
