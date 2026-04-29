from __future__ import annotations

import argparse
import os
from pathlib import Path

BASE_DIR = Path("/content/drive/MyDrive/PV-FDD/optuna")
ML_DB = BASE_DIR / "db" / "optuna_ml.db"
DL_DB = BASE_DIR / "db" / "optuna_dl.db"


def _required_env() -> list[str]:
    return ["DAGSHUB_USERNAME", "DAGSHUB_REPO", "DAGSHUB_USER_TOKEN"]


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


def env_status() -> dict[str, bool]:
    return {name: bool(os.environ.get(name)) for name in _required_env()}


def sqlite_urls() -> dict[str, str]:
    ml_url = f"sqlite:////{ML_DB.as_posix().lstrip('/')}"
    dl_url = f"sqlite:////{DL_DB.as_posix().lstrip('/')}"
    return {"ml": ml_url, "dl": dl_url}


def print_config_snippet() -> None:
    urls = sqlite_urls()
    print("Paste into configs/model_config.yaml:")
    print("anomaly_detection:")
    print("  ml:")
    print("    hpo:")
    print(f"      storage_url: \"{urls['ml']}\"")
    print("  dl:")
    print("    hpo:")
    print(f"      storage_url: \"{urls['dl']}\"")
    print("classification:")
    print("  ml:")
    print("    hpo:")
    print(f"      storage_url: \"{urls['ml']}\"")
    print("  dl:")
    print("    hpo:")
    print(f"      storage_url: \"{urls['dl']}\"")


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab DL tracking template")
    parser.add_argument("--init", action="store_true", help="Create required Drive folders")
    parser.add_argument("--show-config", action="store_true", help="Print storage_url config snippet")
    parser.add_argument("--status", action="store_true", help="Show env and path status")
    args = parser.parse_args()

    if args.init:
        ensure_colab_tracking_dirs()
        print(f"Initialized tracking folders under: {BASE_DIR}")

    if args.status:
        statuses = env_status()
        print("DagsHub env status:")
        for key, ok in statuses.items():
            print(f"- {key}: {'OK' if ok else 'MISSING'}")
        print("Path status:")
        print(f"- base: {'OK' if BASE_DIR.exists() else 'MISSING'} -> {BASE_DIR}")
        print(f"- ml db parent: {'OK' if ML_DB.parent.exists() else 'MISSING'} -> {ML_DB.parent}")
        print(f"- dl db parent: {'OK' if DL_DB.parent.exists() else 'MISSING'} -> {DL_DB.parent}")

    if args.show_config:
        print_config_snippet()

    if not any([args.init, args.show_config, args.status]):
        print("Usage examples:")
        print("- uv run python -m src.training.colab_dl_template --init --status")
        print("- uv run python -m src.training.colab_dl_template --show-config")


if __name__ == "__main__":
    main()
