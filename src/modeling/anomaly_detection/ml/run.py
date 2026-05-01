from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


def _load_model_config() -> dict:
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_active_model(config: dict) -> str:
    anomaly_cfg = config.get("anomaly_detection")
    if not isinstance(anomaly_cfg, dict):
        raise KeyError("Missing 'anomaly_detection' section in model_config.yaml")

    ml_cfg = anomaly_cfg.get("ml")
    if not isinstance(ml_cfg, dict):
        raise KeyError("Missing 'anomaly_detection.ml' section in model_config.yaml")

    active_model = ml_cfg.get("active_model")
    if not isinstance(active_model, str) or not active_model:
        raise KeyError("Missing non-empty 'anomaly_detection.ml.active_model' in model_config.yaml")

    return active_model


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=None)
    known_args, unknown_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *unknown_args]

    config = _load_model_config()
    active_model = str(known_args.model or _resolve_active_model(config))

    if active_model in {"isolation_forest", "one_class_svm", "xgboost", "switching_kalman", "bocd"}:
        raise NotImplementedError(
            "Anomaly ML baselines are currently scaffolded but not implemented: "
            "isolation_forest, one_class_svm, xgboost, switching_kalman, bocd."
        )

    raise ValueError(f"Unsupported anomaly_detection ml model: {active_model}")


if __name__ == "__main__":
    main()
