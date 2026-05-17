from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.modeling.classification.ml.catboost_model import run_catboost
from src.modeling.classification.ml.extra_trees_model import run_extra_trees
from src.modeling.classification.ml.lightgbm_model import run_lightgbm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


def _load_model_config() -> dict:
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_active_model(config: dict) -> str:
    classification_cfg = config.get("classification")
    if not isinstance(classification_cfg, dict):
        raise KeyError("Missing 'classification' section in model_config.yaml")

    ml_cfg = classification_cfg.get("ml")
    if not isinstance(ml_cfg, dict):
        raise KeyError("Missing 'classification.ml' section in model_config.yaml")

    active_model = ml_cfg.get("active_model")
    if not isinstance(active_model, str) or not active_model:
        raise KeyError("Missing non-empty 'classification.ml.active_model' in model_config.yaml")

    return active_model


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=None)
    known_args, remaining_args = parser.parse_known_args()

    # Remove dispatcher-only --model argument before delegating to model entrypoints,
    # because model-specific parsers do not accept this flag.
    sys.argv = [sys.argv[0], *remaining_args]

    config = _load_model_config()
    active_model = str(known_args.model or _resolve_active_model(config))

    if active_model == "lightgbm":
        run_lightgbm(config=config)
        return

    if active_model == "catboost":
        run_catboost(config=config)
        return

    if active_model == "extra_trees":
        run_extra_trees(config=config)
        return

    if active_model == "xgboost":
        raise NotImplementedError(
            f"Classification ML model '{active_model}' is scaffolded but not implemented yet."
        )

    raise ValueError(f"Unsupported classification ml model: {active_model}")


if __name__ == "__main__":
    main()
