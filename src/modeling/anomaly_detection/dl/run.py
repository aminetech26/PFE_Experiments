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
    dl_cfg = config.get("anomaly_detection", {}).get("dl", {})
    active_model = dl_cfg.get("active_model")
    if not isinstance(active_model, str) or not active_model:
        raise KeyError(
            "Missing non-empty 'anomaly_detection.dl.active_model' in model_config.yaml"
        )
    return active_model


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=None)
    known_args, unknown_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *unknown_args]

    config = _load_model_config()
    active_model = str(known_args.model or _resolve_active_model(config))

    if active_model == "maat":
        from src.modeling.anomaly_detection.dl.maat.trainer import run_maat  # noqa: PLC0415
        run_maat(config=config)
        return

    if active_model == "dlssm":
        from src.modeling.anomaly_detection.dl.dlssm.trainer import run_dlssm  # noqa: PLC0415
        run_dlssm(config=config)
        return

    raise ValueError(
        f"Unsupported anomaly_detection dl model: '{active_model}'. "
        f"Supported: maat, dlssm"
    )


if __name__ == "__main__":
    main()
