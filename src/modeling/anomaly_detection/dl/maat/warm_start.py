from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WARM_START_PARAMS: dict = {
    "learning_rate":     0.000319861718220356,
    "weight_decay":      4.809461967501575e-06,
    "dropout":           0.019515477895583853,
    "k":                 4.7699849176399995,
    "temperature":       10,
    "batch_size":        256,
    "gradient_clip_val": 1.0,
    "win_size":          30,
    "block_size":        10,
    "d_model":           64,
    "n_heads":           2,
    "e_layers":          2,
    "d_ff":              256,
}


def load_warm_start_params(source: str | Path | None = None) -> dict:
    if source is not None:
        p = Path(source)
        f = p / "best_params.json" if p.is_dir() else p
        if f.exists():
            params = json.loads(f.read_text(encoding="utf-8"))
            logger.info(f"warm_start: loaded from {f}")
            return params
        logger.warning(f"warm_start: {f} not found — using WARM_START_PARAMS")
    return dict(WARM_START_PARAMS)
