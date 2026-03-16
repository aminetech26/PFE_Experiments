from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import yaml
from loguru import logger

from src.data.splitting import segment_aware_temporal_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"


# Core responsibility: Data IO + deterministic filtering + invoking split algorithm + artifact persistence.


def _load_embargo_seconds_from_eda_json(
    path: Path,
    *,
    json_key_path: str = "acf.embargo_seconds",
) -> tuple[int, dict]:
    if not path.exists():
        return 0, {
            "status": "fallback",
            "reason": "eda_json_not_found",
            "path": str(path),
            "json_key_path": json_key_path,
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parsing
        return 0, {
            "status": "fallback",
            "reason": "eda_json_parse_error",
            "path": str(path),
            "json_key_path": json_key_path,
            "error": str(exc),
        }

    current = payload
    try:
        for key in json_key_path.split("."):
            current = current[key]
        embargo_seconds = int(current)
    except Exception:
        return 0, {
            "status": "fallback",
            "reason": "eda_json_key_missing",
            "path": str(path),
            "json_key_path": json_key_path,
        }

    if embargo_seconds <= 0:
        return 0, {
            "status": "fallback",
            "reason": "eda_json_non_positive_value",
            "path": str(path),
            "json_key_path": json_key_path,
            "embargo_seconds": int(embargo_seconds),
        }

    return embargo_seconds, {
        "status": "derived",
        "path": str(path),
        "json_key_path": json_key_path,
        "embargo_seconds": int(embargo_seconds),
    }

def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    split_cfg = cfg["splits"]
    label_col = cfg["label_columns"]["reunion"]

    daytime_gti_threshold = float(split_cfg.get("daytime_gti_threshold", 10.0))
    segmentation_gap_seconds = int(split_cfg.get("segmentation_gap_seconds", 300))
    segment_gap_count = int(split_cfg.get("segment_gap_count", 0))
    legacy_manual_seconds = int(split_cfg.get("embargo_seconds", 0))
    embargo_fallback_seconds = int(split_cfg.get("embargo_fallback_seconds", legacy_manual_seconds))
    eda_json_relpath = str(split_cfg.get("eda_recommendations_path", "data/interim/eda_split_recommendations.json"))
    eda_json_key_path = str(split_cfg.get("eda_embargo_key_path", "acf.embargo_seconds"))
    min_train_count = int(split_cfg.get("min_train_count", 5))
    min_val_count = int(split_cfg.get("min_val_count", 2))
    train_min_frac = float(split_cfg.get("train_min_frac", 0.01))
    val_min_frac = float(split_cfg.get("val_min_frac", 0.005))
    rare_class_threshold = int(split_cfg.get("rare_class_threshold", 30))

    interim_dir = PROJECT_ROOT / paths["interim_dir"]
    processed_dir = PROJECT_ROOT / paths["processed_dir"]
    split_dir = processed_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    input_path = interim_dir / "reunion_dt2_merged.parquet"
    logger.info(f"Loading ingestion artifact: {input_path}")

    df = pl.read_parquet(input_path).to_pandas()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", label_col]).sort_values("time").reset_index(drop=True)

    # Universal deterministic pre-split logic: day filtering and contiguous segmentation.
    gti = df["GTI"].fillna(0) if "GTI" in df.columns else pd.Series(0, index=df.index)
    df["is_daytime"] = gti > daytime_gti_threshold
    df = df[df["is_daytime"]].reset_index(drop=True)
    dt_s = df["time"].diff().dt.total_seconds().fillna(0)
    df["segment_id"] = (dt_s > segmentation_gap_seconds).cumsum().astype(int)

    embargo_derivation = {"mode": "eda_json"}
    eda_json_path = PROJECT_ROOT / eda_json_relpath
    derived_seconds, derived_info = _load_embargo_seconds_from_eda_json(
        eda_json_path,
        json_key_path=eda_json_key_path,
    )
    if derived_seconds > 0:
        embargo_seconds = derived_seconds
    else:
        embargo_seconds = embargo_fallback_seconds
        derived_info["fallback_seconds"] = int(embargo_fallback_seconds)
    embargo_derivation.update(derived_info)

    artifacts = segment_aware_temporal_split(
        df,
        time_col="time",
        label_col=label_col,
        segment_col="segment_id",
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        gap_segments=segment_gap_count,
        embargo_seconds=embargo_seconds,
        min_train_count=min_train_count,
        min_val_count=min_val_count,
        train_min_frac=train_min_frac,
        val_min_frac=val_min_frac,
        rare_class_threshold=rare_class_threshold,
    )

    artifacts.manifest["embargo_derivation"] = embargo_derivation

    train_path = split_dir / "train_raw.parquet"
    val_path = split_dir / "val_raw.parquet"
    test_path = split_dir / "test_raw.parquet"
    dropped_path = split_dir / "dropped_embargo_raw.parquet"
    manifest_path = split_dir / "split_manifest.json"

    pl.from_pandas(artifacts.train).write_parquet(train_path)
    pl.from_pandas(artifacts.val).write_parquet(val_path)
    pl.from_pandas(artifacts.test).write_parquet(test_path)
    pl.from_pandas(artifacts.dropped).write_parquet(dropped_path)

    manifest_path.write_text(json.dumps(artifacts.manifest, indent=2), encoding="utf-8")

    logger.success("Split artifacts written.")
    logger.info(json.dumps(artifacts.manifest, indent=2))


if __name__ == "__main__":
    main()
