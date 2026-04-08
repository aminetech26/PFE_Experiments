"""
Feature engineering pipeline.

Reads preprocessed splits, applies flag-driven feature engineering, optional
correlation/VIF pruning, optional tsfresh extraction, and writes engineered outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from src.data.features import (
    add_physics_features,
    add_time_cyclic_features,
    add_wavelet_feature,
    apply_correlation_pruning,
    apply_vif_pruning,
    extract_tsfresh_segment_features,
    infer_base_feature_columns,
    infer_label_column,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "anomaly_supervised"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "features"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_profile(config: dict, profile_name: str | None) -> tuple[dict, dict, dict, str | None]:
    """Resolve effective flags, selection, and tsfresh settings from base + profile overrides."""
    fe_cfg = config.get("feature_engineering", {})
    flags = dict(fe_cfg.get("flags", {}))
    selection = dict(fe_cfg.get("selection", {}))
    tsfresh_cfg = dict(fe_cfg.get("tsfresh", {}))

    if profile_name:
        profile = fe_cfg.get("profiles", {}).get(profile_name)
        if profile is None:
            raise ValueError(f"Unknown feature engineering profile: {profile_name}")
        for key, value in profile.items():
            if key.startswith("enable_"):
                flags[key] = value
            elif key == "tsfresh_mode":
                tsfresh_cfg["mode"] = value

    return flags, selection, tsfresh_cfg, profile_name


def get_base_feature_columns(df: pd.DataFrame) -> list[str]:
    return infer_base_feature_columns(df)


def to_json_safe(value):
    """Recursively convert NaN/Inf into JSON-safe values."""
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value)


def _resolve_output_root(config: dict) -> tuple[Path, str]:
    fe_cfg = config.get("feature_engineering", {})
    out_cfg = fe_cfg.get("outputs", {})
    root_dir = out_cfg.get("root_dir", "data/processed/features")
    runs_subdir = out_cfg.get("runs_subdir", "runs")
    return PROJECT_ROOT / root_dir, str(runs_subdir)


def _load_eda_findings(project_root: Path, selection_cfg: dict) -> tuple[dict | None, dict]:
    use_eda = bool(selection_cfg.get("use_eda_findings", False))
    rel_path = selection_cfg.get("eda_findings_path", "data/interim/eda_feature_findings.json")
    path = project_root / rel_path

    if not use_eda:
        return None, {"enabled": False, "path": str(path), "loaded": False}

    if not path.exists():
        return None, {"enabled": True, "path": str(path), "loaded": False, "reason": "missing_file"}

    data = json.loads(path.read_text(encoding="utf-8"))
    return data, {
        "enabled": True,
        "path": str(path),
        "loaded": True,
        "version": data.get("version"),
        "created_at": data.get("created_at"),
    }


def _apply_eda_selection_priors(selection_cfg: dict, eda_findings: dict | None) -> tuple[dict, list[str]]:
    effective = dict(selection_cfg)
    pre_drop: list[str] = []

    if not eda_findings:
        return effective, pre_drop

    if effective.get("eda_override_thresholds", False):
        sp_thr = eda_findings.get("spearman", {}).get("recommended_corr_threshold")
        vif_thr = eda_findings.get("vif", {}).get("recommended_vif_threshold")
        if sp_thr is not None:
            effective["corr_threshold"] = float(sp_thr)
        if vif_thr is not None:
            effective["vif_threshold"] = float(vif_thr)

    anchors = set(effective.get("anchor_features", []))
    if effective.get("eda_prefer_anchors_from_findings", True):
        mi_top = eda_findings.get("mutual_information", {}).get("top_features_binary", [])
        mw_sig = eda_findings.get("mannwhitney", {}).get("significant_features", [])
        anchors.update(mi_top[:5])
        anchors.update(mw_sig[:5])
    effective["anchor_features"] = sorted(anchors)

    if effective.get("eda_pre_drop_candidates", True):
        recs = eda_findings.get("recommendations", {})
        pre_drop = [f for f in recs.get("redundant_drop_candidates", []) if f not in anchors]

    return effective, pre_drop


def _tsfresh_mode(tsfresh_cfg: dict) -> str:
    raw_tsfresh_mode = tsfresh_cfg.get("mode", "off")
    if raw_tsfresh_mode in (False, None):
        return "off"
    mode = str(raw_tsfresh_mode).lower()
    if mode in {"false", "0", "none"}:
        return "off"
    return mode


def _build_config_fingerprint(payload: dict) -> str:
    normalized = to_json_safe(payload)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def add_optional_features(df: pd.DataFrame, flags: dict) -> tuple[pd.DataFrame, list[str]]:
    """Apply enabled feature generators and return added feature names."""
    out = df.copy()
    added: list[str] = []

    if flags.get("enable_hour_cyclic", False):
        out = add_time_cyclic_features(out)
        for c in ("hour_sin", "hour_cos"):
            if c in out.columns:
                added.append(c)

    out = add_physics_features(out, flags=flags)
    physics_cols = [
        "performance_ratio",
        "delta_temp",
        "dP_dt",
        "dV_dt",
        "dI_dt",
        "Vg_normalized",
    ]
    for c in physics_cols:
        if c in out.columns and c not in added:
            added.append(c)

    if flags.get("enable_wavelet", False):
        out = add_wavelet_feature(out, source_col="Pg", target_col="Pg_wavelet")
        if "Pg_wavelet" in out.columns:
            added.append("Pg_wavelet")

    if flags.get("enable_differential_signal", False):
        # Differential signal is ablation-only and depends on having paired inverter columns.
        if {"Pg_inv1", "Pg_inv2"}.issubset(out.columns):
            out["delta_p"] = out["Pg_inv1"] - out["Pg_inv2"]
            added.append("delta_p")
        elif {"Pg", "Pg_ref"}.issubset(out.columns):
            out["delta_p"] = out["Pg"] - out["Pg_ref"]
            added.append("delta_p")
        else:
            logger.warning("Differential signal enabled but required columns are missing; skipping.")

    return out, added


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline.")
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional feature-engineering profile from data_config.yaml",
    )
    args = parser.parse_args()

    config = load_config()
    flags, selection_cfg, tsfresh_cfg, profile_name = resolve_profile(config, args.profile)
    profile_key = _safe_name(profile_name or "default")
    output_root, runs_subdir = _resolve_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    runs_root = output_root / runs_subdir
    runs_root.mkdir(parents=True, exist_ok=True)

    eda_findings, eda_meta = _load_eda_findings(PROJECT_ROOT, selection_cfg)
    selection_effective, eda_pre_drop = _apply_eda_selection_priors(selection_cfg, eda_findings)
    tsfresh_mode = _tsfresh_mode(tsfresh_cfg)

    resolved_for_fingerprint = {
        "profile": profile_key,
        "flags": flags,
        "selection": {
            "corr_threshold": float(selection_effective.get("corr_threshold", 0.95)),
            "vif_threshold": float(selection_effective.get("vif_threshold", 10.0)),
            "max_vif_rows": int(selection_effective.get("max_vif_rows", 50000)),
            "anchor_features": selection_effective.get("anchor_features", []),
            "use_eda_findings": bool(selection_effective.get("use_eda_findings", False)),
            "eda_findings_path": selection_effective.get("eda_findings_path"),
            "eda_prefer_anchors_from_findings": bool(
                selection_effective.get("eda_prefer_anchors_from_findings", True)
            ),
            "eda_pre_drop_candidates": bool(selection_effective.get("eda_pre_drop_candidates", True)),
            "eda_override_thresholds": bool(selection_effective.get("eda_override_thresholds", False)),
        },
        "tsfresh": {
            "mode": tsfresh_mode,
            "top_k": int(tsfresh_cfg.get("top_k", 20)),
            "n_segments_sample": int(tsfresh_cfg.get("n_segments_sample", 60)),
            "max_rows_per_segment": int(tsfresh_cfg.get("max_rows_per_segment", 800)),
        },
    }
    config_fingerprint = _build_config_fingerprint(resolved_for_fingerprint)
    run_dir = runs_root / f"{profile_key}__{config_fingerprint}"

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    logger.info(f"Feature profile: {profile_name or 'default'}")
    logger.info(f"Input: {INPUT_DIR}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    split_frames: dict[str, pd.DataFrame] = {}
    for subset in ("train", "val", "test"):
        path = INPUT_DIR / f"{subset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing input split file: {path}")
        frame = pd.read_parquet(path)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        split_frames[subset] = frame

    label_col = infer_label_column(split_frames["train"])
    base_cols = get_base_feature_columns(split_frames["train"])

    generated_cols: dict[str, list[str]] = {}
    for subset in ("train", "val", "test"):
        split_frames[subset], added = add_optional_features(split_frames[subset], flags)
        generated_cols[subset] = added

    candidate_features = sorted(
        set(base_cols + generated_cols["train"] + generated_cols["val"] + generated_cols["test"])
    )
    candidate_features = [c for c in candidate_features if c in split_frames["train"].columns]
    if eda_pre_drop:
        candidate_features = [c for c in candidate_features if c not in set(eda_pre_drop)]

    corr_dropped: list[dict] = []
    vif_dropped: list[dict] = []
    selected_features = candidate_features.copy()

    if flags.get("enable_corr_pruning", False) and selected_features:
        split_frames["train"], split_frames["val"], split_frames["test"], selected_features, corr_dropped = (
            apply_correlation_pruning(
                split_frames["train"],
                split_frames["val"],
                split_frames["test"],
                selected_features,
                threshold=float(selection_effective.get("corr_threshold", 0.95)),
                anchor_cols=selection_effective.get("anchor_features", []),
            )
        )

    if flags.get("enable_vif_pruning", False) and selected_features:
        split_frames["train"], split_frames["val"], split_frames["test"], selected_features, vif_dropped = (
            apply_vif_pruning(
                split_frames["train"],
                split_frames["val"],
                split_frames["test"],
                selected_features,
                threshold=float(selection_effective.get("vif_threshold", 10.0)),
                anchor_cols=selection_effective.get("anchor_features", []),
                max_rows=int(selection_effective.get("max_vif_rows", 50000)),
            )
        )

    tsfresh_top_k = int(tsfresh_cfg.get("top_k", 20))
    tsfresh_cols: list[str] = []
    tsfresh_meta: dict = {"mode": tsfresh_mode, "selected": 0}

    if tsfresh_mode != "off":
        split_frames["train"], split_frames["val"], split_frames["test"], tsfresh_cols, tsfresh_meta = (
            extract_tsfresh_segment_features(
                split_frames["train"],
                split_frames["val"],
                split_frames["test"],
                feature_cols=selected_features,
                mode=tsfresh_mode,
                top_k=tsfresh_top_k,
                segment_col="segment_id",
                time_col="timestamp",
                label_col=label_col,
                n_segments_sample=int(tsfresh_cfg.get("n_segments_sample", 60)),
                max_rows_per_segment=int(tsfresh_cfg.get("max_rows_per_segment", 800)),
                n_jobs=int(tsfresh_cfg.get("n_jobs", -1)),
            )
        )

    final_feature_cols = []
    for col in selected_features + tsfresh_cols:
        if col in split_frames["train"].columns and col not in final_feature_cols:
            final_feature_cols.append(col)

    meta_cols = [c for c in ("timestamp", "segment_id", "label", "Fault") if c in split_frames["train"].columns]
    if label_col not in meta_cols and label_col in split_frames["train"].columns:
        meta_cols.append(label_col)

    for subset in ("train", "val", "test"):
        out_df = split_frames[subset][meta_cols + final_feature_cols].copy()
        out_path = run_dir / f"{subset}.parquet"
        out_df.to_parquet(out_path, index=False)
        logger.info(f"Saved {subset}: {out_path} ({len(out_df):,} rows, {len(final_feature_cols)} features)")

    manifest = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile_name or "default",
        "config_fingerprint": config_fingerprint,
        "output_directory": str(run_dir),
        "label_column": label_col,
        "active_flags": flags,
        "selection": {
            "corr_threshold": float(selection_effective.get("corr_threshold", 0.95)),
            "vif_threshold": float(selection_effective.get("vif_threshold", 10.0)),
            "anchor_features": selection_effective.get("anchor_features", []),
            "use_eda_findings": bool(selection_effective.get("use_eda_findings", False)),
            "eda_pre_drop_applied": eda_pre_drop,
            "corr_dropped": corr_dropped,
            "vif_dropped": vif_dropped,
        },
        "eda_findings": eda_meta,
        "tsfresh": tsfresh_meta,
        "base_feature_count": len(base_cols),
        "final_feature_count": len(final_feature_cols),
        "final_features": final_feature_cols,
    }

    manifest_path = run_dir / "features_manifest.json"
    manifest = to_json_safe(manifest)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    resolved_path = run_dir / "resolved_config.json"
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(resolved_for_fingerprint), f, indent=2, allow_nan=False)

    latest_path = output_root / "latest_run.txt"
    latest_path.write_text(str(run_dir.relative_to(output_root)), encoding="utf-8")

    logger.success(f"Feature engineering complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

