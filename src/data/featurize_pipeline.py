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
from typing import Any

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

TASK_CHOICES = ("anomaly_semisup", "anomaly_supervised", "classification", "prediction")

PROFILE_FLAG_KEYS = {
    "wavelet_threshold_strategy",
    "include_preprocessed_stationarity_features",
    "preprocessed_stationarity_suffixes",
}

DERIVED_FEATURE_ENABLE_FLAGS = {
    "delta_temp": "enable_delta_temp",
    "dP_dt": "enable_dP_dt",
    "dV_dt": "enable_dV_dt",
    "dI_dt": "enable_dI_dt",
    "Vg_normalized": "enable_Vg_normalized",
    "Pg_wavelet": "enable_wavelet",
    "delta_p": "enable_differential_signal",
}


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
            if key.startswith("enable_") or key in PROFILE_FLAG_KEYS:
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


def _resolve_input_dir(task: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / "preprocessed" / task


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_task_and_directives(config: dict, task_arg: str | None) -> tuple[str, dict]:
    fe_cfg = config.get("feature_engineering", {})
    task_cfg_all = fe_cfg.get("task_directives", {})

    default_task = str(task_cfg_all.get("default_task", "anomaly_supervised"))
    task = task_arg or default_task
    if task not in TASK_CHOICES:
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {TASK_CHOICES}")

    common_cfg = task_cfg_all.get("common", {})
    task_cfg = task_cfg_all.get(task, {})
    merged = _merge_dict(common_cfg, task_cfg)
    return task, merged


def _apply_task_overrides(
    flags: dict,
    selection: dict,
    tsfresh_cfg: dict,
    task_directives: dict,
) -> tuple[dict, dict, dict]:
    eff_flags = dict(flags)
    eff_selection = dict(selection)
    eff_tsfresh = dict(tsfresh_cfg)

    for key, value in task_directives.get("flags", {}).items():
        if key.startswith("enable_") or key in PROFILE_FLAG_KEYS:
            eff_flags[key] = value

    eff_selection.update(task_directives.get("selection", {}))
    eff_tsfresh.update(task_directives.get("tsfresh", {}))
    return eff_flags, eff_selection, eff_tsfresh


def _resolve_eda_policy(task: str, task_directives: dict) -> dict:
    eda_cfg = task_directives.get("eda", {})
    mi_top_key = eda_cfg.get("mi_top_key")
    if not mi_top_key:
        mi_top_key = "top_features_multiclass" if task == "classification" else "top_features_binary"

    use_mannwhitney = bool(eda_cfg.get("use_mannwhitney", task != "prediction"))
    return {
        "mi_top_key": str(mi_top_key),
        "use_mannwhitney": use_mannwhitney,
    }


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


def _apply_eda_selection_priors(
    selection_cfg: dict,
    eda_findings: dict | None,
    eda_policy: dict,
) -> tuple[dict, list[str]]:
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
        mi_top_key = eda_policy.get("mi_top_key", "top_features_binary")
        mi_top = eda_findings.get("mutual_information", {}).get(mi_top_key, [])
        mw_sig = (
            eda_findings.get("mannwhitney", {}).get("significant_features", [])
            if eda_policy.get("use_mannwhitney", True)
            else []
        )
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


def _resolve_run_dir(base_run_dir: Path) -> Path:
    """Return a non-colliding run directory path.

    First run keeps deterministic naming. If the exact directory already exists,
    append a timestamp suffix so repeated runs don't overwrite prior artifacts.
    """
    if not base_run_dir.exists():
        return base_run_dir

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return base_run_dir.with_name(f"{base_run_dir.name}__{stamp}")


def _detect_preprocessed_stationarity_columns(
    split_frames: dict[str, pd.DataFrame],
    suffixes: list[str],
) -> list[str]:
    """Detect stationarity columns already created during preprocessing.

    A column is included if it ends with one of the configured suffixes and
    exists across train/val/test, so downstream column selection remains safe.
    """
    if not suffixes:
        return []

    train_cols = list(split_frames["train"].columns)
    matched = [c for c in train_cols if any(c.endswith(suffix) for suffix in suffixes)]
    return [c for c in matched if all(c in split_frames[s].columns for s in ("val", "test"))]


def _apply_eda_predrop_before_feature_generation(
    split_frames: dict[str, pd.DataFrame],
    pre_drop_cols: list[str],
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """Drop EDA pre-drop columns from all splits before feature generation."""
    applied: list[str] = []
    unavailable: list[str] = []

    for col in pre_drop_cols:
        if any(col in split_frames[s].columns for s in ("train", "val", "test")):
            applied.append(col)
        else:
            unavailable.append(col)

    if not applied:
        return split_frames, applied, unavailable

    updated: dict[str, pd.DataFrame] = {}
    for subset, frame in split_frames.items():
        drop_cols = [c for c in applied if c in frame.columns]
        updated[subset] = frame.drop(columns=drop_cols)
    return updated, applied, unavailable


def _apply_predrop_derived_blocking(flags: dict, predropped_cols: list[str]) -> tuple[dict, list[dict]]:
    """Disable derived features that depend on EDA pre-dropped sources."""
    predropped = set(predropped_cols)
    effective = dict(flags)
    blocked: list[dict] = []

    def _block(feature_name: str, blocked_sources: list[str]) -> None:
        enable_key = DERIVED_FEATURE_ENABLE_FLAGS[feature_name]
        if not effective.get(enable_key, False):
            return
        effective[enable_key] = False
        blocked.append(
            {
                "feature": feature_name,
                "flag": enable_key,
                "reason": "source_predropped_by_eda",
                "blocked_by_sources": blocked_sources,
            }
        )

    for feature_name, source_cols in {
        "delta_temp": {"TPV", "TA"},
        "dP_dt": {"Pg"},
        "dV_dt": {"Vg"},
        "dI_dt": {"Ig"},
        "Vg_normalized": {"Vg"},
        "Pg_wavelet": {"Pg"},
    }.items():
        overlap = sorted(source_cols.intersection(predropped))
        if overlap:
            _block(feature_name, overlap)

    # delta_p can be computed from either inverter pair or Pg/Pg_ref pair.
    if effective.get("enable_differential_signal", False):
        source_groups = [
            {"Pg_inv1", "Pg_inv2"},
            {"Pg", "Pg_ref"},
        ]
        group_is_viable = [group.isdisjoint(predropped) for group in source_groups]
        if not any(group_is_viable):
            blocked_sources = sorted({c for g in source_groups for c in g if c in predropped})
            _block("delta_p", blocked_sources)

    return effective, blocked


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
        out = add_wavelet_feature(
            out,
            source_col="Pg",
            target_col="Pg_wavelet",
            threshold_strategy=str(flags.get("wavelet_threshold_strategy", "per_segment")),
        )
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
        "--task",
        default=None,
        choices=TASK_CHOICES,
        help="Task split to featurize (default is configured task_directives.default_task).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional feature-engineering profile from data_config.yaml",
    )
    args = parser.parse_args()

    config = load_config()
    task, task_directives = _resolve_task_and_directives(config, args.task)

    base_flags, base_selection_cfg, base_tsfresh_cfg, profile_name = resolve_profile(config, args.profile)
    flags, selection_cfg, tsfresh_cfg = _apply_task_overrides(
        base_flags,
        base_selection_cfg,
        base_tsfresh_cfg,
        task_directives,
    )

    input_dir = _resolve_input_dir(task)
    profile_key = _safe_name(profile_name or "default")
    task_key = _safe_name(task)
    output_root, runs_subdir = _resolve_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    task_output_root = output_root / task_key
    runs_root = task_output_root / runs_subdir
    runs_root.mkdir(parents=True, exist_ok=True)

    eda_policy = _resolve_eda_policy(task, task_directives)
    eda_findings, eda_meta = _load_eda_findings(PROJECT_ROOT, selection_cfg)
    selection_effective, eda_pre_drop = _apply_eda_selection_priors(selection_cfg, eda_findings, eda_policy)
    tsfresh_mode = _tsfresh_mode(tsfresh_cfg)
    tsfresh_label_strategy = str(task_directives.get("tsfresh_label_strategy", "any_fault"))
    predrop_before_featurization = bool(selection_effective.get("eda_apply_predrop_before_feature_generation", True))
    block_derived_from_predrop = bool(selection_effective.get("eda_block_derived_from_predropped_sources", True))
    include_preprocessed_stationarity = bool(flags.get("include_preprocessed_stationarity_features", False))

    raw_stationarity_suffixes = flags.get("preprocessed_stationarity_suffixes", ["_norm", "_detrend"])
    if isinstance(raw_stationarity_suffixes, str):
        stationarity_suffixes = [raw_stationarity_suffixes]
    elif isinstance(raw_stationarity_suffixes, list):
        stationarity_suffixes = [str(s) for s in raw_stationarity_suffixes if str(s)]
    else:
        stationarity_suffixes = ["_norm", "_detrend"]

    resolved_for_fingerprint = {
        "task": task,
        "profile": profile_key,
        "flags": flags,
        "task_directives": task_directives,
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
            "eda_apply_predrop_before_feature_generation": predrop_before_featurization,
            "eda_block_derived_from_predropped_sources": block_derived_from_predrop,
        },
        "tsfresh": {
            "mode": tsfresh_mode,
            "top_k": int(tsfresh_cfg.get("top_k", 20)),
            "n_segments_sample": int(tsfresh_cfg.get("n_segments_sample", 60)),
            "max_rows_per_segment": int(tsfresh_cfg.get("max_rows_per_segment", 800)),
            "label_strategy": tsfresh_label_strategy,
        },
    }
    config_fingerprint = _build_config_fingerprint(resolved_for_fingerprint)
    base_run_dir = runs_root / f"{profile_key}__{config_fingerprint}"
    run_dir = _resolve_run_dir(base_run_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found for task '{task}': {input_dir}")

    logger.info(f"Feature profile: {profile_name or 'default'}")
    logger.info(f"Task: {task}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    split_frames: dict[str, pd.DataFrame] = {}
    for subset in ("train", "val", "test"):
        path = input_dir / f"{subset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing input split file: {path}")
        frame = pd.read_parquet(path)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        split_frames[subset] = frame

    eda_predrop_applied_before_fe: list[str] = []
    eda_predrop_unavailable_before_fe: list[str] = []
    if predrop_before_featurization and eda_pre_drop:
        split_frames, eda_predrop_applied_before_fe, eda_predrop_unavailable_before_fe = (
            _apply_eda_predrop_before_feature_generation(split_frames, eda_pre_drop)
        )

    generation_flags = dict(flags)
    derived_blocked_by_predrop: list[dict] = []
    if block_derived_from_predrop and eda_predrop_applied_before_fe:
        generation_flags, derived_blocked_by_predrop = _apply_predrop_derived_blocking(
            generation_flags,
            eda_predrop_applied_before_fe,
        )

    label_col = infer_label_column(split_frames["train"])
    base_cols = get_base_feature_columns(split_frames["train"])
    stationarity_cols = (
        _detect_preprocessed_stationarity_columns(split_frames, stationarity_suffixes)
        if include_preprocessed_stationarity
        else []
    )

    generated_cols: dict[str, list[str]] = {}
    for subset in ("train", "val", "test"):
        split_frames[subset], added = add_optional_features(split_frames[subset], generation_flags)
        generated_cols[subset] = added

    candidate_features = sorted(
        set(base_cols + stationarity_cols + generated_cols["train"] + generated_cols["val"] + generated_cols["test"])
    )
    candidate_features = [c for c in candidate_features if c in split_frames["train"].columns]
    candidate_predrop_applied = [c for c in eda_pre_drop if c in set(candidate_features)]
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
                label_strategy=tsfresh_label_strategy,
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
        "source_task": task,
        "source_preprocessed_dir": str(input_dir),
        "label_column": label_col,
        "active_flags": flags,
        "effective_generation_flags": generation_flags,
        "task_directives_effective": task_directives,
        "preprocessed_stationarity": {
            "enabled": include_preprocessed_stationarity,
            "suffixes": stationarity_suffixes,
            "included_columns": stationarity_cols,
            "count": len(stationarity_cols),
        },
        "selection": {
            "corr_threshold": float(selection_effective.get("corr_threshold", 0.95)),
            "vif_threshold": float(selection_effective.get("vif_threshold", 10.0)),
            "anchor_features": selection_effective.get("anchor_features", []),
            "use_eda_findings": bool(selection_effective.get("use_eda_findings", False)),
            "eda_policy": eda_policy,
            "eda_pre_drop_candidates": eda_pre_drop,
            "eda_predrop_before_featurization_enabled": predrop_before_featurization,
            "eda_pre_drop_applied_before_featurization": eda_predrop_applied_before_fe,
            "eda_pre_drop_unavailable_before_featurization": eda_predrop_unavailable_before_fe,
            "eda_pre_drop_applied_candidate_filter": candidate_predrop_applied,
            "derived_blocking": {
                "enabled": block_derived_from_predrop,
                "blocked_features": derived_blocked_by_predrop,
            },
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
    resolved_payload = {
        **to_json_safe(resolved_for_fingerprint),
        "io": {
            "input_dir": str(input_dir),
            "output_root": str(output_root),
            "run_dir": str(run_dir),
        },
    }
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(resolved_payload, f, indent=2, allow_nan=False)

    relative_run_dir = str(run_dir.relative_to(output_root))

    latest_by_task_path = output_root / "latest_runs.json"
    latest_by_task: dict[str, Any] = {"latest_by_task": {}}
    if latest_by_task_path.exists():
        try:
            loaded = json.loads(latest_by_task_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                latest_by_task = loaded
        except Exception:
            latest_by_task = {"latest_by_task": {}}
    if "latest_by_task" not in latest_by_task or not isinstance(latest_by_task.get("latest_by_task"), dict):
        latest_by_task["latest_by_task"] = {}

    if "latest_by_task_profile" not in latest_by_task or not isinstance(
        latest_by_task.get("latest_by_task_profile"), dict
    ):
        latest_by_task["latest_by_task_profile"] = {}

    latest_by_task["latest_by_task"][task] = relative_run_dir
    task_profile_map = latest_by_task["latest_by_task_profile"].setdefault(task, {})
    if not isinstance(task_profile_map, dict):
        task_profile_map = {}
        latest_by_task["latest_by_task_profile"][task] = task_profile_map
    task_profile_map[profile_key] = relative_run_dir

    latest_by_task["last_run"] = {
        "task": task,
        "profile": profile_key,
        "path": relative_run_dir,
    }
    latest_by_task["updated_at"] = datetime.now(UTC).isoformat()
    latest_by_task_path.write_text(json.dumps(to_json_safe(latest_by_task), indent=2), encoding="utf-8")

    logger.success(f"Feature engineering complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

