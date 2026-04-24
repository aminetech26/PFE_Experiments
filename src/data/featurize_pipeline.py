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
    add_multiscale_window_features,
    add_physics_features,
    add_rolling_statistics_features,
    add_time_cyclic_features,
    add_wavelet_feature,
    apply_correlation_pruning,
    apply_hygiene_pruning,
    apply_mrmr_selection,
    apply_vif_pruning,
    extract_tsfresh_segment_features,
    infer_base_feature_columns,
    infer_label_column,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"

TASK_CHOICES = ("anomaly_semisup", "anomaly_supervised", "classification")

PROFILE_FLAG_KEYS = {
    "wavelet_threshold_strategy",
    "rolling_windows",
    "rolling_stats",
    "window_size",
    "window_step",
    "multiscale_window_sizes",
    "spectral_min_fs",
    "spectral_n_top_freqs",
    "psd_n_bands",
    "wpd_level",
    "wpd_wavelet",
    "ceemdan_n_imfs",
    "ceemdan_n_ensemble",
    "temp_ref_c",
    "gamma_pmax_pct_per_c",
    "temp_power_eps",
    "irr_norm_floor",
    "mrmr_k",
}

DERIVED_FEATURE_ENABLE_FLAGS = {
    "delta_temp": "enable_delta_temp",
    "dP_dt": "enable_dP_dt",
    "dV_dt": "enable_dV_dt",
    "dI_dt": "enable_dI_dt",
    "Vg_normalized": "enable_Vg_normalized",
    "power_imbalance": "enable_power_imbalance",
    "current_imbalance": "enable_current_imbalance",
    "voltage_imbalance": "enable_voltage_imbalance",
    "string1_power_share": "enable_string_share",
    "string1_current_share": "enable_string_share",
    "temp_loss_pmax": "enable_temp_power_correction",
    "pdc_temp_corrected": "enable_temp_power_correction",
    "pdc_temp_corrected_norm_irr": "enable_temp_power_correction",
    "Pg_wavelet": "enable_wavelet",
    "delta_p": "enable_differential_signal",
}

DERIVED_SOURCE_COLUMNS = {
    "delta_temp": {"TPV", "TA"},
    "dP_dt": {"Pg"},
    "dV_dt": {"Vg"},
    "dI_dt": {"Ig"},
    "Vg_normalized": {"Vg"},
    "power_imbalance": {"pdc1", "pdc2"},
    "current_imbalance": {"idc1", "idc2"},
    "voltage_imbalance": {"vdc1", "vdc2"},
    "string1_power_share": {"pdc1", "pdc2"},
    "string1_current_share": {"idc1", "idc2"},
    "temp_loss_pmax": {"pvt"},
    "pdc_temp_corrected": {"pdc", "pvt"},
    "pdc_temp_corrected_norm_irr": {"pdc", "pvt", "irr"},
    "Pg_wavelet": {"Pg"},
}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_active_dataset(config: dict) -> str:
    return str(config.get("active_dataset", "la_reunion"))


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


def get_dataset_base_feature_columns(config: dict, dataset: str, df: pd.DataFrame) -> list[str]:
    ds_cfg = config.get("paths", {}).get("datasets", {}).get(dataset, {})
    ds_sensor_cols = ds_cfg.get("feature_engineering", {}).get("sensor_columns", [])
    if ds_sensor_cols:
        resolved = [c for c in ds_sensor_cols if c in df.columns]
        if resolved:
            return resolved
    return get_base_feature_columns(df)


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


def _resolve_output_root(config: dict, dataset: str, split_path: str) -> tuple[Path, str]:
    fe_cfg = config.get("feature_engineering", {})
    out_cfg = fe_cfg.get("outputs", {})
    root_dir = out_cfg.get("root_dir", "data/processed/features")
    runs_subdir = out_cfg.get("runs_subdir", "runs")
    output_root = PROJECT_ROOT / root_dir / dataset
    if split_path == "path_b":
        output_root = output_root / "path_b"
    return output_root, str(runs_subdir)


def _apply_dataset_overrides(
    config: dict,
    dataset: str,
    split_path: str,
    flags: dict,
    selection: dict,
) -> tuple[dict, dict]:
    """Merge dataset-specific feature_engineering flags and selection into global defaults."""
    ds_cfg = config.get("paths", {}).get("datasets", {}).get(dataset, {})
    ds_fe = ds_cfg.get("feature_engineering", {})
    path_fe = (
        ds_cfg.get("splits", {})
        .get("comparison_paths", {})
        .get(split_path, {})
        .get("feature_engineering", {})
    )

    # Dataset flag overrides (only keys explicitly set in dataset config)
    for key, value in ds_fe.get("flags", {}).items():
        flags[key] = value

    # Dataset selection overrides (anchor_features etc.)
    for key, value in ds_fe.get("selection", {}).items():
        selection[key] = value

    # Split-path specific overrides (Path A vs Path B)
    for key, value in path_fe.get("flags", {}).items():
        flags[key] = value
    for key, value in path_fe.get("selection", {}).items():
        selection[key] = value

    # Inject dataset-specific eda_findings_path into selection so _load_eda_findings picks it up
    ds_eda_path = ds_cfg.get("eda_findings_path")
    if ds_eda_path:
        selection["eda_findings_path"] = ds_eda_path

    return flags, selection


def _resolve_input_dir(dataset: str, task: str, split_path: str) -> Path:
    input_root = PROJECT_ROOT / "data" / "processed" / "preprocessed" / dataset
    if split_path == "path_b":
        input_root = input_root / "path_b"
    return input_root / task


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
        mi_top_key = (
            "top_features_multiclass" if task == "classification" else "top_features_binary"
        )

    use_mannwhitney = bool(eda_cfg.get("use_mannwhitney", True))
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


def _apply_predrop_derived_blocking(
    flags: dict, predropped_cols: list[str]
) -> tuple[dict, list[dict]]:
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

    for feature_name, source_cols in DERIVED_SOURCE_COLUMNS.items():
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


def _protect_predrop_sources_for_enabled_derived(
    pre_drop_cols: list[str], flags: dict
) -> tuple[list[str], list[str]]:
    """Avoid pre-dropping source channels required by currently enabled derived features."""
    protected_sources: set[str] = set()
    for feature_name, enable_key in DERIVED_FEATURE_ENABLE_FLAGS.items():
        if flags.get(enable_key, False):
            protected_sources.update(DERIVED_SOURCE_COLUMNS.get(feature_name, set()))

    kept = [c for c in pre_drop_cols if c not in protected_sources]
    removed = [c for c in pre_drop_cols if c in protected_sources]
    return kept, removed


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
        "power_imbalance",
        "current_imbalance",
        "voltage_imbalance",
        "string1_power_share",
        "string1_current_share",
        "temp_loss_pmax",
        "pdc_temp_corrected",
        "pdc_temp_corrected_norm_irr",
    ]
    for c in physics_cols:
        if c in out.columns and c not in added:
            added.append(c)

    if flags.get("enable_rolling_stats", False):
        rolling_source_cols = infer_base_feature_columns(out)
        out, rolling_added = add_rolling_statistics_features(
            out,
            feature_cols=rolling_source_cols,
            segment_col="segment_id",
            time_col="timestamp",
            windows=flags.get("rolling_windows", [5, 10, 30]),
            stats=flags.get("rolling_stats", ["mean", "std", "min", "max"]),
        )
        for c in rolling_added:
            if c not in added:
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
            logger.warning(
                "Differential signal enabled but required columns are missing; skipping."
            )

    return out, added


def main() -> None:
    config = load_config()
    default_dataset = get_active_dataset(config)
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline.")
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help=f"Dataset to featurize (must match a key in data_config.yaml paths.datasets). Default: {default_dataset}",
    )
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
    parser.add_argument(
        "--split-path",
        default="path_a",
        choices=["path_a", "path_b"],
        help="Which preprocessed split path to featurize. Default: path_a",
    )
    args = parser.parse_args()

    # Validate dataset
    available_datasets = list(config.get("paths", {}).get("datasets", {}).keys())
    if args.dataset not in available_datasets:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {available_datasets}")

    task, task_directives = _resolve_task_and_directives(config, args.task)

    # Resolution order: global defaults → dataset overrides → task_directive → profile
    base_flags, base_selection_cfg, base_tsfresh_cfg, profile_name = resolve_profile(
        config, args.profile
    )
    base_flags, base_selection_cfg = _apply_dataset_overrides(
        config,
        args.dataset,
        args.split_path,
        base_flags,
        base_selection_cfg,
    )
    flags, selection_cfg, tsfresh_cfg = _apply_task_overrides(
        base_flags,
        base_selection_cfg,
        base_tsfresh_cfg,
        task_directives,
    )

    # Path-level admissibility guards.
    if args.split_path == "path_a":
        if flags.get("enable_hour_cyclic", False):
            logger.info("Path A policy: disabling hour-cyclic temporal context by default")
            flags["enable_hour_cyclic"] = False
        if flags.get("enable_wavelet", False):
            logger.info("Path A policy: disabling wavelet/spectral features (Path B only)")
            flags["enable_wavelet"] = False
        for _spectral_flag in ("enable_spectral", "enable_psd", "enable_wpd", "enable_ceemdan"):
            if flags.get(_spectral_flag, False):
                logger.info(f"Path A policy: disabling {_spectral_flag} (Path B only)")
                flags[_spectral_flag] = False

    # Windowing takes precedence: rolling stats encode the same temporal structure redundantly.
    if flags.get("enable_explicit_windowing", False) and flags.get("enable_rolling_stats", False):
        logger.info(
            "Windowing policy: auto-disabling rolling stats (redundant when windowing is active)"
        )
        flags["enable_rolling_stats"] = False

    input_dir = _resolve_input_dir(args.dataset, task, args.split_path)
    profile_key = _safe_name(profile_name or "default")
    task_key = _safe_name(task)
    output_root, runs_subdir = _resolve_output_root(config, args.dataset, args.split_path)
    output_root.mkdir(parents=True, exist_ok=True)
    task_output_root = output_root / task_key
    runs_root = task_output_root / runs_subdir
    runs_root.mkdir(parents=True, exist_ok=True)

    eda_policy = _resolve_eda_policy(task, task_directives)
    eda_findings, eda_meta = _load_eda_findings(PROJECT_ROOT, selection_cfg)
    selection_effective, eda_pre_drop = _apply_eda_selection_priors(
        selection_cfg, eda_findings, eda_policy
    )
    tsfresh_mode = _tsfresh_mode(tsfresh_cfg)
    tsfresh_label_strategy = str(task_directives.get("tsfresh_label_strategy", "any_fault"))
    predrop_before_featurization = bool(
        selection_effective.get("eda_apply_predrop_before_feature_generation", True)
    )
    block_derived_from_predrop = bool(
        selection_effective.get("eda_block_derived_from_predropped_sources", True)
    )

    eda_pre_drop_original = list(eda_pre_drop)
    eda_pre_drop, eda_predrop_protected_sources = _protect_predrop_sources_for_enabled_derived(
        eda_pre_drop,
        flags,
    )
    resolved_for_fingerprint = {
        "task": task,
        "split_path": args.split_path,
        "profile": profile_key,
        "flags": flags,
        "task_directives": task_directives,
        "selection": {
            "enable_hygiene_pruning": bool(flags.get("enable_hygiene_pruning", False)),
            "enable_mrmr_selection": bool(flags.get("enable_mrmr_selection", False)),
            "mrmr_k": int(flags.get("mrmr_k", 64)),
            "near_constant_std": float(selection_effective.get("near_constant_std", 1e-10)),
            "corr_threshold": float(selection_effective.get("corr_threshold", 0.95)),
            "vif_threshold": float(selection_effective.get("vif_threshold", 10.0)),
            "max_vif_rows": int(selection_effective.get("max_vif_rows", 50000)),
            "anchor_features": selection_effective.get("anchor_features", []),
            "use_eda_findings": bool(selection_effective.get("use_eda_findings", False)),
            "eda_findings_path": selection_effective.get("eda_findings_path"),
            "eda_prefer_anchors_from_findings": bool(
                selection_effective.get("eda_prefer_anchors_from_findings", True)
            ),
            "eda_pre_drop_candidates": bool(
                selection_effective.get("eda_pre_drop_candidates", True)
            ),
            "eda_pre_drop_original": eda_pre_drop_original,
            "eda_pre_drop_after_source_protection": eda_pre_drop,
            "eda_pre_drop_removed_due_to_derived_sources": eda_predrop_protected_sources,
            "eda_override_thresholds": bool(
                selection_effective.get("eda_override_thresholds", False)
            ),
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
    base_cols = get_dataset_base_feature_columns(config, args.dataset, split_frames["train"])

    generated_cols: dict[str, list[str]] = {}
    for subset in ("train", "val", "test"):
        split_frames[subset], added = add_optional_features(split_frames[subset], generation_flags)
        generated_cols[subset] = added

    windowing_meta: dict = {"enabled": False}
    if flags.get("enable_explicit_windowing", False):
        _window_sizes: list[int] = (
            [int(w) for w in flags.get("multiscale_window_sizes", [flags.get("window_size", 60)])]
            if flags.get("enable_multiscale", False)
            else [int(flags.get("window_size", 60))]
        )
        _window_step = int(flags.get("window_step", 30))
        _fs = float(
            config.get("paths", {})
            .get("datasets", {})
            .get(args.dataset, {})
            .get("sampling_hz", 1.0)
        )

        _window_input_cols = sorted(
            set(base_cols + generated_cols["train"]) & set(split_frames["train"].columns)
        )

        _windowed_counts: dict[str, int] = {}
        _windowed_col_names: list[str] = []
        for subset in ("train", "val", "test"):
            windowed_df, windowed_cols = add_multiscale_window_features(
                split_frames[subset],
                feature_cols=_window_input_cols,
                window_sizes=_window_sizes,
                step=_window_step,
                label_col=label_col,
                segment_col="segment_id",
                fs=_fs,
                enable_fft=bool(flags.get("enable_spectral", False)),
                n_top_freqs=int(flags.get("spectral_n_top_freqs", 5)),
                enable_psd=bool(flags.get("enable_psd", False)),
                psd_n_bands=int(flags.get("psd_n_bands", 5)),
                enable_wpd=bool(flags.get("enable_wpd", False)),
                wpd_level=int(flags.get("wpd_level", 3)),
                wpd_wavelet=str(flags.get("wpd_wavelet", "db4")),
                enable_ceemdan=bool(flags.get("enable_ceemdan", False)),
                ceemdan_n_imfs=int(flags.get("ceemdan_n_imfs", 3)),
                ceemdan_n_ensemble=int(flags.get("ceemdan_n_ensemble", 20)),
                spectral_min_fs=float(flags.get("spectral_min_fs", 0.5)),
            )
            split_frames[subset] = windowed_df
            _windowed_counts[subset] = len(windowed_df)
            if subset == "train":
                _windowed_col_names = windowed_cols
            logger.info(f"Windowed {subset}: {_windowed_counts[subset]:,} windows")

        windowing_meta = {
            "enabled": True,
            "window_sizes": _window_sizes,
            "window_step": _window_step,
            "primary_window": max(_window_sizes),
            "multiscale": bool(flags.get("enable_multiscale", False)),
            "sampling_hz": _fs,
            "spectral": {
                "fft_top_n": bool(flags.get("enable_spectral", False)),
                "psd": bool(flags.get("enable_psd", False)),
                "wpd": bool(flags.get("enable_wpd", False)),
                "ceemdan": bool(flags.get("enable_ceemdan", False)),
            },
            "n_windows_per_split": _windowed_counts,
        }

        # Rebuild candidate pool from windowed output; original base_cols no longer exist.
        base_cols = []
        generated_cols = {s: _windowed_col_names for s in ("train", "val", "test")}

    candidate_features = sorted(
        set(base_cols + generated_cols["train"] + generated_cols["val"] + generated_cols["test"])
    )
    candidate_features = [c for c in candidate_features if c in split_frames["train"].columns]
    candidate_predrop_applied = [c for c in eda_pre_drop if c in set(candidate_features)]
    if eda_pre_drop:
        candidate_features = [c for c in candidate_features if c not in set(eda_pre_drop)]

    corr_dropped: list[dict] = []
    vif_dropped: list[dict] = []
    hygiene_dropped: list[dict] = []
    mrmr_dropped: list[dict] = []
    selected_features = candidate_features.copy()

    if flags.get("enable_hygiene_pruning", False) and selected_features:
        (
            split_frames["train"],
            split_frames["val"],
            split_frames["test"],
            selected_features,
            hygiene_dropped,
        ) = apply_hygiene_pruning(
            split_frames["train"],
            split_frames["val"],
            split_frames["test"],
            selected_features,
            near_constant_std=float(selection_effective.get("near_constant_std", 1e-10)),
        )

    if flags.get("enable_mrmr_selection", False) and selected_features:
        (
            split_frames["train"],
            split_frames["val"],
            split_frames["test"],
            selected_features,
            mrmr_dropped,
        ) = apply_mrmr_selection(
            split_frames["train"],
            split_frames["val"],
            split_frames["test"],
            selected_features,
            label_col=label_col,
            k=int(flags.get("mrmr_k", 64)),
        )

    if flags.get("enable_corr_pruning", False) and selected_features:
        (
            split_frames["train"],
            split_frames["val"],
            split_frames["test"],
            selected_features,
            corr_dropped,
        ) = apply_correlation_pruning(
            split_frames["train"],
            split_frames["val"],
            split_frames["test"],
            selected_features,
            threshold=float(selection_effective.get("corr_threshold", 0.95)),
            anchor_cols=selection_effective.get("anchor_features", []),
        )

    if flags.get("enable_vif_pruning", False) and selected_features:
        (
            split_frames["train"],
            split_frames["val"],
            split_frames["test"],
            selected_features,
            vif_dropped,
        ) = apply_vif_pruning(
            split_frames["train"],
            split_frames["val"],
            split_frames["test"],
            selected_features,
            threshold=float(selection_effective.get("vif_threshold", 10.0)),
            anchor_cols=selection_effective.get("anchor_features", []),
            max_rows=int(selection_effective.get("max_vif_rows", 50000)),
        )

    tsfresh_top_k = int(tsfresh_cfg.get("top_k", 20))
    tsfresh_cols: list[str] = []
    tsfresh_meta: dict = {"mode": tsfresh_mode, "selected": 0}

    if tsfresh_mode != "off":
        (
            split_frames["train"],
            split_frames["val"],
            split_frames["test"],
            tsfresh_cols,
            tsfresh_meta,
        ) = extract_tsfresh_segment_features(
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

    final_feature_cols = []
    for col in selected_features + tsfresh_cols:
        if col in split_frames["train"].columns and col not in final_feature_cols:
            final_feature_cols.append(col)

    meta_cols = [
        c
        for c in ("timestamp", "segment_id", "label", "Fault")
        if c in split_frames["train"].columns
    ]
    if label_col not in meta_cols and label_col in split_frames["train"].columns:
        meta_cols.append(label_col)

    for subset in ("train", "val", "test"):
        out_df = split_frames[subset][meta_cols + final_feature_cols].copy()
        out_path = run_dir / f"{subset}.parquet"
        out_df.to_parquet(out_path, index=False)
        logger.info(
            f"Saved {subset}: {out_path} ({len(out_df):,} rows, {len(final_feature_cols)} features)"
        )

    manifest = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile_name or "default",
        "config_fingerprint": config_fingerprint,
        "output_directory": str(run_dir),
        "source_task": task,
        "split_path": args.split_path,
        "source_preprocessed_dir": str(input_dir),
        "label_column": label_col,
        "active_flags": flags,
        "effective_generation_flags": generation_flags,
        "task_directives_effective": task_directives,
        "selection": {
            "enable_hygiene_pruning": bool(flags.get("enable_hygiene_pruning", False)),
            "enable_mrmr_selection": bool(flags.get("enable_mrmr_selection", False)),
            "mrmr_k": int(flags.get("mrmr_k", 64)),
            "near_constant_std": float(selection_effective.get("near_constant_std", 1e-10)),
            "corr_threshold": float(selection_effective.get("corr_threshold", 0.95)),
            "vif_threshold": float(selection_effective.get("vif_threshold", 10.0)),
            "anchor_features": selection_effective.get("anchor_features", []),
            "use_eda_findings": bool(selection_effective.get("use_eda_findings", False)),
            "eda_policy": eda_policy,
            "eda_pre_drop_original": eda_pre_drop_original,
            "eda_pre_drop_candidates": eda_pre_drop,
            "eda_pre_drop_removed_due_to_derived_sources": eda_predrop_protected_sources,
            "eda_predrop_before_featurization_enabled": predrop_before_featurization,
            "eda_pre_drop_applied_before_featurization": eda_predrop_applied_before_fe,
            "eda_pre_drop_unavailable_before_featurization": eda_predrop_unavailable_before_fe,
            "eda_pre_drop_applied_candidate_filter": candidate_predrop_applied,
            "derived_blocking": {
                "enabled": block_derived_from_predrop,
                "blocked_features": derived_blocked_by_predrop,
            },
            "hygiene_dropped": hygiene_dropped,
            "mrmr_dropped": mrmr_dropped,
            "corr_dropped": corr_dropped,
            "vif_dropped": vif_dropped,
        },
        "eda_findings": eda_meta,
        "tsfresh": tsfresh_meta,
        "explicit_windowing": windowing_meta,
        "base_feature_count": len(base_cols),
        "final_feature_count": len(final_feature_cols),
        "final_features": final_feature_cols,
    }

    manifest_path = run_dir / "features_manifest.json"
    manifest = to_json_safe(manifest)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)

    resolved_path = run_dir / "resolved_config.json"
    resolved_base = to_json_safe(resolved_for_fingerprint)
    if not isinstance(resolved_base, dict):
        raise TypeError("Resolved feature config must serialize to a mapping")

    resolved_payload = {
        **resolved_base,
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
    if "latest_by_task" not in latest_by_task or not isinstance(
        latest_by_task.get("latest_by_task"), dict
    ):
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
    latest_by_task_path.write_text(
        json.dumps(to_json_safe(latest_by_task), indent=2), encoding="utf-8"
    )

    logger.success(f"Feature engineering complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
