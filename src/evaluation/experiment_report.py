"""
Automated comparison report — shared by anomaly and classification suites.

Three axes:
  - between_profiles: for each (model, split), compare all profile pairs
  - inter_family:     for each (profile, split), compare all model pairs
  - inter_split:      for each (model, profile), compare split_path variants

Wilcoxon + Cohen's d are only attempted when both groups have ≥2 observations.
Smaller groups fall back to descriptive-only — the first run never crashes.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.statistical_tests import cohen_d, format_metric_report, wilcoxon_test


def _load(records_path: Path, metric: str) -> pd.DataFrame:
    records = []
    with records_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)

    def _model_key(row: pd.Series) -> str:
        if row.get("model") == "one_class_svm" and row.get("kernel"):
            return f"ocsvm_{row['kernel']}"
        return str(row.get("model", "unknown"))

    df["model_key"] = df.apply(_model_key, axis=1)
    if metric in df.columns:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def _compare_pair(
    df: pd.DataFrame,
    group_col: str,
    val_a: str,
    val_b: str,
    metric: str,
) -> dict:
    scores_a = df[df[group_col].astype(str) == val_a][metric].dropna().tolist()
    scores_b = df[df[group_col].astype(str) == val_b][metric].dropna().tolist()

    result: dict = {
        "group_a": val_a,
        "group_b": val_b,
        "n_a": len(scores_a),
        "n_b": len(scores_b),
        "mean_a": float(np.mean(scores_a)) if scores_a else None,
        "mean_b": float(np.mean(scores_b)) if scores_b else None,
    }
    if scores_a:
        result["report_a"] = format_metric_report(scores_a, metric_name=f"{metric} ({val_a})")
    if scores_b:
        result["report_b"] = format_metric_report(scores_b, metric_name=f"{metric} ({val_b})")

    pair_n = min(len(scores_a), len(scores_b))
    if pair_n < 2:
        result["stat_test"] = "skipped"
        result["reason"] = f"insufficient paired samples ({pair_n}<2)"
        return result

    paired_a = scores_a[-pair_n:]
    paired_b = scores_b[-pair_n:]
    try:
        w = wilcoxon_test(paired_a, paired_b, model_a_name=val_a, model_b_name=val_b)
        result.update({
            "wilcoxon_stat": w["statistic"],
            "wilcoxon_p": w["p_value"],
            "significant": w["significant"],
            "better": w["better_model"],
            "effect_cohen_d": cohen_d(paired_a, paired_b),
            "stat_test": "wilcoxon",
        })
    except Exception as exc:
        result["stat_test"] = "error"
        result["error"] = str(exc)

    return result


def run_comparison_report(
    records_path: Path,
    output_dir: Path,
    models: list[str],
    profiles: list[str],
    splits: list[str],
    seeds: list[int],
    metric: str = "test_pr_auc",
) -> None:
    if not records_path.exists() or records_path.stat().st_size == 0:
        logger.warning("Records file empty — no comparison report generated")
        return

    df = _load(records_path, metric)
    if df.empty or metric not in df.columns:
        logger.warning("No valid records with '{}' — skipping report", metric)
        return

    all_model_keys = df["model_key"].dropna().unique().tolist()
    all_profiles   = df["feature_profile"].dropna().unique().tolist()
    all_splits     = df["split_path"].dropna().unique().tolist()

    logger.info(
        "Comparison report | metric={} models={} profiles={} splits={} records={}",
        metric, all_model_keys, all_profiles, all_splits, len(df),
    )

    report: dict = {
        "total_records": len(df),
        "metric": metric,
        "models": all_model_keys,
        "profiles": all_profiles,
        "splits": all_splits,
        "between_profiles": {},
        "inter_family": {},
        "inter_split": {},
    }

    # 1. between_profiles
    for model_key in all_model_keys:
        for split in all_splits:
            sub = df[(df["model_key"] == model_key) & (df["split_path"] == split)]
            available = sub["feature_profile"].dropna().unique().tolist()
            pairs = [
                _compare_pair(sub, "feature_profile", pa, pb, metric)
                for pa, pb in combinations(available, 2)
            ]
            if pairs:
                report["between_profiles"][f"{model_key}|{split}"] = pairs

    # 2. inter_family
    for profile in all_profiles:
        for split in all_splits:
            sub = df[(df["feature_profile"] == profile) & (df["split_path"] == split)]
            available = sub["model_key"].dropna().unique().tolist()
            pairs = [
                _compare_pair(sub, "model_key", ma, mb, metric)
                for ma, mb in combinations(available, 2)
            ]
            if pairs:
                report["inter_family"][f"{profile}|{split}"] = pairs

    # 3. inter_split
    for model_key in all_model_keys:
        for profile in all_profiles:
            sub = df[(df["model_key"] == model_key) & (df["feature_profile"] == profile)]
            available = sub["split_path"].dropna().unique().tolist()
            if len(available) >= 2:
                pairs = [
                    _compare_pair(sub, "split_path", sa, sb, metric)
                    for sa, sb in combinations(available, 2)
                ]
                if pairs:
                    report["inter_split"][f"{model_key}|{profile}"] = pairs

    out_path = output_dir / "comparison_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.success("Comparison report saved → {}", out_path)
    _log_summary(report)


def _log_summary(report: dict) -> None:
    logger.info("── Comparison summary ──────────────────────────────────────")
    for axis in ("between_profiles", "inter_family", "inter_split"):
        results = report.get(axis, {})
        sig   = sum(1 for pairs in results.values() for p in pairs if p.get("significant"))
        total = sum(len(pairs) for pairs in results.values())
        logger.info("  {:20s} — {}/{} pairs significant", axis, sig, total)
