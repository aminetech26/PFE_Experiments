"""
Export structured EDA findings for downstream pipeline reuse.

This module computes and persists findings for:
1. Mann-Whitney (normal vs fault)
2. Spearman correlation
3. VIF collinearity
4. Mutual information (binary + multiclass)
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _to_json_safe(value):
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return float(1.0 - (2.0 * u_stat) / (n1 * n2))


def _apply_fdr(rows: list[dict], *, p_key: str = "p_value", alpha: float = 0.05) -> list[dict]:
    if not rows:
        return rows

    valid_indices = [idx for idx, row in enumerate(rows) if row.get(p_key) is not None]
    if not valid_indices:
        return rows

    pvals = [float(rows[idx][p_key]) for idx in valid_indices]
    reject, qvals, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")

    for idx, is_sig, qval in zip(valid_indices, list(reject), list(qvals)):
        rows[idx][f"{p_key}_fdr"] = float(qval)
        rows[idx]["significant_fdr"] = bool(is_sig)

    return rows


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    return cast(pd.Series, df.loc[:, col])


def _compute_mannwhitney(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    min_group_size: int = 10,
) -> dict:
    normal = df[df[label_col] == 0.0]
    fault = df[df[label_col] != 0.0]

    rows: list[dict] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        a = _series(normal, col).dropna()
        b = _series(fault, col).dropna()
        if len(a) < min_group_size or len(b) < min_group_size:
            continue

        stat, pval = stats.mannwhitneyu(a.values, b.values, alternative="two-sided")
        rows.append(
            {
                "feature": col,
                "u_statistic": float(stat),
                "p_value": float(pval),
                "n_normal": int(len(a)),
                "n_fault": int(len(b)),
                "rank_biserial": _rank_biserial_from_u(float(stat), len(a), len(b)),
                "normal_mean": float(a.mean()),
                "fault_mean": float(b.mean()),
                "normal_std": float(a.std(ddof=1)),
                "fault_std": float(b.std(ddof=1)),
            }
        )

    rows = sorted(rows, key=lambda x: x["p_value"])
    rows = _apply_fdr(rows)
    significant = [r["feature"] for r in rows if r["p_value"] < 0.05]
    significant_fdr = [r["feature"] for r in rows if r.get("significant_fdr", False)]
    return {
        "min_group_size": int(min_group_size),
        "results": rows,
        "significant_features": significant,
        "significant_features_fdr": significant_fdr,
    }


def _compute_normal_vs_each_fault(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    min_group_size: int = 10,
) -> dict:
    normal = df[df[label_col] == 0.0]
    fault_labels = sorted(lbl for lbl in df[label_col].dropna().unique() if float(lbl) != 0.0)

    per_class: dict[str, dict] = {}
    for fault_label in fault_labels:
        fault = df[df[label_col] == fault_label]
        rows: list[dict] = []

        for col in feature_cols:
            if col not in df.columns:
                continue
            a = _series(normal, col).dropna()
            b = _series(fault, col).dropna()
            if len(a) < min_group_size or len(b) < min_group_size:
                continue

            stat, pval = stats.mannwhitneyu(a.values, b.values, alternative="two-sided")
            rows.append(
                {
                    "feature": col,
                    "fault_label": float(fault_label),
                    "u_statistic": float(stat),
                    "p_value": float(pval),
                    "n_normal": int(len(a)),
                    "n_fault": int(len(b)),
                    "rank_biserial": _rank_biserial_from_u(float(stat), len(a), len(b)),
                    "normal_mean": float(a.mean()),
                    "fault_mean": float(b.mean()),
                    "normal_std": float(a.std(ddof=1)),
                    "fault_std": float(b.std(ddof=1)),
                }
            )

        rows = sorted(rows, key=lambda x: x["p_value"])
        rows = _apply_fdr(rows)
        per_class[str(fault_label)] = {
            "fault_label": float(fault_label),
            "results": rows,
            "significant_features": [r["feature"] for r in rows if r["p_value"] < 0.05],
            "significant_features_fdr": [
                r["feature"] for r in rows if r.get("significant_fdr", False)
            ],
        }

    return {
        "min_group_size": int(min_group_size),
        "fault_labels": [float(lbl) for lbl in fault_labels],
        "per_class": per_class,
    }


def _kruskal_effect_size(h_stat: float, n_total: int, n_groups: int) -> float | None:
    if n_total <= n_groups or n_groups <= 1:
        return None
    return float(max(0.0, (h_stat - n_groups + 1) / (n_total - n_groups)))


def _compute_kruskal_wallis(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    min_group_size: int = 10,
) -> dict:
    labels = sorted(_series(df, label_col).dropna().unique())
    rows: list[dict] = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        groups = []
        group_sizes = {}
        for label in labels:
            values = _series(df.loc[df[label_col] == label], col).dropna().to_numpy()
            if len(values) >= min_group_size:
                groups.append(values)
                group_sizes[str(float(label))] = int(len(values))

        if len(groups) < 2:
            continue

        h_stat, pval = stats.kruskal(*groups)
        rows.append(
            {
                "feature": col,
                "h_statistic": float(h_stat),
                "p_value": float(pval),
                "n_groups": int(len(groups)),
                "group_sizes": group_sizes,
                "epsilon_squared": _kruskal_effect_size(
                    float(h_stat), sum(group_sizes.values()), len(groups)
                ),
            }
        )

    rows = sorted(rows, key=lambda x: x["p_value"])
    rows = _apply_fdr(rows)
    return {
        "min_group_size": int(min_group_size),
        "results": rows,
        "significant_features": [r["feature"] for r in rows if r["p_value"] < 0.05],
        "significant_features_fdr": [r["feature"] for r in rows if r.get("significant_fdr", False)],
    }


def _compute_spearman(
    df: pd.DataFrame,
    feature_cols: list[str],
    corr_threshold: float,
) -> dict:
    available = [c for c in feature_cols if c in df.columns]
    corr = cast(pd.DataFrame, df.loc[:, available].corr(method="spearman"))
    abs_corr = corr.abs()

    pairs = []
    for i, a in enumerate(available):
        for b in available[i + 1 :]:
            rho = abs_corr.loc[a, b]
            if pd.isna(rho):
                continue
            if float(rho) >= corr_threshold:
                pairs.append({"feature_a": a, "feature_b": b, "abs_rho": float(rho)})

    pairs = sorted(pairs, key=lambda x: x["abs_rho"], reverse=True)
    return {
        "recommended_corr_threshold": float(corr_threshold),
        "high_corr_pairs": pairs,
        "matrix": corr.to_dict(),
    }


def _compute_regime_binned_spearman(
    df: pd.DataFrame,
    feature_cols: list[str],
    regime_col: str = "irr",
    corr_threshold: float = 0.95,
    n_bins: int = 4,
) -> dict:
    if regime_col not in df.columns:
        return {"error": f"{regime_col} not found"}

    normal_only = cast(pd.DataFrame, df.loc[df["label"] == 0.0].copy())
    if normal_only.empty:
        return {"error": "no normal rows available"}

    regime_values = _series(normal_only, regime_col).dropna()
    if regime_values.nunique() < n_bins:
        return {"error": f"insufficient unique {regime_col} values for binning"}

    normal_only = normal_only.loc[regime_values.index].copy()
    normal_only["regime_bin"] = pd.qcut(
        regime_values,
        q=n_bins,
        duplicates="drop",
    )

    bins = []
    for bin_label, bin_df in normal_only.groupby("regime_bin", observed=True, sort=True):
        if len(bin_df) < 25:
            continue
        findings = _compute_spearman(bin_df, feature_cols, corr_threshold)
        bins.append(
            {
                "bin_label": str(bin_label),
                "n_rows": int(len(bin_df)),
                "regime_min": float(_series(bin_df, regime_col).min()),
                "regime_max": float(_series(bin_df, regime_col).max()),
                "high_corr_pairs": findings["high_corr_pairs"],
                "matrix": findings["matrix"],
            }
        )

    focus_pairs = [("pdc1", "pdc2"), ("pdc1", "pdc"), ("pdc2", "pdc"), ("idc1", "idc2")]
    pair_profiles = []
    for a, b in focus_pairs:
        values = []
        for bin_result in bins:
            matrix = bin_result.get("matrix", {})
            rho = matrix.get(a, {}).get(b)
            if rho is None:
                continue
            values.append(
                {
                    "bin_label": bin_result["bin_label"],
                    "regime_min": bin_result["regime_min"],
                    "regime_max": bin_result["regime_max"],
                    "rho": float(rho),
                }
            )
        if values:
            pair_profiles.append({"feature_a": a, "feature_b": b, "by_bin": values})

    return {
        "regime_column": regime_col,
        "binning_strategy": "normal_only_quantile_bins",
        "n_bins_requested": int(n_bins),
        "n_bins_realized": int(len(bins)),
        "bins": bins,
        "focus_pair_profiles": pair_profiles,
    }


def _compute_segment_medians(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    segment_col: str,
) -> pd.DataFrame:
    available = [c for c in feature_cols if c in df.columns]
    if segment_col not in df.columns or label_col not in df.columns or not available:
        return pd.DataFrame()

    agg = {label_col: "first", **{col: "median" for col in available}}
    grouped = df.groupby(segment_col, observed=True).agg(agg).reset_index(drop=False)
    return grouped


def _compute_vif(
    df: pd.DataFrame,
    feature_cols: list[str],
    vif_threshold: float,
    max_rows: int = 50000,
) -> dict:
    available = [c for c in feature_cols if c in df.columns]
    data = df[available].dropna()
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)

    if data.empty:
        return {
            "max_rows": int(max_rows),
            "recommended_vif_threshold": float(vif_threshold),
            "results": [],
            "high_vif_features": [],
        }

    scaled = pd.DataFrame(StandardScaler().fit_transform(data), index=data.index, columns=available)
    results = []
    for i, col in enumerate(available):
        try:
            vif_val = float(variance_inflation_factor(scaled.values, i))
        except Exception:
            vif_val = float("inf")
        if not np.isfinite(vif_val):
            vif_val = float("inf")
        results.append({"feature": col, "vif": vif_val})

    results = sorted(results, key=lambda x: x["vif"], reverse=True)
    high = [r["feature"] for r in results if r["vif"] > vif_threshold]
    return {
        "max_rows": int(max_rows),
        "recommended_vif_threshold": float(vif_threshold),
        "results": results,
        "high_vif_features": high,
    }


def _compute_mutual_info(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    max_rows: int = 100000,
) -> dict:
    available = [c for c in feature_cols if c in df.columns]
    data = df[available + [label_col]].dropna()
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)

    if data.empty:
        return {
            "max_rows": int(max_rows),
            "results": [],
            "top_features_binary": [],
            "top_features_multiclass": [],
        }

    x = cast(pd.DataFrame, data.loc[:, available]).to_numpy()
    y_series = _series(data, label_col)
    y_bin = (y_series.to_numpy() != 0.0).astype(int)
    y_multi = y_series.to_numpy().astype(int)

    mi_bin = mutual_info_classif(x, y_bin, random_state=42, n_neighbors=5)
    mi_multi = mutual_info_classif(x, y_multi, random_state=42, n_neighbors=5)

    rows = [
        {
            "feature": f,
            "mi_binary_anomaly": float(b),
            "mi_multiclass_fault": float(m),
        }
        for f, b, m in zip(available, mi_bin, mi_multi)
    ]

    by_bin = sorted(rows, key=lambda r: r["mi_binary_anomaly"], reverse=True)
    by_multi = sorted(rows, key=lambda r: r["mi_multiclass_fault"], reverse=True)
    return {
        "max_rows": int(max_rows),
        "results": rows,
        "top_features_binary": [r["feature"] for r in by_bin[:20]],
        "top_features_multiclass": [r["feature"] for r in by_multi[:20]],
    }


def _derive_drop_candidates(spearman: dict, mutual_info: dict) -> list[str]:
    mi_map = {
        r["feature"]: r["mi_binary_anomaly"]
        for r in mutual_info.get("results", [])
        if "feature" in r
    }
    drops: list[str] = []

    for pair in spearman.get("high_corr_pairs", []):
        a = pair["feature_a"]
        b = pair["feature_b"]
        mi_a = mi_map.get(a, 0.0)
        mi_b = mi_map.get(b, 0.0)
        drop = a if mi_a < mi_b else b
        if drop not in drops:
            drops.append(drop)

    return drops


def export_eda_feature_findings(
    pdf_complete: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    dataset: str,
    output_dir: Path,
    corr_threshold: float = 0.95,
    vif_threshold: float = 10.0,
    segment_col: str | None = None,
) -> tuple[dict, dict]:
    """
    Compute and persist EDA findings JSON files.

    Returns:
      findings dict and a dict of file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "EDA findings: starting feature statistics | dataset={} | features={} | rows={:,}",
        dataset,
        len(feature_cols),
        len(pdf_complete),
    )

    logger.info("EDA findings: Mann-Whitney normal vs fault ...")
    findings_mw = _compute_mannwhitney(pdf_complete, feature_cols, label_col)

    logger.info("EDA findings: Spearman correlations (all rows) ...")
    findings_sp_all = _compute_spearman(pdf_complete, feature_cols, corr_threshold)
    normal_only = cast(pd.DataFrame, pdf_complete.loc[pdf_complete[label_col] == 0.0].copy())

    logger.info("EDA findings: Spearman correlations (normal only) ...")
    findings_sp_normal = _compute_spearman(normal_only, feature_cols, corr_threshold)
    logger.info("EDA findings: Spearman correlations (normal-only irradiance bins) ...")
    findings_sp_regime = _compute_regime_binned_spearman(
        pdf_complete,
        feature_cols,
        regime_col="irr",
        corr_threshold=corr_threshold,
        n_bins=4,
    )
    findings_sp = {
        **findings_sp_all,
        "contexts": {
            "all_rows": findings_sp_all,
            "normal_only": findings_sp_normal,
            "normal_only_irr_bins": findings_sp_regime,
        },
    }

    logger.info("EDA findings: VIF collinearity scan ...")
    findings_vif = _compute_vif(pdf_complete, feature_cols, vif_threshold)

    logger.info("EDA findings: Mutual information (binary + multiclass) ...")
    findings_mi = _compute_mutual_info(pdf_complete, feature_cols, label_col)

    logger.info("EDA findings: per-class normal-vs-fault tests ...")
    findings_per_class = _compute_normal_vs_each_fault(pdf_complete, feature_cols, label_col)

    logger.info("EDA findings: Kruskal-Wallis multiclass tests ...")
    findings_kw = _compute_kruskal_wallis(pdf_complete, feature_cols, label_col)

    segment_findings = None
    if segment_col and segment_col in pdf_complete.columns:
        logger.info("EDA findings: segment-aware aggregation via '{}' ...", segment_col)
        segment_frame = _compute_segment_medians(pdf_complete, feature_cols, label_col, segment_col)
        if not segment_frame.empty:
            logger.info(
                "EDA findings: segment-aware tests on {:,} grouped rows ...",
                len(segment_frame),
            )
            segment_findings = {
                "segment_column": segment_col,
                "n_segments": int(segment_frame[segment_col].nunique()),
                "aggregation": "median_per_segment",
                "mannwhitney_normal_vs_fault": _compute_mannwhitney(
                    segment_frame, feature_cols, label_col, min_group_size=3
                ),
                "normal_vs_each_fault": _compute_normal_vs_each_fault(
                    segment_frame, feature_cols, label_col, min_group_size=3
                ),
                "kruskal_wallis": _compute_kruskal_wallis(
                    segment_frame, feature_cols, label_col, min_group_size=3
                ),
            }

    recommendations = {
        "redundant_drop_candidates": _derive_drop_candidates(findings_sp_normal, findings_mi),
        "vif_drop_candidates": findings_vif.get("high_vif_features", []),
    }

    consolidated = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "label_column": label_col,
        "feature_columns": list(feature_cols),
        "mannwhitney": findings_mw,
        "normal_vs_each_fault": findings_per_class,
        "kruskal_wallis": findings_kw,
        "spearman": findings_sp,
        "vif": findings_vif,
        "mutual_information": findings_mi,
        "segment_aware": segment_findings,
        "recommendations": recommendations,
    }

    files = {
        "mannwhitney": output_dir / "eda_mannwhitney_findings.json",
        "spearman": output_dir / "eda_spearman_findings.json",
        "vif": output_dir / "eda_vif_findings.json",
        "mutual_information": output_dir / "eda_mutual_info_findings.json",
        "consolidated": output_dir / "eda_feature_findings.json",
    }

    payloads = {
        files["mannwhitney"]: findings_mw,
        files["spearman"]: findings_sp,
        files["vif"]: findings_vif,
        files["mutual_information"]: findings_mi,
        files["consolidated"]: consolidated,
    }

    for path, payload in payloads.items():
        safe_payload = _to_json_safe(payload)
        path.write_text(
            json.dumps(safe_payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    logger.info("EDA findings: wrote {} artifact files", len(payloads))

    return consolidated, {k: str(v) for k, v in files.items()}
