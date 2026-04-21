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

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _to_json_safe(value):
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return float(1.0 - (2.0 * u_stat) / (n1 * n2))


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
        a = normal[col].dropna()
        b = fault[col].dropna()
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
    significant = [r["feature"] for r in rows if r["p_value"] < 0.05]
    return {
        "min_group_size": int(min_group_size),
        "results": rows,
        "significant_features": significant,
    }


def _compute_spearman(
    df: pd.DataFrame,
    feature_cols: list[str],
    corr_threshold: float,
) -> dict:
    available = [c for c in feature_cols if c in df.columns]
    corr = df[available].corr(method="spearman")
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

    scaled = pd.DataFrame(StandardScaler().fit_transform(data), columns=available)
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

    x = data[available].values
    y_bin = (data[label_col].values != 0.0).astype(int)
    y_multi = data[label_col].values.astype(int)

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
) -> tuple[dict, dict]:
    """
    Compute and persist EDA findings JSON files.

    Returns:
      findings dict and a dict of file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    findings_mw = _compute_mannwhitney(pdf_complete, feature_cols, label_col)
    findings_sp = _compute_spearman(pdf_complete, feature_cols, corr_threshold)
    findings_vif = _compute_vif(pdf_complete, feature_cols, vif_threshold)
    findings_mi = _compute_mutual_info(pdf_complete, feature_cols, label_col)

    recommendations = {
        "redundant_drop_candidates": _derive_drop_candidates(findings_sp, findings_mi),
        "vif_drop_candidates": findings_vif.get("high_vif_features", []),
    }

    consolidated = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "label_column": label_col,
        "feature_columns": list(feature_cols),
        "mannwhitney": findings_mw,
        "spearman": findings_sp,
        "vif": findings_vif,
        "mutual_information": findings_mi,
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

    return consolidated, {k: str(v) for k, v in files.items()}
