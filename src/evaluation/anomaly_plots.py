"""
Experiment result plots — shared by anomaly and classification suites.

All functions guard against empty/insufficient data and log a warning instead
of raising. Pass metric= to switch between test_pr_auc (anomaly) and
test_f1_weighted (classification).

Plots saved to output_dir/plots/:
  profile_ablation.png, inter_family.png, inter_split.png,
  seed_robustness.png, heatmap.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from loguru import logger

_DEFAULT_METRIC = "test_pr_auc"
_METRIC_LABELS: dict[str, str] = {
    "test_pr_auc":      "Test PR-AUC",
    "test_f1_weighted": "Test F1-weighted",
    "test_roc_auc":     "Test ROC-AUC",
}
_PROFILE_ORDER = [
    "baseline_raw", "plus_physics", "plus_rolling",
    "plus_wpd", "plus_ceemdan", "plus_wpd_ceemdan",
]


# ── data helpers ──────────────────────────────────────────────────────────────

def _load(records_path: Path, metric: str) -> pd.DataFrame:
    if not records_path.exists():
        return pd.DataFrame()
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
    if metric not in df.columns:
        return pd.DataFrame()

    def _model_key(row: pd.Series) -> str:
        if row.get("model") == "one_class_svm" and row.get("kernel"):
            return f"ocsvm_{row['kernel']}"
        return str(row.get("model", "unknown"))

    df["model_key"] = df.apply(_model_key, axis=1)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df.dropna(subset=[metric])


def _agg(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    return (
        df.groupby(group_cols)[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )


def _profile_order(profiles: list[str]) -> list[str]:
    ordered = [p for p in _PROFILE_ORDER if p in profiles]
    ordered += [p for p in profiles if p not in ordered]
    return ordered


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success("Plot saved → {}", path)


def _bar_group(
    ax: plt.Axes,
    groups: list[str],
    x_labels: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    title: str,
    ylabel: str,
) -> None:
    n_groups = len(groups)
    width = 0.8 / max(n_groups, 1)
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width
    x = np.arange(len(x_labels))
    for i, (grp, off) in enumerate(zip(groups, offsets)):
        ax.bar(x + off, means[i], width, yerr=stds[i], capsize=4, label=grp, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0] - 0.02))


# ── individual plots ──────────────────────────────────────────────────────────

def plot_profile_ablation(df: pd.DataFrame, output_dir: Path, metric: str, ylabel: str) -> None:
    if df.empty:
        logger.warning("profile_ablation: no data")
        return
    splits   = sorted(df["split_path"].dropna().unique())
    models   = sorted(df["model_key"].unique())
    profiles = _profile_order(df["feature_profile"].dropna().unique().tolist())
    if not profiles or not models:
        logger.warning("profile_ablation: nothing to plot")
        return

    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5), squeeze=False)
    fig.suptitle("Profile Ablation — Feature Family Impact", fontsize=13, fontweight="bold")
    for col, split in enumerate(splits):
        ax = axes[0][col]
        sub = df[df["split_path"] == split]
        agg = _agg(sub, ["model_key", "feature_profile"], metric)
        means = np.zeros((len(models), len(profiles)))
        stds  = np.zeros_like(means)
        for i, m in enumerate(models):
            for j, p in enumerate(profiles):
                row = agg[(agg["model_key"] == m) & (agg["feature_profile"] == p)]
                if not row.empty:
                    means[i, j] = row["mean"].iloc[0]
                    stds[i, j]  = row["std"].fillna(0).iloc[0]
        _bar_group(ax, models, profiles, means, stds, title=f"Split: {split}", ylabel=ylabel)
    fig.tight_layout()
    _save(fig, output_dir / "plots" / "profile_ablation.png")


def plot_inter_family(df: pd.DataFrame, output_dir: Path, metric: str, ylabel: str) -> None:
    if df.empty:
        logger.warning("inter_family: no data")
        return
    splits   = sorted(df["split_path"].dropna().unique())
    profiles = _profile_order(df["feature_profile"].dropna().unique().tolist())
    models   = sorted(df["model_key"].unique())
    if len(models) < 2:
        logger.warning("inter_family: need ≥2 models, found {}", models)
        return

    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5), squeeze=False)
    fig.suptitle("Inter-Family Comparison — Model vs Model", fontsize=13, fontweight="bold")
    for col, split in enumerate(splits):
        ax = axes[0][col]
        sub = df[df["split_path"] == split]
        agg = _agg(sub, ["feature_profile", "model_key"], metric)
        means = np.zeros((len(profiles), len(models)))
        stds  = np.zeros_like(means)
        for i, p in enumerate(profiles):
            for j, m in enumerate(models):
                row = agg[(agg["feature_profile"] == p) & (agg["model_key"] == m)]
                if not row.empty:
                    means[i, j] = row["mean"].iloc[0]
                    stds[i, j]  = row["std"].fillna(0).iloc[0]
        _bar_group(ax, profiles, models, means, stds, title=f"Split: {split}", ylabel=ylabel)
    fig.tight_layout()
    _save(fig, output_dir / "plots" / "inter_family.png")


def plot_inter_split(df: pd.DataFrame, output_dir: Path, metric: str, ylabel: str) -> None:
    if df.empty:
        logger.warning("inter_split: no data")
        return
    splits = sorted(df["split_path"].dropna().unique())
    if len(splits) < 2:
        logger.warning("inter_split: need ≥2 splits, found {}", splits)
        return
    models   = sorted(df["model_key"].unique())
    profiles = _profile_order(df["feature_profile"].dropna().unique().tolist())
    x_labels = [f"{m}\n{p}" for m in models for p in profiles]

    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 1.2), 5))
    fig.suptitle("Inter-Split Comparison — Path A vs Path B", fontsize=13, fontweight="bold")
    agg = _agg(df, ["model_key", "feature_profile", "split_path"], metric)
    means = np.zeros((len(splits), len(x_labels)))
    stds  = np.zeros_like(means)
    for si, split in enumerate(splits):
        for idx, (m, p) in enumerate([(m, p) for m in models for p in profiles]):
            row = agg[
                (agg["model_key"] == m) &
                (agg["feature_profile"] == p) &
                (agg["split_path"] == split)
            ]
            if not row.empty:
                means[si, idx] = row["mean"].iloc[0]
                stds[si, idx]  = row["std"].fillna(0).iloc[0]
    _bar_group(ax, splits, x_labels, means, stds,
               title="Event-centric (Path A) vs Continuity-oriented (Path B)", ylabel=ylabel)
    fig.tight_layout()
    _save(fig, output_dir / "plots" / "inter_split.png")


def plot_seed_robustness(df: pd.DataFrame, output_dir: Path, metric: str, ylabel: str) -> None:
    if df.empty:
        logger.warning("seed_robustness: no data")
        return
    models   = sorted(df["model_key"].unique())
    profiles = _profile_order(df["feature_profile"].dropna().unique().tolist())
    splits   = sorted(df["split_path"].dropna().unique())

    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5), squeeze=False)
    fig.suptitle("Seed Robustness — Score Distribution across Seeds", fontsize=13, fontweight="bold")
    for col, split in enumerate(splits):
        ax = axes[0][col]
        sub = df[df["split_path"] == split]
        data, labels = [], []
        for m in models:
            for p in profiles:
                vals = sub[(sub["model_key"] == m) & (sub["feature_profile"] == p)][metric].tolist()
                if vals:
                    data.append(vals)
                    labels.append(f"{m}\n{p}")
        if not data:
            ax.set_title(f"Split: {split} — no data")
            continue
        bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 2})
        colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Split: {split}", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / "plots" / "seed_robustness.png")


def plot_heatmap(df: pd.DataFrame, output_dir: Path, metric: str, ylabel: str) -> None:
    if df.empty:
        logger.warning("heatmap: no data")
        return
    splits   = sorted(df["split_path"].dropna().unique())
    models   = sorted(df["model_key"].unique())
    profiles = _profile_order(df["feature_profile"].dropna().unique().tolist())
    if not models or not profiles:
        return

    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), max(3, len(models))), squeeze=False)
    fig.suptitle(f"Mean {ylabel} — Model × Profile", fontsize=13, fontweight="bold")
    agg = _agg(df, ["model_key", "feature_profile", "split_path"], metric)
    for col, split in enumerate(splits):
        ax = axes[0][col]
        sub = agg[agg["split_path"] == split]
        matrix = np.full((len(models), len(profiles)), np.nan)
        for i, m in enumerate(models):
            for j, p in enumerate(profiles):
                row = sub[(sub["model_key"] == m) & (sub["feature_profile"] == p)]
                if not row.empty:
                    matrix[i, j] = row["mean"].iloc[0]
        vmin = np.nanmin(matrix) if not np.all(np.isnan(matrix)) else 0.0
        vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1.0
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto",
                       cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(profiles)))
        ax.set_xticklabels(profiles, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_title(f"Split: {split}", fontsize=11)
        span = max(vmax - vmin, 1e-6)
        for i in range(len(models)):
            for j in range(len(profiles)):
                if not np.isnan(matrix[i, j]):
                    rel = (matrix[i, j] - vmin) / span
                    color = "black" if 0.3 < rel < 0.7 else "white"
                    ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                            fontsize=8, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, output_dir / "plots" / "heatmap.png")


# ── entry point ───────────────────────────────────────────────────────────────

def plot_all(
    records_path: Path,
    output_dir: Path,
    metric: str = _DEFAULT_METRIC,
) -> None:
    ylabel = _METRIC_LABELS.get(metric, metric)
    df = _load(records_path, metric)
    if df.empty:
        logger.warning("plot_all: no data for metric '{}' in {}", metric, records_path)
        return
    logger.info("Generating plots | metric={} records={}", metric, len(df))
    plot_profile_ablation(df, output_dir, metric, ylabel)
    plot_inter_family(df, output_dir, metric, ylabel)
    plot_inter_split(df, output_dir, metric, ylabel)
    plot_seed_robustness(df, output_dir, metric, ylabel)
    plot_heatmap(df, output_dir, metric, ylabel)
    logger.success("All plots saved → {}", output_dir / "plots")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--records-path", required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--metric", default=_DEFAULT_METRIC)
    a = p.parse_args()
    plot_all(Path(a.records_path), Path(a.output_dir), metric=a.metric)
