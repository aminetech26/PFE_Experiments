"""
EDA visualization pipeline.

Reads dataset-scoped EDA computation artifacts and generates reusable figures under
data/interim/eda/<dataset>/figures/.

Usage:
    uv run python -m src.data.eda_visualization --dataset la_reunion
    uv run python -m src.data.eda_visualization --dataset costa
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_active_dataset(config: dict) -> str:
    return str(config.get("active_dataset", "la_reunion"))


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing EDA artifact: {path}\n"
            f"Run: uv run python -m src.data.eda_pipeline --dataset {path.parent.name}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 5),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.dpi": 140,
        }
    )


def save_figure(fig: plt.Figure, output_path: Path, manifest: list[dict], title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"title": title, "path": str(output_path)})
    logger.info("Figure → {}", output_path)


def plot_sampling_intervals(df: pd.DataFrame, output_dir: Path, manifest: list[dict]) -> None:
    dt_s = df["timestamp"].diff().dt.total_seconds().dropna()
    positive = dt_s[dt_s > 0]
    if positive.empty:
        logger.warning("Skipping sampling interval plot: no positive intervals found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    zoom_cutoff = positive.quantile(0.99)
    sns.histplot(positive[positive <= zoom_cutoff], bins=60, ax=axes[0], color="#2D6A4F")
    axes[0].set_title("Sampling Intervals (Core Range)")
    axes[0].set_xlabel("Seconds")

    sns.histplot(positive, bins=60, ax=axes[1], color="#BC4749", log_scale=(False, True))
    axes[1].set_title("Sampling Intervals (Full Range)")
    axes[1].set_xlabel("Seconds")

    save_figure(fig, output_dir / "sampling_intervals.png", manifest, "Sampling intervals")


def plot_daily_coverage(df: pd.DataFrame, output_dir: Path, manifest: list[dict]) -> None:
    daily_counts = df.set_index("timestamp").resample("D").size()
    if daily_counts.empty:
        logger.warning("Skipping daily coverage plot: no rows after resampling")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    daily_counts.plot(ax=ax, color="#1D3557")
    ax.set_title("Daily Coverage")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rows")
    save_figure(fig, output_dir / "daily_coverage.png", manifest, "Daily coverage")


def plot_label_distribution(df: pd.DataFrame, output_dir: Path, manifest: list[dict]) -> None:
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#457B9D")
    ax.set_title("Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Rows")
    save_figure(fig, output_dir / "label_distribution.png", manifest, "Label distribution")


def plot_label_timeline(df: pd.DataFrame, output_dir: Path, manifest: list[dict]) -> None:
    sampled = df.iloc[:: max(1, len(df) // 20000)].copy()
    if sampled.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(sampled["timestamp"], sampled["label"], s=8, alpha=0.6, color="#E76F51")
    ax.set_title("Fault Timeline")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Label")
    save_figure(fig, output_dir / "label_timeline.png", manifest, "Label timeline")


def plot_top_feature_boxplots(
    df_eda: pd.DataFrame,
    findings: dict,
    output_dir: Path,
    manifest: list[dict],
) -> None:
    top_features = findings.get("mutual_information", {}).get("top_features_binary", [])[:6]
    available = [col for col in top_features if col in df_eda.columns]
    if not available:
        logger.warning("Skipping boxplots: no top MI features found in prepared frame")
        return

    plot_df = df_eda[available + ["label"]].copy()
    plot_df["target_group"] = plot_df["label"].apply(lambda x: "fault" if x != 0 else "normal")
    melted = plot_df.melt(
        id_vars=["target_group"], value_vars=available, var_name="feature", value_name="value"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=melted, x="feature", y="value", hue="target_group", ax=ax, showfliers=False)
    ax.set_title("Top Binary-MI Features: Normal vs Fault")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, output_dir / "top_feature_boxplots.png", manifest, "Top feature boxplots")


def plot_correlation_heatmap(
    df_eda: pd.DataFrame,
    sensor_columns: list[str],
    output_dir: Path,
    manifest: list[dict],
) -> None:
    available = [col for col in sensor_columns if col in df_eda.columns]
    if len(available) < 2:
        logger.warning("Skipping heatmap: fewer than 2 sensor columns available")
        return

    corr = df_eda[available].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Spearman Correlation Heatmap")
    save_figure(fig, output_dir / "correlation_heatmap.png", manifest, "Correlation heatmap")


def plot_mutual_information(findings: dict, output_dir: Path, manifest: list[dict]) -> None:
    results = findings.get("mutual_information", {}).get("results", [])
    if not results:
        logger.warning("Skipping MI plot: no mutual information results found")
        return

    mi_df = pd.DataFrame(results)
    top_binary = mi_df.sort_values("mi_binary_anomaly", ascending=False).head(10)
    top_multi = mi_df.sort_values("mi_multiclass_fault", ascending=False).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(data=top_binary, x="mi_binary_anomaly", y="feature", ax=axes[0], color="#2A9D8F")
    axes[0].set_title("Top 10 MI Features (Binary)")
    axes[0].set_xlabel("Mutual information")
    axes[0].set_ylabel("")

    sns.barplot(data=top_multi, x="mi_multiclass_fault", y="feature", ax=axes[1], color="#E9C46A")
    axes[1].set_title("Top 10 MI Features (Multiclass)")
    axes[1].set_xlabel("Mutual information")
    axes[1].set_ylabel("")

    save_figure(fig, output_dir / "mutual_information.png", manifest, "Mutual information")


def plot_class_imbalance(report: dict, output_dir: Path, manifest: list[dict]) -> None:
    rows = report.get("class_imbalance", {}).get("per_class", [])
    if not rows:
        logger.warning("Skipping class imbalance plot: no class imbalance section found")
        return

    df = pd.DataFrame(rows)
    melted = df.melt(
        id_vars=["label"],
        value_vars=["pct_rows", "pct_episodes"],
        var_name="share_type",
        value_name="pct",
    )
    melted["share_type"] = melted["share_type"].map(
        {"pct_rows": "Rows", "pct_episodes": "Episodes"}
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=melted, x="label", y="pct", hue="share_type", ax=ax)
    ax.set_title("Class Imbalance: Row Share vs Episode Share")
    ax.set_xlabel("Label")
    ax.set_ylabel("Share (%)")
    save_figure(
        fig,
        output_dir / "class_imbalance_rows_vs_episodes.png",
        manifest,
        "Class imbalance rows vs episodes",
    )


def plot_regime_binned_correlation(findings: dict, output_dir: Path, manifest: list[dict]) -> None:
    regime = findings.get("spearman", {}).get("contexts", {}).get("normal_only_irr_bins", {})
    profiles = regime.get("focus_pair_profiles", [])
    if not profiles:
        logger.warning("Skipping regime-binned correlation plot: no profile data found")
        return

    rows = []
    for profile in profiles:
        pair = f"{profile['feature_a']} ~ {profile['feature_b']}"
        for item in profile.get("by_bin", []):
            rows.append({"pair": pair, "bin_label": item["bin_label"], "rho": item["rho"]})
    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=corr_df, x="bin_label", y="rho", hue="pair", marker="o", ax=ax)
    ax.set_title("Normal-Only Correlation by Irradiance Regime")
    ax.set_xlabel("Irradiance quantile bin")
    ax.set_ylabel("Spearman rho")
    ax.tick_params(axis="x", rotation=25)
    save_figure(
        fig,
        output_dir / "regime_binned_correlation_profiles.png",
        manifest,
        "Regime-binned correlation profiles",
    )


def plot_vdc1_outlier_localization(report: dict, output_dir: Path, manifest: list[dict]) -> None:
    outliers = report.get("vdc1_outlier_localization", {})
    top_days = outliers.get("top_days", [])
    hourly = outliers.get("hourly_distribution", [])
    if not top_days and not hourly:
        logger.warning("Skipping VDC1 outlier localization plots: no data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    if top_days:
        daily_df = pd.DataFrame(top_days)
        sns.barplot(data=daily_df, x="date", y="n_outliers", ax=axes[0], color="#C1121F")
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].set_title("Top Days for Normal-Period VDC1 Outliers")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Outlier rows")
    else:
        axes[0].set_visible(False)

    if hourly:
        hourly_df = pd.DataFrame(hourly)
        sns.barplot(data=hourly_df, x="hour", y="n_outliers", ax=axes[1], color="#669BBC")
        axes[1].set_title("Hourly Distribution of Normal-Period VDC1 Outliers")
        axes[1].set_xlabel("Hour of day")
        axes[1].set_ylabel("Outlier rows")
    else:
        axes[1].set_visible(False)

    save_figure(
        fig, output_dir / "vdc1_outlier_localization.png", manifest, "VDC1 outlier localization"
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _robust_positive_z(series: pd.Series, baseline: pd.Series) -> pd.Series:
    median = float(baseline.median())
    mad = float((baseline - median).abs().median())
    scale = 1.4826 * mad if mad > 0 else float(baseline.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return ((series - median) / scale).clip(lower=0)


def _build_costa_targeted_frame(df: pd.DataFrame) -> pd.DataFrame | None:
    required = {"label", "vdc1", "vdc2", "idc1", "idc2", "pdc1", "pdc2"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        logger.warning("Skipping Costa targeted evidence plots: missing columns {}", missing)
        return None

    out = df[list(required)].copy()
    total_voltage = out["vdc1"].abs() + out["vdc2"].abs()
    total_current = out["idc1"].abs() + out["idc2"].abs()
    total_power = out["pdc1"].abs() + out["pdc2"].abs()
    out["voltage_imbalance"] = _safe_ratio((out["vdc1"] - out["vdc2"]).abs(), total_voltage)
    out["current_imbalance"] = _safe_ratio((out["idc1"] - out["idc2"]).abs(), total_current)
    out["power_imbalance"] = _safe_ratio((out["pdc1"] - out["pdc2"]).abs(), total_power)

    normal = out[out["label"] == 0.0]
    if normal.empty:
        logger.warning("Skipping Costa targeted evidence plots: no normal rows found")
        return None

    normal_vdc2 = normal["vdc2"].dropna()
    if normal_vdc2.empty:
        logger.warning("Skipping Costa targeted evidence plots: normal vdc2 is empty")
        return None

    normal_vdc2_median = float(normal_vdc2.median())
    normal_vdc2_mad = float((normal_vdc2 - normal_vdc2_median).abs().median())
    vdc2_scale = 1.4826 * normal_vdc2_mad if normal_vdc2_mad > 0 else float(normal_vdc2.std(ddof=0))
    if not np.isfinite(vdc2_scale) or vdc2_scale <= 0:
        vdc2_scale = 1.0
    out["voltage_drop_vdc2_z"] = ((normal_vdc2_median - out["vdc2"]) / vdc2_scale).clip(lower=0)

    imbalance_cols = ["power_imbalance", "current_imbalance", "voltage_imbalance"]
    for col in imbalance_cols:
        out[f"{col}_z"] = _robust_positive_z(out[col], normal[col].dropna())
    out["mismatch_imbalance_z"] = out[[f"{col}_z" for col in imbalance_cols]].max(axis=1)
    return out


def plot_costa_targeted_evidence(
    df: pd.DataFrame, output_dir: Path, manifest: list[dict]
) -> None:
    evidence = _build_costa_targeted_frame(df)
    if evidence is None:
        return

    sample = evidence.iloc[:: max(1, len(evidence) // 60000)].copy()
    sample["label"] = sample["label"].map(lambda x: f"{x:g}")
    melted = sample.melt(
        id_vars="label",
        value_vars=["voltage_drop_vdc2_z", "mismatch_imbalance_z"],
        var_name="signal",
        value_name="normal_robust_z",
    )
    melted["signal"] = melted["signal"].map(
        {
            "voltage_drop_vdc2_z": "Voltage drop (VDC2)",
            "mismatch_imbalance_z": "String mismatch",
        }
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.boxplot(
        data=melted,
        x="label",
        y="normal_robust_z",
        hue="signal",
        ax=ax,
        showfliers=False,
    )
    ax.set_title("Costa Targeted Physics Signals by Fault Class")
    ax.set_xlabel("Label (0=normal, 1=short-circuit, 2=degradation, 3=open-circuit, 4=shadowing)")
    ax.set_ylabel("Positive robust z-score vs normal baseline")
    save_figure(
        fig,
        output_dir / "costa_targeted_physics_signals.png",
        manifest,
        "Costa targeted physics signals",
    )

    summary_cols = [
        "voltage_drop_vdc2_z",
        "power_imbalance_z",
        "current_imbalance_z",
        "voltage_imbalance_z",
        "mismatch_imbalance_z",
    ]
    heatmap_df = evidence.groupby("label", observed=True)[summary_cols].median().rename(
        columns={
            "voltage_drop_vdc2_z": "VDC2 drop",
            "power_imbalance_z": "Power mismatch",
            "current_imbalance_z": "Current mismatch",
            "voltage_imbalance_z": "Voltage mismatch",
            "mismatch_imbalance_z": "Max mismatch",
        }
    )
    heatmap_df.index = [f"{idx:g}" for idx in heatmap_df.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heatmap_df, cmap="mako", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Costa Median Targeted Signal Strength by Class")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Label")
    save_figure(
        fig,
        output_dir / "costa_targeted_signal_heatmap.png",
        manifest,
        "Costa targeted signal heatmap",
    )


def write_manifest(
    output_dir: Path, dataset: str, figures: list[dict], source_artifacts: dict
) -> None:
    payload = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "figures_dir": str(output_dir),
        "source_artifacts": source_artifacts,
        "figures": figures,
    }
    manifest_path = output_dir.parent / "eda_figure_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success("Figure manifest → {}", manifest_path)


def generate_visualizations(config: dict, dataset: str) -> int:
    from src.data.eda_pipeline import load_dataset, prepare_eda_frame

    setup_plot_style()
    dataset_cfg = config["paths"]["datasets"][dataset]
    sensor_columns = list(dataset_cfg.get("feature_engineering", {}).get("sensor_columns", []))

    output_root = PROJECT_ROOT / "data" / "interim" / "eda" / dataset
    figures_dir = output_root / "figures"

    report_path = output_root / "eda_dataset_report.json"
    findings_path = output_root / "eda_feature_findings.json"
    artifact_manifest_path = output_root / "eda_artifact_manifest.json"

    report = load_artifact(report_path)
    findings = load_artifact(findings_path)
    artifact_manifest = load_artifact(artifact_manifest_path)

    logger.info("Loading dataset for visualization: {}", dataset)
    df_raw = load_dataset(config, dataset)
    df_eda, _ = prepare_eda_frame(df_raw, dataset)

    figures: list[dict] = []
    plot_sampling_intervals(df_raw, figures_dir, figures)
    plot_daily_coverage(df_raw, figures_dir, figures)
    plot_label_distribution(df_raw, figures_dir, figures)
    plot_label_timeline(df_raw, figures_dir, figures)
    plot_top_feature_boxplots(df_eda, findings, figures_dir, figures)
    plot_correlation_heatmap(df_eda, sensor_columns, figures_dir, figures)
    plot_mutual_information(findings, figures_dir, figures)
    plot_class_imbalance(report, figures_dir, figures)
    plot_regime_binned_correlation(findings, figures_dir, figures)
    plot_vdc1_outlier_localization(report, figures_dir, figures)
    if dataset == "costa":
        plot_costa_targeted_evidence(df_raw, figures_dir, figures)

    write_manifest(
        figures_dir,
        dataset,
        figures,
        {
            "artifact_manifest": str(artifact_manifest_path),
            "dataset_report": str(report_path),
            "feature_findings": str(findings_path),
            "dataset_report_version": report.get("version"),
            "feature_findings_version": findings.get("version"),
            "artifact_manifest_version": artifact_manifest.get("version"),
        },
    )

    logger.success("EDA visualization complete | dataset={} | figures={}", dataset, len(figures))
    return len(figures)


def main() -> None:
    config = load_config()
    default_dataset = get_active_dataset(config)
    parser = argparse.ArgumentParser(
        description="Generate dataset-scoped EDA figures from computation artifacts."
    )
    parser.add_argument(
        "--dataset", default=default_dataset, help="Dataset key from data_config.yaml"
    )
    args = parser.parse_args()
    generate_visualizations(config, args.dataset)


if __name__ == "__main__":
    main()
