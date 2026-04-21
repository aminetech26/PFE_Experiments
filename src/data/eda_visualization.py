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
import pandas as pd
import seaborn as sns
import yaml
from loguru import logger

from src.data.eda_pipeline import load_dataset, prepare_eda_frame

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

    setup_plot_style()
    dataset_cfg = config["paths"]["datasets"][args.dataset]
    sensor_columns = list(dataset_cfg.get("feature_engineering", {}).get("sensor_columns", []))

    output_root = PROJECT_ROOT / "data" / "interim" / "eda" / args.dataset
    figures_dir = output_root / "figures"

    report_path = output_root / "eda_dataset_report.json"
    findings_path = output_root / "eda_feature_findings.json"
    artifact_manifest_path = output_root / "eda_artifact_manifest.json"

    report = load_artifact(report_path)
    findings = load_artifact(findings_path)
    artifact_manifest = load_artifact(artifact_manifest_path)

    logger.info("Loading dataset for visualization: {}", args.dataset)
    df_raw = load_dataset(config, args.dataset)
    df_eda, _ = prepare_eda_frame(df_raw, args.dataset)

    figures: list[dict] = []
    plot_sampling_intervals(df_raw, figures_dir, figures)
    plot_daily_coverage(df_raw, figures_dir, figures)
    plot_label_distribution(df_raw, figures_dir, figures)
    plot_label_timeline(df_raw, figures_dir, figures)
    plot_top_feature_boxplots(df_eda, findings, figures_dir, figures)
    plot_correlation_heatmap(df_eda, sensor_columns, figures_dir, figures)
    plot_mutual_information(findings, figures_dir, figures)

    write_manifest(
        figures_dir,
        args.dataset,
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

    logger.success(
        "EDA visualization complete | dataset={} | figures={}", args.dataset, len(figures)
    )


if __name__ == "__main__":
    main()
