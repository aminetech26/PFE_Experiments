"""Diagnose cls2 val→test gap on Costa anomaly_semisup split.

Counts episodes per class per split, computes operating-point (irr, pvt) coverage,
and reports overlap between val and test cls2 regions. Output: console report +
PNG scatter plot of (irr, pvt) per split, colored by class.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

SPLIT_DIR = Path("data/interim/splits/costa/anomaly_semisup")
OUT_DIR = Path("experiments/diagnostics/costa_cls2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> pd.DataFrame:
    p = SPLIT_DIR / f"{name}.parquet"
    df = pd.read_parquet(p)
    logger.info("{}: {} rows, labels={}", name, len(df), sorted(df["label"].unique().tolist()))
    return df


def episode_summary(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """One row per episode: label, n_samples, irr/pvt summary stats."""
    rows = []
    for ep_id, sub in df.groupby("episode_id"):
        labels = sub["label"].unique()
        # episode label = majority (should be unique for fault episodes by design)
        label = int(sub["label"].mode().iat[0])
        rows.append({
            "split": split,
            "episode_id": ep_id,
            "label": label,
            "n_samples": len(sub),
            "irr_mean": sub["irr"].mean(),
            "irr_min": sub["irr"].min(),
            "irr_max": sub["irr"].max(),
            "pvt_mean": sub["pvt"].mean(),
            "pvt_min": sub["pvt"].min(),
            "pvt_max": sub["pvt"].max(),
            "n_unique_labels": len(labels),
        })
    return pd.DataFrame(rows)


def coverage_overlap(val_ep: pd.DataFrame, test_ep: pd.DataFrame, cls: int) -> dict:
    """Quantify (irr, pvt) overlap between val and test episodes of one class."""
    v = val_ep[val_ep["label"] == cls]
    t = test_ep[test_ep["label"] == cls]
    if len(v) == 0 or len(t) == 0:
        return {"class": cls, "val_episodes": len(v), "test_episodes": len(t),
                "overlap_irr": None, "overlap_pvt": None, "iou_2d_grid": None}

    irr_v = (v["irr_min"].min(), v["irr_max"].max())
    irr_t = (t["irr_min"].min(), t["irr_max"].max())
    pvt_v = (v["pvt_min"].min(), v["pvt_max"].max())
    pvt_t = (t["pvt_min"].min(), t["pvt_max"].max())

    def overlap_1d(a, b):
        lo = max(a[0], b[0])
        hi = min(a[1], b[1])
        if hi <= lo:
            return 0.0
        union = max(a[1], b[1]) - min(a[0], b[0])
        return (hi - lo) / union if union > 0 else 0.0

    # 2D IoU on rasterized (irr, pvt) grid using episode-mean as the representative point
    irr_all = np.concatenate([v["irr_mean"].values, t["irr_mean"].values])
    pvt_all = np.concatenate([v["pvt_mean"].values, t["pvt_mean"].values])
    if len(irr_all) < 2:
        iou_2d = None
    else:
        nbins = 20
        irr_edges = np.linspace(irr_all.min(), irr_all.max() + 1e-9, nbins + 1)
        pvt_edges = np.linspace(pvt_all.min(), pvt_all.max() + 1e-9, nbins + 1)
        h_v, _, _ = np.histogram2d(v["irr_mean"], v["pvt_mean"], bins=[irr_edges, pvt_edges])
        h_t, _, _ = np.histogram2d(t["irr_mean"], t["pvt_mean"], bins=[irr_edges, pvt_edges])
        bv = (h_v > 0).astype(int)
        bt = (h_t > 0).astype(int)
        inter = int((bv & bt).sum())
        union = int((bv | bt).sum())
        iou_2d = inter / union if union > 0 else 0.0

    return {
        "class": cls,
        "val_episodes": len(v),
        "test_episodes": len(t),
        "val_irr_range": irr_v,
        "test_irr_range": irr_t,
        "irr_iou_1d": overlap_1d(irr_v, irr_t),
        "val_pvt_range": pvt_v,
        "test_pvt_range": pvt_t,
        "pvt_iou_1d": overlap_1d(pvt_v, pvt_t),
        "iou_2d_grid": iou_2d,
    }


def plot_coverage(splits_dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    """Scatter (irr_mean, pvt_mean) per split, colored by class."""
    classes = sorted({c for d in splits_dfs.values() for c in d["label"].unique()})
    fig, axes = plt.subplots(1, len(splits_dfs), figsize=(5 * len(splits_dfs), 5), sharex=True, sharey=True)
    if len(splits_dfs) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")

    for ax, (split, df) in zip(axes, splits_dfs.items()):
        for cls in classes:
            sub = df[df["label"] == cls]
            ax.scatter(sub["irr_mean"], sub["pvt_mean"], s=60, alpha=0.7,
                       color=cmap(cls), label=f"cls{cls} (n={len(sub)})",
                       edgecolors="black", linewidths=0.5)
        ax.set_title(f"{split} — episode (irr, pvt) means")
        ax.set_xlabel("irr (mean over episode)")
        ax.set_ylabel("pvt (mean over episode)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    logger.info("Plot saved: {}", out_path)


def main() -> None:
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    # Per-split label distribution (sample-level)
    print("\n=== Sample-level label counts per split ===")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        counts = df["label"].value_counts().sort_index()
        print(f"{name:6s}  {counts.to_dict()}")

    # Per-episode summaries
    val_ep = episode_summary(val, "val")
    test_ep = episode_summary(test, "test")

    print("\n=== Episode counts per class per split ===")
    print("val   :", val_ep.groupby("label")["episode_id"].nunique().to_dict())
    print("test  :", test_ep.groupby("label")["episode_id"].nunique().to_dict())

    # Overlap analysis for every fault class
    print("\n=== (irr, pvt) coverage overlap between val and test ===")
    overlap_records = []
    for cls in sorted(set(val_ep["label"]).union(test_ep["label"])):
        if cls == 0:
            continue
        rec = coverage_overlap(val_ep, test_ep, cls)
        overlap_records.append(rec)
        print(f"\nclass {cls}:")
        for k, v in rec.items():
            print(f"  {k}: {v}")

    # Detail: every cls2 episode
    print("\n=== ALL cls2 episodes (val) ===")
    cls2_val = val_ep[val_ep["label"] == 2].sort_values("episode_id")
    print(cls2_val[["episode_id", "n_samples", "irr_mean", "irr_min", "irr_max",
                    "pvt_mean", "pvt_min", "pvt_max"]].to_string(index=False))

    print("\n=== ALL cls2 episodes (test) ===")
    cls2_test = test_ep[test_ep["label"] == 2].sort_values("episode_id")
    print(cls2_test[["episode_id", "n_samples", "irr_mean", "irr_min", "irr_max",
                     "pvt_mean", "pvt_min", "pvt_max"]].to_string(index=False))

    # Save CSVs + summary JSON
    val_ep.to_csv(OUT_DIR / "val_episodes.csv", index=False)
    test_ep.to_csv(OUT_DIR / "test_episodes.csv", index=False)

    summary = {
        "sample_counts": {
            "train": {int(k): int(v) for k, v in train["label"].value_counts().to_dict().items()},
            "val": {int(k): int(v) for k, v in val["label"].value_counts().to_dict().items()},
            "test": {int(k): int(v) for k, v in test["label"].value_counts().to_dict().items()},
        },
        "episode_counts": {
            "val": {int(k): int(v) for k, v in val_ep.groupby("label")["episode_id"].nunique().to_dict().items()},
            "test": {int(k): int(v) for k, v in test_ep.groupby("label")["episode_id"].nunique().to_dict().items()},
        },
        "overlap": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in overlap_records],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Summary saved: {}", OUT_DIR / "summary.json")

    # Coverage plot
    plot_coverage({"val": val_ep, "test": test_ep}, OUT_DIR / "coverage_irr_pvt.png")

    # Headline answer
    cls2_rec = next(r for r in overlap_records if r["class"] == 2)
    print("\n" + "=" * 60)
    print(f"HEADLINE: cls2 (irr, pvt) 2D grid IoU = {cls2_rec['iou_2d_grid']}")
    print(f"          val cls2 episodes = {cls2_rec['val_episodes']}")
    print(f"          test cls2 episodes = {cls2_rec['test_episodes']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
