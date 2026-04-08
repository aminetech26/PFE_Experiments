from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.statistical_tests import cohen_d, format_metric_report, wilcoxon_test


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDS_PATH = PROJECT_ROOT / "experiments" / "metrics" / "classification_comparison_records.jsonl"


def _load_records(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Comparison records file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in: {path}")

    df = pd.DataFrame(records)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df


def _filter_group(df: pd.DataFrame, field: str, value: str, max_rows: int | None) -> pd.DataFrame:
    filtered = df[df[field].astype(str) == value].copy()
    if max_rows is not None and max_rows > 0:
        filtered = filtered.sort_values("timestamp_utc").tail(max_rows)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classification runs with statistical tests")
    parser.add_argument("--records-path", default=str(DEFAULT_RECORDS_PATH), help="Path to comparison records jsonl")
    parser.add_argument("--metric", default="test_f1_weighted", help="Metric column to compare")
    parser.add_argument("--group-field", default="feature_profile", help="Field used to create A/B groups")
    parser.add_argument("--group-a", required=True, help="Group A value")
    parser.add_argument("--group-b", required=True, help="Group B value")
    parser.add_argument("--max-runs-per-group", type=int, default=None, help="Optional cap on recent runs per group")
    args = parser.parse_args()

    records_path = Path(args.records_path)
    if not records_path.is_absolute():
        records_path = PROJECT_ROOT / records_path

    df = _load_records(records_path)

    required_cols = {args.group_field, args.metric}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in records: {missing}")

    group_a = _filter_group(df, args.group_field, args.group_a, args.max_runs_per_group)
    group_b = _filter_group(df, args.group_field, args.group_b, args.max_runs_per_group)

    if group_a.empty or group_b.empty:
        raise ValueError(
            f"Not enough data for comparison. Group A rows={len(group_a)}, Group B rows={len(group_b)}"
        )

    scores_a = group_a[args.metric].dropna().astype(float).tolist()
    scores_b = group_b[args.metric].dropna().astype(float).tolist()

    pair_n = min(len(scores_a), len(scores_b))
    if pair_n < 2:
        raise ValueError("Need at least 2 paired scores per group for Wilcoxon test")

    # Pair most recent comparable runs by chronological order.
    scores_a = scores_a[-pair_n:]
    scores_b = scores_b[-pair_n:]

    result = wilcoxon_test(
        scores_a,
        scores_b,
        model_a_name=f"{args.group_field}={args.group_a}",
        model_b_name=f"{args.group_field}={args.group_b}",
    )
    result["metric"] = args.metric
    result["effect_size_cohen_d"] = cohen_d(scores_a, scores_b)
    result["group_a_report"] = format_metric_report(scores_a, metric_name=f"{args.metric} (A)")
    result["group_b_report"] = format_metric_report(scores_b, metric_name=f"{args.metric} (B)")
    result["paired_n"] = pair_n

    logger.info("Comparison result:\n{}", json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
