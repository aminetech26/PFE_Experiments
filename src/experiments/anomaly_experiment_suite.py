"""
Anomaly detection experiment suite.

Runs the full (model × profile × split × seed) matrix, skipping combinations
already present in the comparison records, then auto-generates the comparison
report and plots.

Usage (from PFE_Experiments/):
    uv run python -m src.experiments.anomaly_experiment_suite \\
        --models isolation_forest ocsvm_rbf \\
        --profiles baseline_raw plus_physics plus_rolling \\
        --splits path_a path_b \\
        --dataset costa --task anomaly_semisup

    # dry-run to see what would be launched:
    uv run python -m src.experiments.anomaly_experiment_suite ... --dry-run

Model shorthand:
    isolation_forest  → runs isolation_forest
    ocsvm_rbf         → runs one_class_svm --kernel rbf
    ocsvm_poly        → runs one_class_svm --kernel poly
    xgboost           → runs xgboost (requires anomaly_supervised task)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
DEFAULT_RECORDS_PATH = PROJECT_ROOT / "experiments" / "metrics" / "anomaly_comparison_records.jsonl"

_MODEL_SHORTHANDS: dict[str, tuple[str, list[str]]] = {
    "isolation_forest": ("isolation_forest", []),
    "ocsvm_rbf":        ("one_class_svm",    ["--kernel", "rbf"]),
    "ocsvm_poly":       ("one_class_svm",    ["--kernel", "poly"]),
    "xgboost":          ("xgboost",          []),
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_done(records_path: Path) -> set[tuple]:
    """Return set of (model_key, profile, split, seed) already in records."""
    done: set[tuple] = set()
    if not records_path.exists():
        return done
    with records_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = rec.get("model", "")
            kernel = rec.get("kernel", "")
            model_key = f"{model}_{kernel}" if kernel else model
            done.add((
                model_key,
                str(rec.get("feature_profile", "")),
                str(rec.get("split_path", "")),
                int(rec.get("seed", -1)),
            ))
    return done


def _run_one(
    model_name: str,
    extra_model_args: list[str],
    profile: str,
    split: str,
    seed: int,
    dataset: str,
    task: str,
    records_path: Path,
    dry_run: bool,
) -> bool:
    """Launch one experiment subprocess. Returns True on success."""
    cmd = [
        sys.executable, "-m", "src.modeling.anomaly_detection.ml.run",
        "--model", model_name,
        "--profile", profile,
        "--split-path", split,
        "--seed", str(seed),
        "--dataset", dataset,
        "--task", task,
        "--run-type", "suite",
        "--comparison-records-path", str(records_path),
        *extra_model_args,
    ]

    label = f"{model_name}{'_' + extra_model_args[extra_model_args.index('--kernel') + 1] if '--kernel' in extra_model_args else ''} | {profile} | {split} | seed={seed}"

    if dry_run:
        logger.info("[DRY-RUN] would run: {}", " ".join(cmd))
        return True

    import subprocess
    logger.info("▶ {}", label)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)
    if result.returncode != 0:
        logger.error("✗ FAILED (rc={}) — {}", result.returncode, label)
        return False
    logger.success("✓ {}", label)
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed anomaly experiment suite")
    parser.add_argument(
        "--models", nargs="+",
        default=["isolation_forest", "ocsvm_rbf"],
        choices=list(_MODEL_SHORTHANDS),
        help="Models to run (shorthands)",
    )
    parser.add_argument(
        "--profiles", nargs="+",
        default=["baseline_raw", "plus_physics", "plus_rolling"],
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=["path_a", "path_b"],
    )
    parser.add_argument("--dataset", default="costa")
    parser.add_argument("--task", default="anomaly_semisup")
    parser.add_argument(
        "--records-path", default=str(DEFAULT_RECORDS_PATH),
        help="Shared JSONL file all runs append to",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip auto-plot generation after runs",
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip running experiments, only generate plots and report",
    )
    args = parser.parse_args()

    config = _load_config()
    seeds: list[int] = [int(s) for s in config.get("experiment", {}).get("seeds", [42])]
    records_path = Path(args.records_path)

    logger.info(
        "Suite | models={} profiles={} splits={} seeds={} dataset={} task={}",
        args.models, args.profiles, args.splits, seeds, args.dataset, args.task,
    )

    # ── run experiment matrix ────────────────────────────────────────────────
    if not args.plots_only:
        done = _load_done(records_path)
        total = len(args.models) * len(args.profiles) * len(args.splits) * len(seeds)
        completed = 0
        skipped = 0
        failed = 0

        for shorthand in args.models:
            model_name, extra_args = _MODEL_SHORTHANDS[shorthand]
            kernel = extra_args[extra_args.index("--kernel") + 1] if "--kernel" in extra_args else ""
            model_key = f"{model_name}_{kernel}" if kernel else model_name

            for profile in args.profiles:
                for split in args.splits:
                    for seed in seeds:
                        key = (model_key, profile, split, seed)
                        if key in done:
                            logger.debug("skip (already done): {}", key)
                            skipped += 1
                            continue
                        ok = _run_one(
                            model_name, extra_args,
                            profile, split, seed,
                            args.dataset, args.task,
                            records_path, args.dry_run,
                        )
                        if ok:
                            completed += 1
                            done.add(key)
                        else:
                            failed += 1

        logger.info(
            "Matrix done — ran={} skipped={} failed={} / total={}",
            completed, skipped, failed, total,
        )

    # ── comparison report + plots ────────────────────────────────────────────
    if not args.dry_run and not args.skip_plots:
        _run_analysis(records_path, args.models, args.profiles, args.splits, seeds)


def _run_analysis(
    records_path: Path,
    models: list[str],
    profiles: list[str],
    splits: list[str],
    seeds: list[int],
) -> None:
    from src.evaluation.experiment_report import run_comparison_report
    from src.evaluation.anomaly_plots import plot_all

    if not records_path.exists() or records_path.stat().st_size == 0:
        logger.warning("Records file empty or missing — skipping analysis ({})", records_path)
        return

    output_dir = PROJECT_ROOT / "experiments" / "anomaly_suite"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_comparison_report(records_path, output_dir, models, profiles, splits, seeds)
    plot_all(records_path, output_dir)


if __name__ == "__main__":
    main()
