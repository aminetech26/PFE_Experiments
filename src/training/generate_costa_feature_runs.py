from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"

TASKS = ("anomaly_semisup", "anomaly_supervised", "classification")
SPLIT_PATHS = ("path_a", "path_b")


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _all_profiles(config: dict) -> list[str]:
    profiles = config.get("feature_engineering", {}).get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("No feature_engineering.profiles found in configs/data_config.yaml")
    return [name for name in profiles.keys() if name != "plus_differential"]


def _features_root(dataset: str, split_path: str) -> Path:
    root = PROJECT_ROOT / "data" / "processed" / "features" / dataset
    if split_path == "path_b":
        root = root / "path_b"
    return root


def _canonical_run_dir(dataset: str, split_path: str, task: str, profile: str) -> Path:
    return _features_root(dataset, split_path) / task / "runs" / profile


def _run_module(module: str, *args: str, dry_run: bool = False) -> None:
    command = [sys.executable, "-m", module, *args]
    rendered = " ".join(command)
    logger.info("{}", rendered)
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _prepare_base_artifacts(dataset: str, split_paths: list[str], dry_run: bool) -> None:
    _run_module("src.data.ingestion", "--dataset", dataset, dry_run=dry_run)
    _run_module("src.data.split_pipeline", "--dataset", dataset, dry_run=dry_run)
    for split_path in split_paths:
        _run_module(
            "src.data.preprocess_pipeline",
            "--dataset",
            dataset,
            "--split-path",
            split_path,
            dry_run=dry_run,
        )


def _generate_feature_runs(
    dataset: str,
    split_paths: list[str],
    tasks: list[str],
    profiles: list[str],
    force: bool,
    dry_run: bool,
) -> tuple[int, int]:
    generated = 0
    skipped = 0

    for split_path in split_paths:
        for task in tasks:
            for profile in profiles:
                run_dir = _canonical_run_dir(dataset, split_path, task, profile)
                if run_dir.exists() and not force:
                    logger.info("Skipping existing run: {}", run_dir)
                    skipped += 1
                    continue

                _run_module(
                    "src.data.featurize_pipeline",
                    "--dataset",
                    dataset,
                    "--split-path",
                    split_path,
                    "--task",
                    task,
                    "--profile",
                    profile,
                    dry_run=dry_run,
                )
                generated += 1

    return generated, skipped


def main() -> None:
    config = _load_config()
    available_profiles = _all_profiles(config)

    parser = argparse.ArgumentParser(
        description="Generate Costa feature runs for all configured profiles and split paths"
    )
    parser.add_argument("--dataset", default="costa", choices=["costa"])
    parser.add_argument("--split-paths", nargs="+", choices=SPLIT_PATHS, default=list(SPLIT_PATHS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--profiles", nargs="+", default=available_profiles)
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Skip ingestion/split/preprocess and only run featurization",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate runs even when the canonical profile directory already exists",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    unknown_profiles = [profile for profile in args.profiles if profile not in available_profiles]
    if unknown_profiles:
        raise ValueError(f"Unknown profiles requested: {unknown_profiles}")

    logger.info("Costa feature run batch start")
    logger.info("Dataset: {}", args.dataset)
    logger.info("Split paths: {}", ", ".join(args.split_paths))
    logger.info("Tasks: {}", ", ".join(args.tasks))
    logger.info("Profiles: {}", ", ".join(args.profiles))

    if not args.features_only:
        _prepare_base_artifacts(args.dataset, args.split_paths, dry_run=args.dry_run)

    generated, skipped = _generate_feature_runs(
        dataset=args.dataset,
        split_paths=args.split_paths,
        tasks=args.tasks,
        profiles=args.profiles,
        force=args.force,
        dry_run=args.dry_run,
    )

    total = len(args.split_paths) * len(args.tasks) * len(args.profiles)
    logger.success(
        "Costa feature batch complete | total={} | generated={} | skipped={}",
        total,
        generated,
        skipped,
    )


if __name__ == "__main__":
    main()
