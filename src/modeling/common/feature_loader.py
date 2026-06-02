from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _features_root(dataset: str, split_path: str) -> Path:
    root = PROJECT_ROOT / "data" / "processed" / "features" / dataset
    if split_path == "path_b":
        root = root / "path_b"
    return root


def _normalize_relative_path(relative_path: str) -> Path:
    return Path(relative_path.replace("\\", "/"))


def resolve_run_dir(
    task: str,
    profile: str | None = None,
    run_dir: str | None = None,
    dataset: str = "costa",
    split_path: str = "path_a",
) -> Path:
    features_root = _features_root(dataset=dataset, split_path=split_path)

    if run_dir:
        path = Path(run_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    latest_runs_path = features_root / "latest_runs.json"
    if latest_runs_path.exists():
        latest_payload = json.loads(latest_runs_path.read_text(encoding="utf-8"))

        if profile:
            by_task_profile = latest_payload.get("latest_by_task_profile", {})
            rel = by_task_profile.get(task, {}).get(profile)
            if rel:
                return features_root / _normalize_relative_path(rel)

        by_task = latest_payload.get("latest_by_task", {})
        rel = by_task.get(task)
        if rel:
            return features_root / _normalize_relative_path(rel)

    # latest_runs.json absent or incomplete — scan the standard directory tree.
    # Priority 1: features_root/task/runs/<profile>  (exact profile match)
    # Priority 2: most-recently-modified dir under features_root/task/runs/
    runs_root = features_root / task / "runs"
    if profile:
        profile_dir = runs_root / profile
        if profile_dir.is_dir():
            return profile_dir

    if runs_root.is_dir():
        candidates = [d for d in runs_root.iterdir() if d.is_dir()]
        if candidates:
            return max(candidates, key=lambda d: d.stat().st_mtime)

    raise FileNotFoundError(
        f"Cannot resolve feature run dir for task='{task}'"
        + (f", profile='{profile}'" if profile else "")
        + f" under {features_root}. "
        "Run featurization first or pass --run-dir / --run-id explicitly."
    )


def resolve_run_dir_by_id(
    task: str,
    run_id: str,
    dataset: str = "costa",
    split_path: str = "path_a",
) -> Path:
    features_root = _features_root(dataset=dataset, split_path=split_path)
    run_path = features_root / task / "runs" / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Feature run id not found: {run_path}")
    return run_path


def load_features_for_task(
    task: str,
    profile: str | None = None,
    run_dir: str | None = None,
    run_id: str | None = None,
    dataset: str = "costa",
    split_path: str = "path_a",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, Path]:
    if run_id:
        resolved_run_dir = resolve_run_dir_by_id(
            task=task,
            run_id=run_id,
            dataset=dataset,
            split_path=split_path,
        )
    else:
        resolved_run_dir = resolve_run_dir(
            task=task,
            profile=profile,
            run_dir=run_dir,
            dataset=dataset,
            split_path=split_path,
        )

    manifest_path = resolved_run_dir / "features_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_path = resolved_run_dir / "train.parquet"
    val_path = resolved_run_dir / "val.parquet"
    test_path = resolved_run_dir / "test.parquet"
    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing feature split file: {path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    return train_df, val_df, test_df, manifest, resolved_run_dir
