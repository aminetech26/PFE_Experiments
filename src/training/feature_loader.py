from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_ROOT = PROJECT_ROOT / "data" / "processed" / "features"


def _normalize_relative_path(relative_path: str) -> Path:
    # latest_runs.json may contain Windows-style separators; normalize cross-platform.
    return Path(relative_path.replace("\\", "/"))


def resolve_run_dir(task: str, profile: str | None = None, run_dir: str | None = None) -> Path:
    if run_dir:
        path = Path(run_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    latest_runs_path = FEATURES_ROOT / "latest_runs.json"
    if not latest_runs_path.exists():
        raise FileNotFoundError(f"Missing latest runs file: {latest_runs_path}")

    latest_payload = json.loads(latest_runs_path.read_text(encoding="utf-8"))

    if profile:
        by_task_profile = latest_payload.get("latest_by_task_profile", {})
        rel = by_task_profile.get(task, {}).get(profile)
        if rel:
            return FEATURES_ROOT / _normalize_relative_path(rel)

    by_task = latest_payload.get("latest_by_task", {})
    rel = by_task.get(task)
    if not rel:
        raise KeyError(
            f"Could not resolve latest run for task='{task}'"
            + (f", profile='{profile}'" if profile else "")
            + f" in {latest_runs_path}"
        )

    return FEATURES_ROOT / _normalize_relative_path(rel)


def resolve_run_dir_by_id(task: str, run_id: str) -> Path:
    run_path = FEATURES_ROOT / task / "runs" / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Feature run id not found: {run_path}")
    return run_path


def load_features_for_task(
    task: str,
    profile: str | None = None,
    run_dir: str | None = None,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, Path]:
    if run_id:
        resolved_run_dir = resolve_run_dir_by_id(task=task, run_id=run_id)
    else:
        resolved_run_dir = resolve_run_dir(task=task, profile=profile, run_dir=run_dir)

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
