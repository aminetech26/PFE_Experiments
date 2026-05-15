from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

_TASK_A_THRESHOLD_POLICY = "normal_validation_quantile"
_TASK_A_THRESHOLD_QUANTILE = 0.95
_TASK_A_SCORE_DIRECTION = "higher_is_more_anomalous"


_METADATA_KEYS = frozenset({
    "threshold_policy", "threshold_quantile", "score_direction",
    "selection_metric", "run_type", "model", "model_family", "task",
})


def _is_numeric_leaf(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_numeric_json(payload: object, *, key: str = "") -> bool:
    """Metrics JSON must be numeric, except known metadata keys which may be str/bool."""
    if isinstance(payload, dict):
        return all(
            _validate_numeric_json(v, key=k)
            for k, v in payload.items()
        )
    if isinstance(payload, list):
        return all(_validate_numeric_json(v, key=key) for v in payload)
    if key in _METADATA_KEYS:
        return isinstance(payload, (int, float, str, bool))
    return _is_numeric_leaf(payload)


def _validate_task_a_operating_point(artifact_dir: Path, dm: dict) -> list[str]:
    """Enforce the shared q95 operating-point contract for finalized Task A methods.

    Checks:
    - score_calibration.json exists and is referenced in the deployment manifest
    - threshold_policy == "normal_validation_quantile"
    - threshold_quantile == 0.95
    - score_direction == "higher_is_more_anomalous"
    - test_not_used_for_calibration is true
    - deployment_manifest.threshold matches score_calibration.threshold (within 1e-9)
    """
    errors: list[str] = []

    cal_name = dm.get("score_calibration_artifact")
    if not cal_name:
        errors.append(
            "deployment_manifest.json missing score_calibration_artifact "
            "(required for Task A operating-point contract)"
        )
        return errors

    cal_path = artifact_dir / str(cal_name)
    if not cal_path.exists():
        errors.append(f"score_calibration_artifact does not exist: {cal_name}")
        return errors

    try:
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"score_calibration.json unreadable: {exc}")
        return errors

    if cal.get("threshold_policy") != _TASK_A_THRESHOLD_POLICY:
        errors.append(
            f"score_calibration.threshold_policy must be '{_TASK_A_THRESHOLD_POLICY}', "
            f"got '{cal.get('threshold_policy')}'"
        )
    try:
        cal_quantile = float(cal.get("threshold_quantile", 0))
    except (TypeError, ValueError):
        errors.append(f"score_calibration.threshold_quantile is not numeric: {cal.get('threshold_quantile')!r}")
        cal_quantile = None
    if cal_quantile is not None and not math.isclose(cal_quantile, _TASK_A_THRESHOLD_QUANTILE, rel_tol=1e-9):
        errors.append(
            f"score_calibration.threshold_quantile must be {_TASK_A_THRESHOLD_QUANTILE}, "
            f"got {cal.get('threshold_quantile')}"
        )
    if cal.get("score_direction") != _TASK_A_SCORE_DIRECTION:
        errors.append(
            f"score_calibration.score_direction must be '{_TASK_A_SCORE_DIRECTION}', "
            f"got '{cal.get('score_direction')}'"
        )
    if cal.get("test_not_used_for_calibration") is not True:
        errors.append("score_calibration.test_not_used_for_calibration must be true")

    dm_threshold = dm.get("threshold")
    cal_threshold = cal.get("threshold")
    if dm_threshold is not None and cal_threshold is not None:
        try:
            if not math.isclose(float(dm_threshold), float(cal_threshold), rel_tol=1e-9, abs_tol=1e-12):
                errors.append(
                    f"deployment_manifest.threshold ({dm_threshold}) does not match "
                    f"score_calibration.threshold ({cal_threshold})"
                )
        except (TypeError, ValueError):
            errors.append(
                f"threshold comparison failed — non-numeric values: "
                f"manifest={dm_threshold!r}, calibration={cal_threshold!r}"
            )
    elif dm_threshold is None:
        errors.append("deployment_manifest.json missing threshold")
    elif cal_threshold is None:
        errors.append("score_calibration.json missing threshold")

    # Also verify the manifest itself carries consistent policy metadata.
    if dm.get("score_direction") != _TASK_A_SCORE_DIRECTION:
        errors.append(
            f"deployment_manifest.score_direction must be '{_TASK_A_SCORE_DIRECTION}', "
            f"got '{dm.get('score_direction')}'"
        )
    if dm.get("threshold_policy") != _TASK_A_THRESHOLD_POLICY:
        errors.append(
            f"deployment_manifest.threshold_policy must be '{_TASK_A_THRESHOLD_POLICY}', "
            f"got '{dm.get('threshold_policy')}'"
        )
    try:
        dm_quantile = float(dm.get("threshold_quantile", 0))
    except (TypeError, ValueError):
        errors.append(f"deployment_manifest.threshold_quantile is not numeric: {dm.get('threshold_quantile')!r}")
        dm_quantile = None
    if dm_quantile is not None and not math.isclose(dm_quantile, _TASK_A_THRESHOLD_QUANTILE, rel_tol=1e-9):
        errors.append(
            f"deployment_manifest.threshold_quantile must be {_TASK_A_THRESHOLD_QUANTILE}, "
            f"got '{dm.get('threshold_quantile')}'"
        )

    return errors


def validate_artifact_contract(artifact_dir: Path, *, task: str | None = None) -> list[str]:
    errors: list[str] = []
    required = [
        "deployment_manifest.json",
        "run_manifest.json",
        "global_metrics.json",
        "per_class_metrics.json",
        "features_manifest.json",
    ]
    for name in required:
        if not (artifact_dir / name).exists():
            errors.append(f"missing required file: {name}")

    gm_path = artifact_dir / "global_metrics.json"
    if gm_path.exists():
        payload = json.loads(gm_path.read_text(encoding="utf-8"))
        if not _validate_numeric_json(payload):
            errors.append("global_metrics.json contains non-numeric values (check for unexpected string fields)")

    dm_path = artifact_dir / "deployment_manifest.json"
    dm: dict = {}
    if dm_path.exists():
        dm = json.loads(dm_path.read_text(encoding="utf-8"))
        for key in ("task", "model", "model_family", "model_artifact", "feature_names"):
            if key not in dm:
                errors.append(f"deployment_manifest.json missing key: {key}")
        model_artifact = dm.get("model_artifact")
        if isinstance(model_artifact, str) and model_artifact:
            if not (artifact_dir / model_artifact).exists():
                errors.append(f"model_artifact does not exist: {model_artifact}")
        else:
            errors.append("deployment_manifest.json has empty model_artifact")

        scaler_artifact = dm.get("scaler_artifact")
        if scaler_artifact and not (artifact_dir / str(scaler_artifact)).exists():
            errors.append(f"scaler_artifact does not exist: {scaler_artifact}")

        feature_names = dm.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            errors.append("deployment_manifest.json feature_names must be a non-empty list")

    # Task A operating-point contract — applied when task is explicitly "anomaly"
    # or when the deployment manifest declares task == "anomaly_semisup" / "anomaly_supervised".
    # Skipped for Task B (classification) and DLSSM (pending methodology alignment).
    resolved_task = task or dm.get("task", "")
    is_task_a = str(resolved_task).startswith("anomaly")
    model_family = dm.get("model_family", "")
    is_dlssm = "dlssm" in str(dm.get("model", "")).lower() or "dlssm" in model_family.lower()

    if is_task_a and not is_dlssm and dm_path.exists():
        errors.extend(_validate_task_a_operating_point(artifact_dir, dm))

    # Task B probability calibration contract
    if dm_path.exists() and dm.get("confidence_is_calibrated") is True:
        errors.extend(_validate_task_b_calibration(artifact_dir, dm))

    return errors


def _validate_task_b_calibration(artifact_dir: Path, dm: dict) -> list[str]:
    """Enforce calibration contract when confidence_is_calibrated == true."""
    errors: list[str] = []

    cal_name = dm.get("probability_calibration_artifact")
    if not cal_name:
        errors.append(
            "deployment_manifest.json missing probability_calibration_artifact "
            "(required when confidence_is_calibrated is true)"
        )
        return errors

    cal_path = artifact_dir / str(cal_name)
    if not cal_path.exists():
        errors.append(f"probability_calibration_artifact does not exist: {cal_name}")
        return errors

    try:
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"probability_calibration.json unreadable: {exc}")
        return errors

    method = cal.get("calibration_method")
    if method not in ("sigmoid", "isotonic"):
        errors.append(
            f"probability_calibration.calibration_method must be 'sigmoid' or 'isotonic', "
            f"got {method!r}"
        )

    if cal.get("test_not_used_for_calibration") is not True:
        errors.append("probability_calibration.test_not_used_for_calibration must be true")

    dm_method = dm.get("calibration_method")
    if dm_method and dm_method != method:
        errors.append(
            f"deployment_manifest.calibration_method ({dm_method!r}) does not match "
            f"probability_calibration.calibration_method ({method!r})"
        )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deployment artifact contract")
    parser.add_argument("artifact_dir", help="Artifact directory path")
    parser.add_argument(
        "--task",
        default=None,
        help="Override task type for contract selection (e.g. anomaly, classification)",
    )
    args = parser.parse_args()

    root = Path(args.artifact_dir)
    errors = validate_artifact_contract(root, task=args.task)
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        raise SystemExit(1)
    print("Artifact contract validation passed")


if __name__ == "__main__":
    main()
