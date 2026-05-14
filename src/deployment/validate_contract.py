from __future__ import annotations

import argparse
import json
from pathlib import Path


def _is_numeric_leaf(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_numeric_json(payload: object) -> bool:
    if isinstance(payload, dict):
        return all(_validate_numeric_json(v) for v in payload.values())
    if isinstance(payload, list):
        return all(_validate_numeric_json(v) for v in payload)
    return _is_numeric_leaf(payload)


def validate_artifact_contract(artifact_dir: Path) -> list[str]:
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
            errors.append("global_metrics.json contains non-numeric values")

    dm_path = artifact_dir / "deployment_manifest.json"
    if dm_path.exists():
        dm = json.loads(dm_path.read_text(encoding="utf-8"))
        for key in ("task", "model", "model_family", "model_artifact", "feature_names"):
            if key not in dm:
                errors.append(f"deployment_manifest.json missing key: {key}")
        model_artifact = dm.get("model_artifact")
        if isinstance(model_artifact, str) and model_artifact:
            model_path = artifact_dir / model_artifact
            if not model_path.exists():
                errors.append(f"model_artifact does not exist: {model_artifact}")
        else:
            errors.append("deployment_manifest.json has empty model_artifact")

        scaler_artifact = dm.get("scaler_artifact")
        if scaler_artifact:
            scaler_path = artifact_dir / str(scaler_artifact)
            if not scaler_path.exists():
                errors.append(f"scaler_artifact does not exist: {scaler_artifact}")

        feature_names = dm.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            errors.append("deployment_manifest.json feature_names must be a non-empty list")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deployment artifact contract")
    parser.add_argument("artifact_dir", help="Artifact directory path")
    args = parser.parse_args()

    root = Path(args.artifact_dir)
    errors = validate_artifact_contract(root)
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        raise SystemExit(1)
    print("Artifact contract validation passed")


if __name__ == "__main__":
    main()
