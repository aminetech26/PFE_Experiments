# PFE_Experiments — PC-Flow: Physics-Conditioned Fault Detection & Diagnosis for PV Systems

End-to-end pipeline for photovoltaic (PV) fault detection and diagnosis (FDD), built on real field data and validated for edge deployment. This is the experiment codebase behind **PC-Flow**, a lightweight physics-conditioned normalizing flow for anomaly detection, paired with an Extra Trees classifier for fault diagnosis, deployed end-to-end on an NVIDIA Jetson Nano.

Developed as a final-year engineering project (PFE) at École nationale Supérieure d'Informatique (ESI), Algiers, hosted at the Centre de Développement des Énergies Renouvelables (CDER).

## What's here

- **Data pipeline** (`src/data/`): ingestion, temporal splitting, preprocessing, and physics-informed feature engineering for the [Costa PV fault dataset](https://github.com/clayton-h-costa/pv_fault_dataset), with leakage-prevention enforced at every stage (partition-first splitting, train-only statistics).
- **Anomaly detection** (`src/modeling/anomaly_detection/`): PC-Flow and five baselines (BOCD, Isolation Forest, OC-SVM, MAAT, GTBAD), evaluated under 5-fold episode-stratified cross-validation.
- **Fault classification** (`src/modeling/classification/`): Extra Trees, LightGBM, and CatBoost for 4-class fault diagnosis.
- **Deployment** (`src/deployment/`): ONNX export, artifact contract validation, and inference profiling for the Jetson Nano target.
- **Experiment tracking**: DVC (data/model versioning) + MLflow via DagsHub (metrics/runs).

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` (if present) or set the following environment variables for DagsHub-backed DVC/MLflow tracking:

```
DAGSHUB_USERNAME=<your DagsHub username>
DAGSHUB_USER_TOKEN=<your DagsHub token>   # never commit this — see .gitignore
```

Pull DVC-tracked data and model artifacts:

```bash
dvc pull
```

## Running the pipeline

The data pipeline is wired as a DVC DAG:

```bash
dvc repro
```

This runs, in order: `ingest` → `split` → `preprocess` → `featurize` → `train_classification`, reading all stage parameters from `configs/data_config.yaml` and `configs/model_config.yaml`.

Anomaly detection models are run directly (not yet a DVC stage), e.g.:

```bash
uv run python -m src.modeling.anomaly_detection.ml.run \
  --model pc_flow --task anomaly_semisup --dataset costa \
  --profile baseline_raw --run-type baseline --seed 42
```

Swap `--model` for any of `bocd`, `isolation_forest`, `ocsvm`, `maat`, `gtbad` to run a baseline instead. Fault classification baselines are trained via `src.modeling.classification.ml.run` (see `dvc.yaml` for the exact invocation).

Colab is supported for GPU-bound training runs; see `notebooks/colab_baseline_runner.ipynb` (mounts Google Drive for persistent artifacts, reads secrets from Colab's built-in Secrets manager rather than hardcoding them).

## Data

The [Costa dataset](https://github.com/clayton-h-costa/pv_fault_dataset) (public) is used throughout: 1.37M measurements, four induced fault types (short-circuit, open-circuit, degradation, partial shadowing) plus normal operation. Raw data is DVC-tracked (`data/raw.dvc`), not committed to git.

## Configuration

All experiment behavior is driven by YAML configs, not hardcoded in scripts:

- `configs/data_config.yaml` — dataset paths, splitting, preprocessing, feature profiles
- `configs/model_config.yaml` — hyperparameters and HPO search spaces per model/task
- `configs/deploy_config.yaml` — deployment target, runtime, artifact contract, alarm policy

## Related work

This codebase supports the preprint *"PC-Flow: An Efficient, Lightweight Physics-Conditioned Normalizing Flow for Anomaly Detection in a Real-Data, Edge-Deployed PV Fault Diagnosis System"* (in preparation, 2026).

## License

Not yet specified.
