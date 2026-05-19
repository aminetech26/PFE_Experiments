# Feature Engineering Pipeline — Implemented Runtime Behavior

This document explains what the current feature-engineering pipeline actually does in code (`PFE_Experiments/src/data/featurize_pipeline.py` and `PFE_Experiments/src/data/features.py`), with a focus on EDA priors and tsfresh controls.

## 1) Pipeline flow (current implementation)

1. Resolve selected task split using `--task` (`anomaly_semisup`, `anomaly_supervised`, `classification`).
2. Resolve split path using `--split-path` (`path_a` or `path_b`).
3. Load preprocessed splits from one of:
   - `data/processed/preprocessed/<dataset>/<task>/{train,val,test}.parquet` (Path A)
   - `data/processed/preprocessed/<dataset>/path_b/<task>/{train,val,test}.parquet` (Path B)
4. Resolve base flags from global config, then apply dataset/path overrides and task directives.
5. Apply feature profile (`--profile`) as the final override layer.
6. Load EDA findings from `feature_engineering.selection.eda_findings_path` when enabled.
7. Build base features from dataset sensor columns present in data.
8. Add optional engineered features based on effective flags (physics, rolling statistics, wavelet/spectral, differential signal, optional temporal context).
   - Physics family now includes compact Costa-style imbalance descriptors:
     - `power_imbalance`
     - `current_imbalance`
     - `voltage_imbalance`
     - optional directional shares (`string1_power_share`, `string1_current_share`)
   - Physics family also includes temperature-aware power correction (when enabled):
     - `temp_loss_pmax`
     - `pdc_temp_corrected`
     - `pdc_temp_corrected_norm_irr`
   - For Costa-style schemas, derivative helpers use dataset-available channels:
     - `dP_dt` from `pdc` (fallback: `pdc1+pdc2`)
     - `dI_dt` from `idc1+idc2`
     - `dV_dt` from mean DC voltage `(vdc1+vdc2)/2`
   - Costa temperature-correction defaults use module datasheet coefficient (`gamma_Pmax = -0.40 %/C` for CS6U-330P).
9. Build candidate set from the admissible base + engineered columns.
10. Apply train-only hygiene pruning (near-constants, exact duplicates, deterministic aliases) when enabled.
11. Apply train-only mRMR selection (MID criterion) when enabled.
12. Apply train-only correlation pruning and train-only VIF pruning only for explicit late ablations or cleanup checks (both disabled by default).
13. Optionally run tsfresh segment-level extraction and merge selected tsfresh features back to train/val/test.
14. Write output run artifacts to:
  - `data/processed/features/<dataset>/<task>/runs/<profile>/` (Path A)
  - `data/processed/features/<dataset>/path_b/<task>/runs/<profile>/` (Path B)
  - files: `train.parquet`, `val.parquet`, `test.parquet`, `features_manifest.json`, `resolved_config.json`
  - reruns keep the profile name when available and append a timestamp suffix only to avoid overwriting an existing run directory
  - pointer: `data/processed/features/latest_runs.json`

CLI examples:
- `uv run python -m src.data.featurize_pipeline --dataset costa --task classification --split-path path_a --profile baseline_raw`
- `uv run python -m src.data.featurize_pipeline --dataset costa --task classification --split-path path_b --profile baseline_raw`

Default task behavior:
- If `--task` is omitted, the pipeline uses `feature_engineering.task_directives.default_task` (currently `classification`).

---

## 2) Task directives (`feature_engineering.task_directives`)

Task directives are layered with the same config style as flags/profiles:
- base config (`flags`, `selection`, `tsfresh`)
- profile overrides (`--profile`)
- task directives `common`
- task directives for selected task

Supported task-directive keys:
- `flags`: per-task toggles for `enable_*`
- `selection`: per-task selection overrides
- `tsfresh`: per-task tsfresh parameter overrides
- `eda`: EDA policy overrides (`mi_top_key`, `use_mannwhitney`)
- `tsfresh_label_strategy`: segment target strategy (`any_fault`, `majority_label`, `fault_fraction`)

Current layered resolution order:
- base config (`flags`, `selection`, `tsfresh`)
- dataset + split-path overrides
- task directives `common`
- task directives for selected task
- profile overrides (`--profile`, final precedence)

Costa preprocessing invariant relevant to FE:
- Preprocessing now drops normal-class outlier rows using IQR bounds fitted on primary measured channels (`vdc1`, `vdc2`, `idc1`, `idc2`, `irr`, `pvt`).
- `pdc1`, `pdc2`, `pdc` are restored from cleaned primaries before FE (`pdc1=vdc1*idc1`, `pdc2=vdc2*idc2`, `pdc=pdc1+pdc2`), so deterministic alias checks in hygiene are physically meaningful again.

Path-level admissibility guards currently enforced for Costa-oriented workflow:
- Path A disables `enable_hour_cyclic` by default policy
- Path A disables `enable_wavelet` (spectral features are Path B only)

---

## 3) Selection priors from EDA (`feature_engineering.selection`)

### `anchor_features` (dataset-specific)
- **Role:** protected/priority features during pruning.
- **Used in:** optional correlation pruning and optional VIF pruning (`anchor_cols` argument).
- **Effect:** if a correlated/VIF conflict happens, anchors are preferred to stay when possible.

### `eda_prefer_anchors_from_findings: true`
- **Role:** augment anchors from EDA evidence.
- **Behavior in code:**
  - Start from configured `anchor_features`.
  - Add up to top-5 features from EDA mutual information key selected by task policy (`top_features_binary` or `top_features_multiclass`).
  - Optionally add up to top-5 features from EDA Mann-Whitney (`significant_features`) based on task policy.
  - Final anchor set is de-duplicated and sorted.
- **Important:** this **adds** anchors; it does not discard configured anchors.

### `eda_pre_drop_candidates: false`
- **Current policy:** automatic EDA pre-drop is disabled.
- **Rationale:** EDA correlation findings are now treated as design guidance, not as automatic deletion rules.
- **Methodological effect:** raw source channels remain available long enough to build derived feature families
  (for example imbalance descriptors) before train-only hygiene + mRMR take over final selection.

### `eda_override_thresholds: false`
- **Role:** control whether EDA-recommended thresholds replace configured thresholds.
- **When `false` (current):**
  - Keep config values (`corr_threshold`, `vif_threshold`) as source of truth.
- **When `true`:**
  - Override with EDA values if present:
    - `spearman.recommended_corr_threshold`
    - `vif.recommended_vif_threshold`

---

## 4) tsfresh configuration (`feature_engineering.tsfresh`)

### `mode: minimal` (`off | minimal | extensive`)
- `off`: skip tsfresh completely.
- `minimal`: `MinimalFCParameters()` in tsfresh.
- `extensive`: `ComprehensiveFCParameters()` in tsfresh.

### `top_k: 20`
- Cap on number of selected tsfresh columns added to the dataset.
- Flow:
  - tsfresh extracts features on sampled train segments,
  - optional `select_features(...)` is applied when train labels are not single-class,
  - first `top_k` columns are kept and prefixed as `tsfresh__...`.

### `n_segments_sample: 60`
- Number of unique train segments used to fit/select tsfresh features.
- Current behavior uses the first N unique segments (not random sampling).

### `max_rows_per_segment: 800`
- For each segment, tsfresh input is truncated to at most this many rows (`head(max_rows_per_segment)` after time sort).
- Limits memory/compute while preserving early segment trajectory.

### `n_jobs`
- Parallelism control passed to tsfresh.
- Current code normalizes values `<= 0` to `max(1, os.cpu_count())` for safe execution.
- Current repo default in `configs/data_config.yaml` is `n_jobs: 1`.

---

## 5) Hygiene + mRMR (current Costa-first default for `plus_physics`)

- `enable_hygiene_pruning`: applies light pre-selection hygiene on train split only.
  - near-constants (`near_constant_std` threshold)
  - exact duplicate columns
  - deterministic Costa alias removal (`pdc` dropped when exactly equal to `pdc1 + pdc2`)
- `enable_mrmr_selection`: greedy mRMR with MID objective (`relevance - redundancy`) on train split only.
- `mrmr_k`: target retained feature count from candidate pool after hygiene.
- EDA findings still matter here indirectly by informing feature-family design and optionally augmenting anchor features for late corr/VIF checks.
- Corr/VIF are now treated as optional post-checks rather than the primary selector stack, and all named profiles keep them off by default.

## 6) Explicit windowing and multi-scale windowing

Windowing collapses N input rows into M output rows (M << N, one row per window). This is fundamentally incompatible with row-aligned features — the two cannot coexist in the same DataFrame.

**Pipeline order when windowing is enabled:**
1. `add_optional_features` runs as normal → row-aligned features (physics, rolling stats, etc.)
2. Rolling stats are **auto-disabled** by pipeline policy if `enable_explicit_windowing` is true (logged at runtime).
3. Windowing block runs after `add_optional_features`, before corr/VIF pruning.
4. `candidate_features` is rebuilt from the windowed column names; original base columns no longer exist.
5. If explicitly enabled for an ablation, Corr/VIF pruning operates on the windowed data.

**`add_multiscale_window_features` (features.py):**
- Input: row-aligned DataFrame + feature column list
- `window_sizes`: list of scales, e.g. `[60, 30, 10]` — sorted descending
- Primary window = largest value; determines output row count and step
- Per primary window: nested sub-windows = last `w` samples of the primary window for each sub-scale
- Stats extracted per scale using `extract_window_statistics` (8 stats: mean, std, min, max, skew, kurtosis, rms, zcr)
- Spectral features extracted on the primary window only (see §6)
- Label per window = mode label within the primary window
- All segment boundaries respected — windows never cross `segment_id` boundaries
- Output column naming: `{feat}_w{scale}_{stat}`, `{feat}_wpd_band{b}`, `{feat}_ceemdan_imf{k}`

**Windowing manifest fields** (in `features_manifest.json → explicit_windowing`):
- `enabled`, `window_sizes`, `window_step`, `primary_window`, `multiscale`, `spectral`, `n_top_freqs`, `n_windows_per_split`

**Profile ladder (windowed track):**
| Profile | Encoding |
|---------|----------|
| `plus_windowed` | Physics → single-scale windows (size 60, step 30) |
| `plus_multiscale` | Physics → multi-scale windows [60, 30, 10], step 30 |
| `plus_wpd` | Physics → multi-scale + WPD only (Path B only) |
| `plus_ceemdan` | Physics → multi-scale + CEEMDAN only (Path B only, slow) |
| `plus_wpd_ceemdan` | Physics → multi-scale + WPD + CEEMDAN (Path B only, hybrid) |

---

## 7) Spectral features (Path B only)

All spectral methods are Path B only. The Path A policy guard at runtime auto-disables `enable_wpd` and `enable_ceemdan` if `split_path == "path_a"` (same mechanism as wavelet).

**Binding constraint:** Costa 1 Hz → Nyquist 0.5 Hz, frequency resolution 0.017 Hz for a 60-sample window.

### WPD (Wavelet Packet Decomposition) — `enable_wpd`
- Method: full binary tree decomposition of each window; both approximation and detail subbands at each level
- Features: energy per subband = `sum(coeff² )` normalized by total window energy
- Column naming: `{feat}_wpd_band{b}` (b = 0..2^level-1)
- Config: `wpd_level: 3` → 8 subbands; wavelet family `db4` (consistent with denoising)
- Rationale: sampling-rate agnostic, handles transients, most physically grounded at 1 Hz
- PyWavelets implementation: `pywt.WaveletPacket`
- Node policy: keep all terminal nodes at the chosen level in frequency order; no manual node picking or best-basis search is applied in the current pipeline.
- Selection policy: WPD node energies enter the common selector stack unchanged, then train-only hygiene + mRMR decide which subbands matter.

### CEEMDAN — `enable_ceemdan`
- Method: Complete Ensemble EMD with Adaptive Noise; decomposes signal into IMFs adaptively
- Features: energy ratio per IMF (IMF energy / total signal energy) for first K IMFs
- Column naming: `{feat}_ceemdan_imf{k}` (k = 0..n_imfs-1); zero-padded if signal yields fewer IMFs
- Config: `ceemdan_n_imfs: 3`, `ceemdan_n_ensemble: 20`
- Slow: N_ensemble EMD trials per window per feature — profile it before running at full scale
- Package/backend: `EMD-signal` / `PyEMD` (`from PyEMD import CEEMDAN`)
- Guard: auto-skip if effective sampling rate < `spectral_min_fs`
- Output policy: keep only the first `K` IMF energy ratios as the CEEMDAN summary; no residue feature or per-IMF shape statistics are added in the current branch.

### Combination policy
- WPD-only is the primary fixed-basis spectral branch.
- CEEMDAN-only is the primary adaptive spectral branch.
- WPD + CEEMDAN is the hybrid branch used to test complementarity between fixed-basis and adaptive decompositions.

### Costa differential policy
- `plus_differential` remains a supported profile family globally, but it is runtime-disabled for Costa.
- Reason: Costa has no healthy-reference channel pair (`Pg_inv1/Pg_inv2` or `Pg/Pg_ref`) required to form meaningful `delta_p`.

### `tsfresh_label_strategy` (task directive)
- Controls how segment targets are built for tsfresh `select_features(...)` on train split.
- Supported values:
  - `any_fault`: segment is 1 if any fault exists in segment.
  - `majority_label`: segment target is modal class label in segment.
  - `fault_fraction`: segment target is fraction of fault rows in segment.

---

## 8) Rolling / Window Statistics Status

- `add_rolling_statistics_features(...)` is now integrated behind flags:
  - `enable_rolling_stats`
  - `rolling_windows`
- rolling features are causal, segment-safe, and row-aligned.

Current status of explicit window-stat helpers:

- `extract_window_statistics(...)` exists in `src/data/features.py`.
- It computes per-window summary stats (mean/std/min/max/skew/kurtosis/RMS/ZCR).
- **Current status:** explicit window-stat extraction is wired through `add_multiscale_window_features(...)` when `enable_explicit_windowing=true`.
- Practical implication: run artifacts can be either row-aligned or window-collapsed, depending on active profile flags.

## 9) Handcrafted vs Automatic Statistics

- Handcrafted statistics (rolling/window families) are explicit and hypothesis-driven.
- tsfresh remains available as automated statistical extraction.
- Current methodology treats tsfresh as an optional comparator branch, not the sole statistical feature strategy.

---

## 10) Why these settings matter together

- Task directives allow one shared feature engineering pipeline to adapt behavior per task without forking code paths.
- EDA priors provide a strong starting point through anchors and interpretive guidance, then train-only pruning enforces leakage-safe final selection.
- `eda_override_thresholds: false` keeps experiment control in config while still benefiting from EDA hints.
- `tsfresh` remains available as a bounded comparator branch, but current named profiles keep it off by default.
- Task-aware run directory fingerprinting guarantees reproducible, non-overwriting outputs across profile/config/task changes.

---

## 11) Output directives and lineage

### `latest_runs.json`
- Stores per-task latest run pointers:
  - `latest_by_task.<task>` => `"<task>/runs/<profile>"`
- Stores per-task profile pointers:
  - `latest_by_task_profile.<task>.<profile>` => `"<task>/runs/<profile>"`
- Stores latest execution metadata:
  - `last_run.task`, `last_run.profile`, `last_run.path`
- Includes `updated_at` timestamp.

### Manifest lineage fields
`features_manifest.json` includes task-aware lineage:
- `source_task`
- `source_preprocessed_dir`
- `task_directives_effective`
- `selection.eda_policy`

`resolved_config.json` includes resolved IO context:
- `io.input_dir`
- `io.output_root`
- `io.run_dir`
   - For large training splits, mRMR now uses a representative capped train subset (`selection.max_mrmr_rows`, default 30k) to keep MI-based relevance/redundancy tractable.
   - The cap uses proportional label stratification and group-aware temporal spacing (`episode_id` -> `segment_id` -> `operating_day_id`) rather than naive row-random truncation.
