# Thesis Log

## 2026-05-01

### Costa FE profile semantics and path-policy corrections

- Observation: several Costa feature profiles looked active by name but were functionally redundant or policy-inconsistent (notably `plus_wavelet`, and Path A/Path B family overlap).
- Interpretation: this weakens ablation validity because profile names no longer map cleanly to distinct feature families.
- Decision: remove `plus_wavelet`, exclude `plus_differential` from Costa generation, enforce Path A profile exclusions directly in the Costa batch generator, and keep advanced spectral/window families as Path B-oriented branches.
- Consequence: profile naming now reflects actual generated families, and run registries are less likely to contain misleading profile artifacts.

### Window-statistics robustness cleanup

- Observation: skew/kurtosis repeatedly triggered catastrophic-cancellation warnings and produced NaNs in Costa windowed artifacts.
- Interpretation: these higher moments are numerically fragile on near-constant windows and were degrading FE artifact quality.
- Decision: remove skew/kurtosis from rolling/window statistics families in the active Costa setup.
- Consequence: saved feature sets are cleaner and more modeling-ready for tabular baselines; temporal-statistics families now prioritize stable descriptors (`mean/std/min/max/rms/zcr`).

### CEEMDAN stability hardening

- Observation: CEEMDAN emitted repeated divide-by-zero/invalid-value warnings on low-energy windows.
- Interpretation: raw CEEMDAN calls were not guarded against degenerate windows, causing unstable decomposition paths.
- Decision: add CEEMDAN guards (flat/low-energy skip, non-finite sanitization, failure counters, and guarded warnings) before/after decomposition.
- Consequence: CEEMDAN runs now fail less noisily and expose skip/failure behavior explicitly, enabling evidence-based keep/drop decisions for Costa spectral branches.

### Costa preprocessing invariant restoration (`pdc = pdc1 + pdc2`)

- Observation: ingestion guarantees `pdc = pdc1 + pdc2`, but preprocessing outlier handling on all sensor columns broke this identity in ~6% of normal rows.
- Interpretation: independent winsorization of derived power channels (`pdc`, `pdc1`, `pdc2`) creates avoidable physical inconsistency and confuses deterministic-alias hygiene.
- Decision: preprocess Costa primary measured channels only (`vdc1`, `vdc2`, `idc1`, `idc2`, `irr`, `pvt`), then recompute `pdc1`, `pdc2`, `pdc` after preprocessing.
- Consequence: preprocessed Costa artifacts now preserve exact power-channel identities, and FE hygiene correctly drops deterministic `pdc` aliases again.

### Selection-policy activation consistency

- Observation: only `plus_physics` explicitly enabled hygiene+mRMR while richer profiles inherited global disabled defaults.
- Interpretation: this silently disabled the intended train-only selector stack in many ablation profiles.
- Decision: enable hygiene and mRMR globally, while keeping `baseline_raw` explicitly unpruned/unselected as a control profile.
- Consequence: richer profiles now consistently apply the thesis-default selector posture; baseline remains a clean reference.

### mRMR runtime stall on rich Costa profiles

- Observation: `anomaly_supervised + plus_rolling` appeared frozen for hours after hygiene pruning because MI-based mRMR redundancy on full train rows became computationally prohibitive.
- Interpretation: full-row MI redundancy is not scalable on high-row/high-feature combinations and needs a representative cap.
- Decision: add `selection.max_mrmr_rows` (default 30k) and run mRMR on a representative train subset using proportional label stratification with group-aware temporal spacing (episode/segment/day), not naive random truncation.
- Consequence: mRMR stays methodologically valid, becomes tractable on rich profiles, and now reports intermediate progress during selection.

### OCSVM subsampling policy upgrade

- Observation: OCSVM subsampling used seeded uniform row-random selection, which is reproducible but regime-coverage weak under 1 Hz autocorrelation.
- Interpretation: random row sampling overrepresents dense stretches and underuses group structure (`episode_id`/`operating_day_id`).
- Decision: replace row-random cap sampling with group-quota sampling plus within-group temporal spacing, with split-path-aware group priority and explicit sampling metadata logging.
- Consequence: capped OCSVM training (`max_train_samples`) remains computationally practical while becoming more representative and auditable.

### OCSVM RBF tuning policy (Costa semisup, Path A)

- Observation: early anomaly runs showed strong `rbf` behavior and unstable/poor `poly` behavior for the same Costa semisup regime.
- Interpretation: the immediate gain is to deepen `rbf` search quality rather than split budget across kernels.
- Decision: increase OCSVM HPO trial budget to 50 and replace coarse categorical RBF gamma with continuous log-scale search; tighten `nu` range toward precision-recall tradeoff sweet spots.
- Consequence: tuning now explores smoother and more informative RBF decision-boundary regimes for `plus_physics` while preserving the same semisup protocol and train-cap policy.

## 2026-05-06

### R1 classification baseline results — Costa, Path A

All three R1 classifiers were run on Costa Path A (features: 9 raw channels for `baseline_raw`, +12 physics for `plus_physics`, +rolling stats for `plus_rolling`; seeds: [42, 777, 1234]; 4 fault classes).

Results:

| Model | Best Profile | Seed | f1_weighted | accuracy | pr_auc_weighted | Top Feature |
|---|---|---|---|---|---|---|
| LightGBM | baseline_raw | 42 | 0.99948 | 0.9928 | 0.9974 | vdc2 (22.7%) |
| CatBoost | plus_physics | 38 | 0.99939 | 0.9928 | 0.9986 | vdc2 / voltage_imbalance (17–27%) |
| Extra Trees | baseline_raw | 42 | 0.99913 | 0.9929–0.9966 | 0.9979 | voltage_imbalance (17–39%) |

Label shuffle test: passed (all three models). This rules out the most obvious leakage.

### R1 classification leakage investigation — resolved as non-leakage

Observation: near-ceiling f1_weighted (~0.999) across three independent classifiers raised a sanity/leakage flag.

Investigation performed:

1. **Test set size check** — test set is 35,312 rows (class 1: 1,182 / class 2: 596 / class 3: 1,201 / class 4: 32,333). The minority class (class 2) has 596 test rows — 99.9% f1 implies ~2–3 misclassifications, which is statistically meaningful at that scale.

2. **Segment integrity check** — split manifest verified: `group_column = episode_id`. Episode IDs are strictly ordered across splits (train IDs < val IDs < test IDs for every class). No episode appears in more than one split. Embargo logic enforced in `segment_stratified_split`. Split is clean.

3. **Label shuffle test** — already passed during training runs for all three models, confirming real signal is being learned.

Resolution: results are genuine. See dataset nature note below.

### Costa dataset nature and classification saturation explanation

Costa is a **real PV installation** where faults were **artificially induced** under a controlled experimental protocol. It is not a simulated dataset. Mendeley/GPVS-Faults is the simulated dataset (circuit simulator output).

Why near-ceiling classification is expected on Costa:

- Artificially induced faults produce stronger and more consistent electrical signatures than naturally occurring faults. When a fault is physically induced at a known severity and condition, the resulting signal is unambiguous.
- The four fault classes correspond to physically distinct electrical failure modes (e.g., line-to-line short, partial shading, open circuit, degraded module). These produce fundamentally different voltage/current patterns — a line-to-line fault immediately collapses string voltage; an open circuit drops current to zero.
- At 1 Hz under stable fault conditions, consecutive samples within an induced fault episode are near-identical. The classifier recognizes a consistent electrical state rather than needing to generalize across noisy or evolving manifestations.

This is an expected artifact of the controlled induction protocol, not evidence of leakage.

### Thesis framing — controlled induction vs field-deployed natural faults

The near-ceiling Costa results motivate the core research question rather than undermining it:

> "Classical ML saturates on controlled fault induction (Costa, f1_weighted ≈ 0.999) because fault signatures are consistent and physically distinct. La Réunion presents naturally occurring faults: progressive onset, unknown severity, real sensor noise, installation-specific variability. The performance gap between these two settings is the deployment challenge — domain generalization across the controlled-to-field axis is the research contribution."

This framing is stronger than a simulation-vs-reality contrast because it reflects the real deployment gap: controlled lab induction is how datasets are built; field-deployed natural faults are what a production FDD system actually encounters.

### OCSVM RBF retuning (narrowed search after over-aggressive fit)

- Observation: a broader RBF search with 50 trials improved validation slightly but degraded test PR-AUC/F1/accuracy on Costa `plus_physics`.
- Interpretation: the wider regime likely selected an overly sharp boundary (high gamma / high nu tendency) that overfit validation dynamics.
- Decision: narrow RBF search to a conservative region: `nu` in `[0.03, 0.10]` and `gamma` in `[0.01, 0.08]` (log scale).
- Consequence: the next tuning round prioritizes stable generalization over aggressive frontier tightness while keeping the same split/profile/sampling protocol.

### Isolation Forest baseline implementation (Task A, ML)

- Observation: the anomaly ML dispatcher scaffold listed Isolation Forest but did not execute it.
- Interpretation: to keep baseline comparability and accelerate tournament progression, Isolation Forest should follow the exact OCSVM experiment contract (same data loading, metrics, thresholding, artifacts, MLflow, and comparison logging).
- Decision: implement `isolation_forest_model.py` with the same semisup evaluation flow, Optuna-on-validation PR-AUC, validation-calibrated thresholding, and shared comparison record schema.
- Consequence: OCSVM and Isolation Forest results are now apples-to-apples comparable on Costa (`task/split/profile/seed`), enabling a cleaner Task A baseline matrix.

### Isolation Forest tuning refinement after first Costa results

- Observation: the first `plus_physics` Isolation Forest run substantially improved PR-AUC/F1 over `baseline_raw`, with a best trial favoring larger forests and larger subsamples (`n_estimators=500`, `max_samples=2048`, `max_features=0.6`).
- Interpretation: IF performance on Costa benefits from higher tree count, larger sample envelopes, and moderate feature subsampling rather than full-feature trees.
- Decision: move the IF HPO search region upward and tighter: `n_estimators` `[400..1000]`, `max_samples` `[1024, 2048, 4096, auto]`, `max_features` `[0.4..0.8]`.
- Consequence: the next IF tuning cycle prioritizes improved ranking quality and stability around the empirically better regime instead of revisiting weak low-capacity settings.

### XGBoost anomaly baseline implementation

- Observation: the anomaly dispatcher listed XGBoost as scaffolded, but no runnable trainer existed for Task A.
- Interpretation: XGBoost should be treated as a supervised binary anomaly baseline (`normal` vs `any fault`), not as a one-class novelty model.
- Decision: implement `src/modeling/anomaly_detection/ml/xgboost_model.py`, wire dispatcher routing, and add a dedicated anomaly XGBoost config block with Optuna search space.
- Consequence: Task A now includes a third ML baseline with the same experiment contract (feature run loading, validation PR-AUC optimization, threshold calibration, artifact/MLflow/comparison-record logging) for apples-to-apples comparison with OCSVM and IF.

### XGBoost tuning refinement for Costa `plus_physics`

- Observation: `plus_physics` was validated as superior, so anomaly XGBoost should focus its tuning budget on that profile rather than profile-wide generic ranges.
- Interpretation: prior XGBoost ranges were broad and under-regularized for engineered-physics space, with no explicit class-weight multiplier search around the auto ratio.
- Decision: tighten search around a stable `plus_physics` regime (lower learning-rate band, moderate depth, stronger regularization), add `early_stopping_rounds`, and introduce `scale_pos_weight_multiplier` in HPO.
- Consequence: XGBoost now searches a more generalizable space for Costa anomaly supervision while preserving PR-AUC-driven selection and threshold-calibrated reporting.

## 2026-04-29

### Feature-selection policy cleanup

- Observation: the previous implementation still carried an EDA pre-drop layer, which conflicted with the newer family-first feature-engineering policy.
- Interpretation: if EDA redundancy findings are used as automatic deletion rules before or alongside selection, they can remove source channels needed to build compact physical descriptors and blur the methodological story.
- Decision: EDA correlation findings are now advisory only for feature-family design and anchor augmentation; automatic EDA pre-drop is disabled in the active pipeline/config.
- Consequence: the defended selector order is now: admissible feature families -> train-only hygiene pruning -> train-only mRMR -> optional corr/VIF post-checks.

### Corr/VIF default posture

- Observation: even as late-stage checks, correlation and especially VIF pruning can still be unnecessarily aggressive on rich multiscale and spectral branches.
- Interpretation: the cleanest default is to trust the hygiene -> mRMR stack first, and reserve corr/VIF for explicit ablations or cleanup experiments.
- Decision: all named feature profiles now keep `enable_corr_pruning=false` and `enable_vif_pruning=false` by default.
- Consequence: the thesis-default selector story is now fully consistent: hygiene first, mRMR second, corr/VIF only when deliberately switched on.

## 2026-04-24

### Modeling structure cleanup and Task A matrix-profile baseline

- Observation: the previous modeling layout mixed concerns and kept legacy compatibility paths that no longer help a solo workflow.
- Interpretation: clean per-task and per-category boundaries (`ml` / `dl`) are more maintainable and align better with thesis methodology.
- Decision: modeling code now lives under `src/modeling/<task>/<category>/` with shared utilities under `src/modeling/common/`; legacy `src/training/*` and compatibility wrappers were removed.
- Consequence: all new modeling work follows a strict decomposition policy (one model per file), with dispatchers per task/category.
- Decision: Task A first anomaly ML baseline is now implemented as Matrix Profile under `src/modeling/anomaly_detection/ml/matrix_profile_model.py`, executed on Costa `Path B` with `baseline_raw` features.
- Consequence: first reproducible Task A Path B run is logged in MLflow (`Task_A_Anomaly`) with saved artifacts: metrics JSON + PR curve + score histogram + score timeline.
- Note: Matrix Profile has no iterative training loop, so loss curves and early stopping are not applicable; thresholding/PR analysis is the correct evaluation lens.

### Feature selection hardening (Costa-first)

- Observation: correlation/VIF-only pruning is useful for diagnostics but is not the cleanest primary selector for the current Costa-first FE phase.
- Interpretation: a simpler selector stack is methodologically cleaner: light hygiene first, then one principled relevance-redundancy selector.
- Decision: feature selection flow now uses train-only hygiene pruning (near-constants, exact duplicates, deterministic aliases) followed by train-only mRMR (`relevance - redundancy`, MID criterion) with configurable `mrmr_k`.
- Consequence: `plus_physics` now defaults to `enable_hygiene_pruning=true`, `enable_mrmr_selection=true`, and keeps corr/VIF disabled by default; corr/VIF remain available as optional post-selection checks.

## 2026-04-23

### Split strategy

- Observation: Costa EDA shows that fault identity is better aligned with label-transition-aware episodes than with gap-only continuity segments.
- Interpretation: gap-only segmentation is sufficient for continuity control but weaker for leakage-safe supervised splitting and later windowing.
- Decision: Costa split generation uses `episode_id` as the canonical atomic split unit. Continuity-only grouping is retained as `continuity_segment_id` for diagnostics.
- Consequence: split manifests, preprocessing, and downstream feature generation now operate on episode-safe grouped units through the generic `segment_id` column.

### Split strategy fork

- Observation: the stricter episode-based split makes continuity-dependent preprocessing and signal-processing methods harder to justify scientifically on Costa.
- Interpretation: split design determines which downstream assumptions are admissible; one Costa split protocol is not enough to evaluate both event-safe and continuity-oriented methods fairly.
- Decision: Costa now keeps two named split paths. `Path A` is the canonical episode-based benchmark. `Path B` is a day-based chronological comparison path with a forward half-day purge on the first validation and test days.
- Consequence: `Path A` remains the primary leakage-safe benchmark, while `Path B` exists only to test methods that need stronger day-level continuity assumptions.

### Path B continuity assumption

- Observation: after `irr >= 100` trimming, Costa within-day retained sampling remains near 1 Hz for the overwhelming majority of consecutive same-day rows, despite a small tail of larger intra-day gaps.
- Interpretation: Costa does not become perfectly continuous, but a day-level quasi-continuity assumption is still defensible for a comparison path without forcing extra data loss on an already small retained-day sample.
- Decision: `Path B` treats retained calendar days as atomic units, tolerates residual intra-day gaps by assumption rather than splitting days further, and uses a forward half-day purge on the first validation and test days instead of dropping full days.
- Consequence: continuity-oriented stat/signal-processing methods can be evaluated in a bounded secondary protocol without weakening the main episode-based benchmark or sacrificing two full retained days by default.

### Path A vs Path B findings

- Observation: Path B improves validation/test support for several minority fault classes at the day-unit level, but it weakens training support for fault classification because whole retained days mix multiple labels.
- Interpretation: Path B is more useful as a secondary comparison path for detection-oriented or continuity-dependent experiments than as a primary fault-classification benchmark.
- Decision: Path A remains the authoritative benchmark path, especially for Task B classification. Path B is retained mainly for Task A and for continuity-dependent spectral / signal-processing feature engineering comparisons.
- Consequence: preprocessing must run cleanly on both paths, but the real methodological fork now lives in feature-engineering admissibility rather than preprocessing transforms.

### Preprocessing scope reduction

- Observation: irradiance normalization and related physics-driven transforms are now treated as representation design choices that belong with feature engineering rather than with the minimal preprocessing layer.
- Interpretation: Costa preprocessing should stay as lean as possible and avoid spending pipeline time on transforms that are now methodologically classified as engineered features.
- Decision: Costa preprocessing is reduced to outlier handling only, with missing-value handling disabled and irradiance-normalized power/current channels moved entirely to feature engineering.
- Consequence: preprocessed Costa artifacts now represent a minimal leakage-safe cleaned signal layer, and explicit normal-manifold shift awareness is delegated to handcrafted features later in the pipeline.

### Distribution Shift Framing

- Observation: the former "stationarity reduction" language was conflating forecasting-style assumptions with the actual need in FDD.
- Interpretation: for Costa, the key concern is normal-manifold distribution shift across operating conditions, not proving strict stationarity before modeling.
- Decision: irradiance-aware and related handcrafted transforms are now framed as feature-engineering mechanisms for distribution-shift awareness rather than preprocessing fixes for stationarity.
- Consequence: Path A uses only grouped-safe local features, while Path B exists mainly to justify stronger continuity-dependent spectral feature engineering.

### Stationarity

- Observation: single-episode ADF/KPSS checks were too narrow and often inconclusive after irradiance normalization and detrending.
- Interpretation: the issue is not irreproducibility but limited conclusiveness under a one-episode protocol.
- Decision: stationarity diagnostics are summarized across multiple long normal episodes and reported as supporting interpretive evidence rather than a hard modeling gate.
- Consequence: thesis claims about dependence and feature significance remain cautious, while split design and benchmark validity do not wait for a binary stationarity verdict.

### Windowing

- Observation: Costa mixes longer induced faults (`1-3`) with natural transient shadowing (`4`), so large windows may help some classes while blurring others.
- Interpretation: windowing is a modeling choice to be tested, not assumed.
- Decision: the canonical Costa baseline starts from non-windowed pointwise features plus causal derivatives/rolling context. Explicit windows remain an ablation axis.
- Consequence: later results must justify windowing empirically before it becomes part of the canonical benchmark narrative.

### Feature engineering pivot

- Observation: after preprocessing reduction, the meaningful methodological fork is feature engineering, not preprocessing.
- Interpretation: one strategy is required for handcrafted hardening (EDA-guided redundancy control + family-wise enrichment), and a separate strategy is required for automatic representation learning.
- Decision: feature engineering now proceeds in two tracks: (1) handcrafted hardening, then (2) representation-learning comparators (starting with TS2Vec / autoencoder-style embeddings).
- Consequence: tsfresh remains available as automated statistical extraction, but it is treated as an optional comparator branch rather than the sole statistics strategy.

### Temporal context policy

- Observation: time-of-day context is weakly justified under Path A event-centric splitting but can be useful in continuity-oriented Path B experiments.
- Interpretation: temporal cyclic features should not be a default Path A dependency.
- Decision: Path A keeps `enable_hour_cyclic` off by default policy; Path B can enable it as an optional context feature.
- Consequence: feature-profile comparisons stay aligned with the split-path scientific assumptions.

### Handcrafted physical hardening (phase 1)

- Observation: Costa uses both measured string-level electrical channels (`pdc1/pdc2`, `idc1/idc2`) and an ingestion-derived aggregate (`pdc = pdc1 + pdc2`), with strong redundancy.
- Interpretation: early destructive pre-drop on string channels would remove source variables needed to build meaningful compact descriptors.
- Decision: phase-1 physical hardening adds compact imbalance descriptors (`power_imbalance`, `current_imbalance`, `voltage_imbalance`) and keeps optional directional shares behind a dedicated flag.
- Consequence: feature engineering can reduce raw duplication pressure while preserving physically meaningful structure before final train-only pruning decisions.
- Implementation note: EDA redundancy findings are kept as advisory evidence rather than as automatic deletion rules, so source channels remain available until hygiene + mRMR operate on the fully built feature space.

### Costa derivative policy (current phase)

- Observation: generic derivative flags in legacy code (`dP_dt`, `dI_dt`) were wired to non-Costa column names (`Pg`, `Ig`).
- Interpretation: Costa derivatives must be mapped to Costa electrical channels to remain meaningful.
- Decision: `dP_dt` now maps to `pdc` (or reconstructed total from `pdc1+pdc2` fallback), `dI_dt` maps to total current (`idc1+idc2`), and `dV_dt` maps to mean DC voltage (`(vdc1+vdc2)/2`) for Costa-style schemas.
- Consequence: the unified `plus_physics` profile now keeps power/current/voltage dynamics active while preserving the compact imbalance descriptors in a single physics track.

### Temperature-aware power correction

- Observation: Costa raw data includes module temperature (`pvt`) and the CS6U-330P datasheet is available, including `gamma_Pmax = -0.40 %/C`.
- Interpretation: temperature-aware correction is physically justified and can separate expected thermal derating from abnormal electrical underperformance.
- Decision: phase-1 physics now includes temperature-aware features (`temp_loss_pmax`, `pdc_temp_corrected`, `pdc_temp_corrected_norm_irr`) behind `enable_temp_power_correction`.
- Consequence: the single `plus_physics` profile captures compact imbalance dynamics and first-order thermal derating effects without introducing a separate physics profile branch.

### Handcrafted statistics family (rolling stats)

- Observation: rolling statistics (mean, std, min, max) over causal windows [5, 10, 30] are already wired into the pipeline. The window captures local temporal context while preserving row alignment.
- Interpretation: adding higher-order moments (skew, kurtosis) and signal descriptors (RMS, ZCR) to the same rolling infrastructure is a zero-overhead extension — same segment-safe groupby mechanism, same output shape.
- Decision: rolling stats family is extended to 10 supported statistics: mean, std, min, max, median, range, skew, kurtosis, rms, zcr. Per-profile config (`rolling_stats` list) controls which subset is active. Profile `plus_rolling` enables all 8 non-redundant stats over windows [5, 10, 30].
- Consequence: rolling and explicit windowing are mutually exclusive by pipeline policy — when `enable_explicit_windowing` is true, rolling stats are auto-disabled and logged. This enforces a clean ablation axis between the two temporal encoding strategies.

### tsfresh disabled across all profiles

- Observation: tsfresh extraction adds significant runtime cost and the handcrafted statistics family now covers the same statistical moments with full interpretability and segment-safety guarantees.
- Interpretation: for Phase 1 (R1 classical ML baselines), handcrafted features are strictly preferable: faster, interpretable, no segment-leakage risk from tsfresh's internal windowing.
- Decision: `tsfresh_mode` is set to `"off"` in all profiles. The `plus_tsfresh_minimal` and `plus_tsfresh_extensive` profiles are removed. tsfresh remains in the codebase as an optional comparator for later ablation if needed.
- Consequence: the profile ladder is cleaner and all runs are faster by default.

### Explicit windowing architecture

- Observation: explicit windowing (collapsing N rows into 1 row per window) is fundamentally incompatible with row-aligned features — they cannot share the same DataFrame.
- Interpretation: windowing must be a separate post-processing step applied after all row-aligned features are generated, not an alternative feature generator.
- Decision: the pipeline applies windowing after `add_optional_features`, before pruning. The windowed DataFrame replaces the row-aligned one; pruning (corr, VIF) then operates on the windowed columns. Rolling stats are auto-disabled when windowing is active.
- Consequence: the windowed track is separated from the row-aligned track, with profile growth starting from `plus_windowed` and `plus_multiscale`, then extending into named spectral ablations later.

### Multi-scale windowing strategy

- Observation: different window sizes produce different numbers of output rows, making naive concatenation of scales impossible.
- Interpretation: the correct multi-scale approach uses the largest window as primary granularity and computes statistics at nested sub-scales (last `w` samples of the primary window) within each primary window. This gives a fixed output shape with features at multiple temporal resolutions.
- Decision: `add_multiscale_window_features` implements nested sub-windows: for each primary window, stats are extracted at scales [60, 30, 10] using slices of the primary window. Column naming is `{feat}_w{scale}_{stat}`.
- Consequence: one output row per primary window stride, features at 3 scales × 8 stats × N_features. Any residual multiscale collinearity is left for optional late ablations rather than default pruning.

### Spectral method selection for Path B (2026-04-23)

- Observation: at Costa's low sampling rate, the spectral branch must stay compact and physically grounded rather than accumulating overlapping representations.
- Interpretation: the cleanest split is fixed-basis multiresolution (`WPD`) versus adaptive decomposition (`CEEMDAN`), then a hybrid branch that tests whether they add complementary information.
- Decision: the named spectral profile ladder retains only `WPD` and `CEEMDAN`.
  - **WPD (Wavelet Packet Decomp):** strongest fixed-basis candidate — sampling-rate agnostic, handles transients, physically grounded. Level-3 db4 gives 8 subbands. Keep all terminal nodes at the chosen level and let downstream selection decide relevance.
  - **CEEMDAN:** adaptive candidate for non-linear/non-stationary modes. Keep only the first K=3 IMF energy ratios (zero-pad if fewer) as a compact adaptive summary. `EMD-signal` already in deps. Slow — gated behind its own flag and documented as such.
- Consequence: the spectral profile ladder is now:
  - `plus_wpd` — WPD only (Path B only)
  - `plus_ceemdan` — CEEMDAN only (Path B only, slow)
  - `plus_wpd_ceemdan` — hybrid fixed-basis + adaptive branch (Path B only)

### Spectral combination rationale

- Observation: WPD and CEEMDAN are the more genuinely complementary pair: fixed-basis multiresolution energy vs. adaptive mode energy.
- Interpretation: the cleanest thesis ablation is to test WPD alone, CEEMDAN alone, then their hybrid.
- Decision: named spectral profiles now isolate those three hypotheses directly. The selector stack remains unchanged: all WPD node energies and CEEMDAN IMF energy ratios enter hygiene -> mRMR by default, with optional corr/VIF only for explicit late checks.
- Consequence: the ablation story now reads as: windowed → multiscale → +WPD → +CEEMDAN → +WPD_CEEMDAN.
