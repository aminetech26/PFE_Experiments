# PC-Flow Design & Recent Architectural Decisions

> Decision log for the Task A (anomaly detection) deep-learning track, centred on
> **PC-Flow** (Physics-Conditioned Normalizing Flow) — the primary architectural
> contribution — plus the shared evaluation infrastructure and the cross-model
> alignment work completed in May 2026. Companion to `THESIS_LOG.md`,
> `MODELING_DECISIONS.md`, and `TECHNICAL_DESIGN.md`.
>
> Created 2026-05-31. Every architectural claim below is grounded in
> `src/modeling/anomaly_detection/dl/pc_flow/{model.py,trainer.py}` and
> `configs/model_config.yaml` as committed.

---

## 1. Scope

PV fault **detection** (Task A) is posed as **semi-supervised anomaly detection**:
the model is trained on normal-class data only and scores how anomalous each test
sample is. PC-Flow is the deep-learning entry in this track, benchmarked against
PC-AE, MAAT (Mamba Anomaly Transformer), PC-DLSSM, PV-GDN, and the classical
baselines (One-Class SVM, Isolation Forest, BOCD).

The thesis claim PC-Flow is meant to support: **a conditional density model scores
subtle, rare faults better than reconstruction- or attention-based detectors under
fair, per-class evaluation** — not "beats baseline X by Δ". The bar is *honest
per-class PR-AUC across all fault classes*.

---

## 2. PC-Flow architecture (the contribution)

PC-Flow is a **conditional normalizing flow**: a pure-PyTorch conditional RealNVP
(Dinh et al. 2017) with affine coupling layers, conditioned on the exogenous
operating point `c`. An optional rational-quadratic-spline coupling variant
(RQ-NSF, Durkan et al. 2019) is implemented as the more-expressive alternative.

### 2.1 Why a normalizing flow (vs autoencoder / attention)

- **Exact likelihood, no reconstruction bottleneck.** PC-AE and attention models
  score via reconstruction MSE, which averages error over feature dimensions
  (`MSE = (1/D)·Σ(x−x̂)²`) and a learned bottleneck that can reconstruct mild
  anomalies away. The flow scores the *exact* conditional density
  `−log p(x | c)` — no lossy bottleneck, no `1/D` averaging.
- **Physics conditioning is native.** The exogenous operating point
  `c = (irradiance, module-temperature)` is fed into every coupling MLP, so the
  flow models `p(x | c)` directly. A sample whose electrical state is unlikely
  *given its irradiance/temperature* is flagged — exactly the FDD question — with
  no separate "expected-power" regressor and no c-invariance regularizer.
- **Deployable.** Stateless (`win_size = 1`), ~7k–23k parameters, ONNX-exportable,
  sub-millisecond CPU inference — suitable for the edge target (Jetson).

### 2.2 Block structure (affine coupling)

Stack of `K` conditional affine-coupling blocks (default `K = 4`,
`hidden_dim = 32`, GELU MLPs). Each block:

1. Split `x` into `(x_a, x_b)` via a fixed binary mask (alternating even/odd
   pass-through across blocks).
2. `(s, t) = MLP([x_a, c])` — scale and translation conditioned on the
   pass-through half **and** the physics context.
3. `x_b' = x_b · exp(s) + t`, invertible, with `log|det J| = Σ s`.
4. A fixed random permutation between blocks so all dimensions mix over depth.

Stabilizers: `s = tanh(s)·2` bounds the log-scale to `(−2, 2)`; the final scale
head is zero-initialized so each block starts at the identity (stable early
training). Latent base distribution is a standard Gaussian.

### 2.3 Anomaly score

```
score(x | c) = −log p(x | c) = 0.5·‖z‖² + 0.5·D·log(2π) − Σ_k log|det J_k|
```
where `z = f(x; c)` is the latent. Higher = more anomalous. No temporal reduction
(stateless), no fusion — the raw conditional NLL is the official score.

### 2.4 Spline variant (negative / expressiveness ablation)

`coupling_type = "spline"` replaces the affine transform with a per-dimension
monotonic rational-quadratic spline conditioned on `[x_a, c]` (Durkan et al.
2019), with linear tails outside `±tail_bound` and `n_bins` knots. Kept as an
ablation arm; affine is the canonical contribution.

### 2.5 ONNX / edge-export discipline (decision)

The model is written so the traced graph is **batch-generalizable**, not baked to
batch-1:
- Coupling uses `index_select` (→ ONNX `Gather`) + slice + concat, **never**
  boolean-mask indexing (which bakes batch-1 shape constants).
- Spline bin lookup uses a comparison-sum (`Greater` + `ReduceSum`) instead of
  `torch.searchsorted`, exact for the left-closed bin convention and exportable.
- Spline parameter reshape keeps the batch dim dynamic.

**Consequence:** the same checkpoint exports cleanly to ONNX for Jetson without a
re-implementation, satisfying the deployment requirement.

---

## 3. Scoring, training, and selection decisions

| Aspect | Decision | Rationale |
|---|---|---|
| Training objective | mean NLL on **normal data only** | semi-supervised AD; no fault labels in train |
| Conditioning `c` | dataset-resolved context features (Costa `[irr, pvt]`) | physics operating point; `--unconditional` ablation drops it |
| Headline / selection metric | **`val_macro_per_class_pr_auc`** | honest per-class detection under one fair threshold; see §4.1 |
| Early stopping & checkpoint | monitored on the same macro per-class PR-AUC | selection signal == reporting signal |
| Scaler | `StandardScaler` fit on **train only** | leakage-safe |
| Threshold | shared **GPD / Peaks-Over-Threshold** calibration on train-normal NLL | uniform across all detectors (§4.2) |
| HPO | single-stage Optuna, TPE + median pruner, objective = best val macro per-class PR-AUC | validation-only selection (§4.3) |
| Operating points | GPD-baseline, sensitive, sensitive+hysteresis, conformal, FDR-BH, CUSUM | reported uniformly via `flatten_operating_points` |
| Leakage audit | label-shuffle + performance-sanity after every run | shuffle PR-AUC must collapse to base rate |
| Ablation switches | `--unconditional`, `--score-mode typicality`, `--coupling-type {affine,spline}` | conditioning / score / expressiveness arms |

---

## 4. Evaluation methodology (shared across all Task A models)

### 4.1 Macro per-class PR-AUC is the headline (not ROC-AUC, not F1 tricks)

- **Observation.** The normal class is ~97% of rows; ROC-AUC is misleading under
  that imbalance, and a single binary F1 hides which fault classes are missed.
- **Decision.** Headline = **macro per-class PR-AUC** — for each fault class,
  PR-AUC of that-class-vs-normal, averaged across classes; report the
  **worst-class** PR-AUC alongside it. Threshold-dependent F1 is secondary.
- **Consequence.** A detector that trivially fires (`recall=1, precision=base
  rate`) is scored as the failure it is, not a win. Per-class numbers expose the
  subtle classes (e.g. Costa "degradation") that headline binary metrics hide.

### 4.2 Uniform GPD threshold calibration

- **Decision.** Every detector's operating threshold comes from the same
  Peaks-Over-Threshold + Generalized Pareto fit on its **train-normal** scores
  (`q0.90` exceedances), not a per-model F1 sweep.
- **Consequence.** Cross-model comparisons are apples-to-apples; the threshold
  policy is a property of the evaluation protocol, not a per-model tuning knob.

### 4.3 Validation-only model selection

- **Decision.** HPO winners and checkpoints are chosen by **validation** macro
  per-class PR-AUC only — never by test performance and never by a val–test gap
  heuristic. Simplicity is the tie-breaker among near-equal configs.
- **Consequence.** No selection leakage; reported test numbers are an honest
  estimate of generalization.

### 4.4 Episode- vs sample-level reporting

- Sample-level per-class PR-AUC is the strict, high-support metric. Episode-level
  (per fault event, p95-aggregated) is reported because operationally one cares
  about catching the *event*, not every sample. Per-class episode numbers are
  always reported **with their episode count** — sparse classes (2–6 episodes)
  carry wide error bars and must not be over-read.

### 4.5 Variance estimation (Costa)

- 5-fold **episode-stratified** cross-validation (`kfold_episode_split.py`):
  train held fixed (normal-only), val/test rotate at the **episode** level so an
  episode never appears in both val and test of a fold. Reports mean ± std per
  metric across folds.

---

## 5. Recent cross-model alignment decisions (May 2026)

### 5.1 MAAT selection metric aligned to PR-AUC

- **Observation.** MAAT selected checkpoints / HPO by `macro_fault_f1` (a
  threshold-dependent F1 sweep) while PC-Flow / PC-AE / DLSSM selected by macro
  per-class PR-AUC. Cross-model comparison was therefore not apples-to-apples.
- **Decision.** Switch MAAT's selection metric to **`macro_per_class_pr_auc`**
  (threshold-free); MAAT still computes its F1-sweep internally for diagnostic
  thresholds, but selection, early stopping, checkpointing, and the HPO objective
  now use macro per-class PR-AUC. Threshold policy reporting switched from the
  hard-coded `validation_macro_fault_f1` to the actual shared GPD policy string.
- **Consequence.** All Task A detectors are now selected and thresholded by the
  same protocol; MAAT numbers are directly comparable to PC-Flow.

### 5.2 MAAT HPO observability and speed

- **Observation.** A 10-trial MAAT HPO ran > 3 h with no per-epoch logging — a
  silent run with no way to spot a hang or a degenerate trial.
- **Decision (two parts).** (a) Add an `_EpochLogger` callback emitting a
  one-line per-epoch summary (selection score, F1, recon loss, NaN count,
  elapsed) for both HPO trials and the final run; "trial N starting" / "done /
  pruned" markers pinpoint a mid-trial hang. (b) Cap the macro-fault-F1 threshold
  calibration at 400 evenly-spaced PR-curve candidates (`_MAX_THRESHOLD_
  CANDIDATES`) — the per-epoch sweep over tens of thousands of unique scores at
  `stride=1` was the dominant cost.
- **Consequence.** HPO is observable and the per-epoch threshold cost drops ~99%
  with <0.5% threshold-quality loss.

### 5.3 Poisoned-sanity-check fix (selection integrity)

- **Observation.** PyTorch-Lightning runs a 2-batch sanity validation before
  epoch 0. With a large batch only one fault class fell in those batches, so the
  partial-class macro PR-AUC was logged as a spuriously high value and, because
  `best_val_*` is a running max, that ceiling poisoned the HPO objective for the
  whole trial.
- **Decision.** Guard `on_validation_epoch_end` with `trainer.sanity_checking` —
  clear buffers and return without logging or updating `best_val_*` during sanity.
- **Consequence.** `best_val_*` only reflects real epochs; HPO objectives are no
  longer ceilinged by a 2-batch artifact.

### 5.4 Evaluable-class filtering

- **Decision.** Fault classes a dataset declares as evaluable
  (`splits.evaluable_classes`) define the val/test scoring set; classes with too
  few episodes for meaningful per-class evaluation (`train_only_classes`) are
  dropped from val/test (normal rows always kept). No-op for datasets that list
  all classes (e.g. Costa `[1,2,3,4]`).
- **Consequence.** The macro per-class PR-AUC driving selection isn't polluted by
  classes that can't be evaluated; comparisons use a stable class set.

---

## 6. Generalization study — La Réunion (investigation, lessons, and why it was dropped)

A second real dataset (University of La Réunion, ~7 s sampling, single inverter)
was trialled to test whether PC-Flow's strong Costa per-class result transfers.
**The dataset was abandoned as a Task A generalization testbed**, but the
investigation produced a result worth recording in the thesis.

### 6.1 The structural mismatch

- La Réunion faults are **shading-only** (`3.1` = 1/3-module, `3.2` = 2/3-module,
  `4.0` — constant partial-shading variants), not Costa's diverse *sharp* faults
  (short-circuit, open-circuit, degradation). Different fault physics ⇒ different
  discriminative features.
- It is a **single inverter**, so Costa's decisive **string-imbalance** channels
  (`pdc1/pdc2`, `idc1/idc2`) — which carried the subtle degradation class — do not
  exist.

### 6.2 Core architectural finding (transferable to the thesis)

Diagnosis was done locally with leakage-clean univariate and Mahalanobis probes:

- Of 17 engineered features, **14 are pure noise for shading** (≤ 0.02 per-class
  PR-AUC) — physically expected: shading is a power deficit and does not perturb
  Vg / Ig / dP_dt / temperature.
- **Any full-joint-density scorer buries a signal localized in 1–2 dimensions.**
  A Mahalanobis distance (a clean Gaussian NLL with *no* heavy tail) on the 17
  features scored the subtle class at **0.10 — identical to PC-Flow** — while the
  single discriminative channel scored **0.55** univariate. The aggregate
  `‖z‖² = Σ z_i²` is dominated by the χ² background of the non-informative
  dimensions, which exceeds a sub-per-sample-noise fault. Being an *exact*-
  likelihood flow does not exempt it; a Gaussian reproduces the same burying.
- **Therefore the lever is feature curation, not score-side tricks.** Curating to
  the discriminative channel lifted the joint scorer's subtle-class PR-AUC from
  0.156 → 0.548 with no change to the model.

### 6.3 Why it was still dropped

- The discriminative channel for shading is a power-deficit signal: either the
  differential `ΔP = Pg − α·Pg_ref` against a **healthy twin inverter** (dt3) —
  **impractical**, needs a known-healthy reference — or the dt2-only
  **expected-power residual** `Pg − E[Pg | GTI, TPV]` (deployable but weaker).
- Even with the best channel, the subtlest fault (1/3-module shading, ~2–3%
  deficit at 7 s sampling) tops out at univariate PR-AUC ≈ 0.34–0.55. Costa's
  ~0.95 per-class is **not physically reachable** for it; forcing it would require
  threshold/selection trickery that reads as leakage, not strength.
- Net: shading-only, heterogeneous-severity faults on a single inverter are a weak
  generalization test for a model whose strength is diverse sharp faults. Decision
  recorded; the replacement-dataset criteria are: diverse sharp faults,
  multi-string (so imbalance features exist), irradiance + temperature sensors.

---

## 7. Headline Costa result (reference)

PC-Flow, Costa Path A, `plus_physics` profile (17 features → 15 modeled + 2
context `[irr, pvt]`), 5-fold episode-stratified CV, ~23k parameters:

| Metric | Mean ± std |
|---|---|
| test macro per-class PR-AUC | **0.990 ± 0.007** |
| └ class 1 (short-circuit) | 1.000 |
| └ class 2 (**degradation, subtle**) | **0.973 ± 0.023** |
| └ class 3 (open-circuit) | 1.000 |
| └ class 4 (shadowing) | 0.985 ± 0.012 |
| test worst-class PR-AUC | 0.968 ± 0.018 |
| test binary PR-AUC | 0.989 ± 0.008 |
| test episode macro per-class PR-AUC | 0.964 ± 0.038 |
| leakage report | CLEAN (all folds) |

The class-2 (subtle degradation) result at 0.97 is the evidence for the
"captures subtle faults" claim — and §6.2 explains *why* it works on Costa (the
signal is above per-sample noise and lives in the imbalance channels) and why a
sub-noise shading fault is a different, harder regime.

---

## 8. Dated decision log

### 2026-05-31
- **La Réunion abandoned as Task A generalization testbed** (§6). Investigation
  artefacts discarded from the working tree; general infrastructure improvements
  retained. Replacement-dataset criteria recorded.

### 2026-05-30
- **MAAT selection metric → macro per-class PR-AUC** (§5.1); threshold-policy
  string made dynamic.
- **MAAT HPO observability + threshold-sweep speedup** (§5.2).
- **Poisoned-sanity-check guard** added to PC-Flow validation (§5.3).
- **Evaluable-class filtering** wired into anomaly selection (§5.4).
- **`--feature-allowlist`** capability prototyped (model only the discriminative
  subset) — motivated by §6.2; reverted with the La Réunion WIP but the finding
  stands.

### Earlier (commit trail)
- `switched to spline transformation` — RQ-spline coupling variant added (§2.4).
- `add pcflow` / `physics conditioning` — PC-Flow introduced with conditional
  affine coupling on `c = (irr, pvt)` (§2).
- `uniform operating point` / `shared threshold calibration + cv variance` —
  shared GPD threshold + operating-point system + 5-fold episode CV (§4.2, §4.5).
- `add AD specific leakage checks` — label-shuffle + sanity audit (§3).
- `episode level metrics at operating point` — episode-level per-class reporting
  (§4.4).

---

## 9. Open items / honest limitations

- PC-Flow's NLL has a heavy upper tail (observed `score_max / p95 ≈ 10³–10⁴`); it
  does not affect rank-based PR-AUC but is worth a note for any threshold whose
  precision depends on the extreme tail. `--score-mode typicality` is the
  implemented two-sided alternative.
- Joint-density scoring buries low-dimensional fault signals (§6.2); for datasets
  where the fault lives in a small physics subspace, feature curation (or a
  curated, conditioned low-dim flow) is required — a general property of
  density-based AD, not specific to PC-Flow.
- A clean, criteria-matched second dataset is still needed to complete the
  generalization claim (§6.3).
