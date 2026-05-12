# Task A DL Roadmap — Three Tracks

## Context

Task A ML baselines complete: OC-SVM ~92% PR-AUC (RBF, plus_physics), XGBoost ~98% (supervised binary). Goal: beat with three deep learning tracks. Binary anomaly (normal vs fault) enables cross-dataset evaluation across Costa, La Réunion, Mendeley.

**Scope constraints:**
- Task A only — Task B (classification) is considered resolved
- No focal loss — Costa supervised split is ~53% normal / ~47% fault, not a severe enough imbalance to justify focal loss complexity
- Physics loss (Track 2) requires a dedicated design session before implementation — open problems documented below

---

## Three-Track Design

### Track 1 — Pure SSM (Feature-Rich)

- **Purpose**: Establish a strong SSM baseline using the richest available feature representations, without any custom loss augmentation
- **Profiles**: `plus_rolling` (Path A), `plus_all` (Path B)
- **Challenge**: High dimensionality (plus_all can be 100+ features) — dimensionality reduction may be needed (PCA, learned linear projection, or input embedding layer before SSM)
- **Model**: SSM variants from papers — architecture and variant selection driven by user's paper references
- **Loss**: Standard reconstruction MSE
- **Anomaly score**: Reconstruction MSE(x̂, x) at test time
- **Note**: No physics loss, no SSL — clean ablation that isolates the SSM architecture's contribution

### Track 2 — SSM + Custom Physics-Informed Loss

- **Purpose**: Augment Track 1's SSM with domain-specific physics constraints on the reconstruction output
- **Profile**: Same as Track 1 (`plus_rolling` / `plus_all`)
- **Model**: Same SSM family as Track 1, with reconstruction head outputting x̂ in feature space
- **Loss**: `L_recon(x̂, x) + λ_physics × L_physics(x̂)`
- **Physics loss**: Applied to x̂ (reconstruction output) — NOT to z (latent). x̂ lives in feature space where each dimension is a named physical channel. See design section below.
- **Anomaly score**: Reconstruction MSE, optionally weighted by physics violation magnitude
- **Contribution**: The physics loss is a major thesis contribution — careful derivation and domain argumentation required

### Track 3 — SSL (Distance-Based Anomaly)

- **Purpose**: Learn a rich temporal representation self-supervisedly from normal-only data, then score anomalies as distance from the normal manifold in latent space
- **SSL methods**: JEPA, BYOL, TS2Vec (one or more, compared as sub-variants)
- **No reconstruction head**: Forcing reconstruction risks pulling representations toward identity mapping, conflicting with SSL's goal of learning abstract invariances and temporal structure
- **Anomaly score**: Distance in latent space from normal centroid (Mahalanobis distance, cosine distance, or NLL under fitted Gaussian on training embeddings)
- **No physics loss**: z has no guaranteed correspondence to physical channels — applying physics equations to abstract latent dimensions is architecturally unjustifiable

---

## Physics-Informed Loss — Design TBD ⚠️ (Track 2 Only)

The physics loss is a major contribution. It must be physically grounded AND anomaly-discriminative — a constraint satisfied by both normal and fault states is a regularizer, not an anomaly signal.

### Architectural anchor (fixed)

Physics loss applies to **x̂** (reconstruction in feature space), **not to z** (latent).

### Candidate constraints (Costa-specific, starting point)

**1. String power additivity** (ingestion invariant `pdc = pdc1 + pdc2`):
```
L_balance = || x̂[pdc] - x̂[pdc1] - x̂[pdc2] ||²
```

**2. Temperature-corrected power plausibility** (Sandia model, CS6U-330P datasheet):
```
P_expected = η_ref × A × x̂[irr] × (1 + γ × (x̂[pvt] - T_ref))
L_plausibility = || x̂[pdc] - clip(P_expected, 0, P_rated) ||²
γ = -0.004 /°C,  T_ref = 25°C
```

**3. Per-string Ohm's law**:
```
L_ohm = || x̂[pdc1] - x̂[vdc1] × x̂[idc1] ||²
       + || x̂[pdc2] - x̂[vdc2] × x̂[idc2] ||²
```

### Open design problems (must resolve before Track 2 implementation)

1. **Reconstruction quality dependency**: `L_plausibility` uses reconstructed irr and pvt — poor reconstruction of these channels can penalize physically correct pdc reconstructions.
2. **Nonlinear gradient interactions**: `L_ohm` involves products of reconstructed channels — cross-channel gradient coupling can destabilize training.
3. **Anomaly-discriminative power**: Some constraints may hold for both normal and fault states — they would contribute no anomaly signal. Must verify per constraint.
4. **Normalization / scale mismatch**: Physics constraints operate on raw units (W, V, A); model sees normalized features — explicit denormalization required.
5. **Thesis justification**: Each term must be argued as anomaly-discriminative, not just physically correct.

---


## Dimensionality Reduction (Track 1 / Track 2 on plus_all)

`plus_all` can produce 100–200+ features. Options before SSM input:

- **Learned input projection**: Linear(F → d_model) — simplest, trained end-to-end, adds no separate step
- **PCA**: Fit on train-only, apply to val/test — preserves variance, no reconstruction needed, interpretable variance ratio
- **Autoencoder bottleneck pre-projection**: Heavier, adds training complexity

**Default recommendation**: Learned linear projection (d_model = 32 or 64) as the first SSM layer — no separate DR step, no leakage risk, consistent with end-to-end training.

---

## Sequence Construction Note

Windows provide **short-term temporal context around the operating point** — not a representation of fault development. Since Costa faults are artificially induced (instantaneous induction, not gradual onset), the window captures local operating regime context for regime comparison and smoothing, not fault evolution. Window size is a hyperparameter for context width.

---

## Files to Create / Expand

### New shared infrastructure
| File | Purpose |
|---|---|
| `src/modeling/anomaly_detection/dl/dataset.py` | `TimeSeriesDataset` — sliding window (episode-safe), normal-only mode for semisup |
| `src/modeling/anomaly_detection/dl/losses.py` | `PhysicsInformedLoss(x_hat, feature_index_map)` (TBD terms) |
| `src/modeling/anomaly_detection/dl/base_trainer.py` | `BaseAnomalyLightningModule` — val/test PR-AUC, threshold calibration, MLflow hooks |
| `src/modeling/anomaly_detection/dl/run.py` | Dispatcher — reads `anomaly_detection.dl.active_model`, routes to track1/2/3 |
| `src/modeling/anomaly_detection/dl/ssl_encoder.py` | `SSLEncoder(nn.Module)` — backbone + SSL pre-training variants (JEPA / BYOL / TS2Vec) |

### Expand stubs
| File | Change |
|---|---|
| `src/modeling/anomaly_detection/dl/ssm_model.py` | Track 1: pure SSM; Track 2: SSM + reconstruction head + physics loss on x̂ |
| `src/modeling/anomaly_detection/dl/reconstruction_autoencoder_model.py` | Repurpose for Track 3 SSL entry point: `run_ssl_anomaly()` — pre-train encoder, fit normal centroid, score by latent distance |

---

## Evaluation Protocol

1. Train on `costa/anomaly_semisup`, profile `plus_rolling` (Tracks 1/2), SSL encoder on normal pool (Track 3)
2. Threshold calibration on val: max-F1 on PR curve
3. Test: PR-AUC, F1, precision, recall
4. Ablations:
   - Track 1: SSM variants, plus_rolling vs plus_all, with/without DR
   - Track 2: ±physics loss terms, Path A vs Path B
   - Track 3: JEPA vs BYOL vs TS2Vec, distance metric (Mahalanobis vs cosine vs NLL)
5. Wilcoxon signed-rank test vs best ML (OC-SVM), p < 0.05
6. Cross-dataset: La Réunion binary, Mendeley binary

---

## Build Order

| Week | Deliverable |
|---|---|
| 1 | `dataset.py`, `base_trainer.py`, `run.py`, config additions |
| 2 | Track 1 — pure SSM on plus_rolling; confirm beats OC-SVM baseline |
| 3 | Track 3 — SSL pre-training (BYOL first, simplest); latent distance scoring |
| 4 | Physics loss design session → `losses.py` → Track 2 SSM + physics |
| 5 | Full ablation matrix on Costa (all three tracks) |
| 6 | Cross-dataset, Wilcoxon tests, thesis writing |

---

## Verification Checklist

- [ ] Track 1: training loop reachable, val PR-AUC logged to `Task_A_Anomaly`
- [ ] Track 2: physics loss terms logged separately per epoch
- [ ] Track 3: SSL pre-training loss decreasing; centroid fit on train embeddings
- [ ] `experiments/metrics/anomaly_dl_results.json` written per track
- [ ] Cross-dataset: `--dataset la_reunion` and `--dataset mendeley` produce scores without error
