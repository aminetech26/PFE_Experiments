# Technical Design Document - PV Fault Detection and Diagnosis System
**Author:** Ahmed Amine GUERRAICHE  
**Revision:** 2.0 - April 2026  
**Status:** Working Draft with scope update

This document contains staff-engineer-level reasoning for the system architecture. It is not a summary of goals; it is a justification of *why* each choice was made over alternatives, grounded in dataset characteristics, hardware constraints, and thesis requirements. Read this before making architectural changes.

**Scope update (April 2026):** the canonical thesis scope is now **two primary tasks** - fault detection and fault classification - with **Costa as the primary benchmark dataset**. Forecasting is no longer a standalone thesis task. It is retained only as an optional residual-based anomaly analysis method. Edge deployment and GUI remain mandatory deliverables.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Dataset-Specific Signal Processing](#2-dataset-specific-signal-processing)
3. [Feature Engineering Decisions](#3-feature-engineering-decisions)
4. [Model Architecture Decisions](#4-model-architecture-decisions)
5. [Multi-Task Learning Design](#5-multi-task-learning-design)
6. [Hyperparameter Optimization Protocol](#6-hyperparameter-optimization-protocol)
7. [Evaluation Protocol and Anti-Leakage Rules](#7-evaluation-protocol-and-anti-leakage-rules)
8. [Edge Deployment Architecture](#8-edge-deployment-architecture)
9. [Domain Adaptation — The Research Gap](#9-domain-adaptation--the-research-gap)
10. [Eliminated Techniques — With Reasons](#10-eliminated-techniques--with-reasons)
11. [Predicted Experimental Outcomes](#11-predicted-experimental-outcomes)

---

## 1. Problem Formulation

### 1.1 Why Two Primary Tasks, Not Three

The original project considered anomaly detection, fault classification, and forecasting as equal tasks. After reviewing the public dataset landscape and the available evidence, that framing is no longer the most defensible research position.

The thesis now centers on **two primary tasks**:

1. detect whether the PV system is operating abnormally,
2. classify the fault type once abnormal behavior is detected.

This still matches the essential operational need: an operator must know **that** something is wrong and **what** is wrong. Forecasting remains useful only as an optional residual-based method, not as a third thesis pillar.

### 1.2 Why Costa Is the Vertical Anchor

Costa is the primary benchmark because it provides the best current combination of:

- real-world fault data,
- meaningful data volume,
- multiple fault classes,
- and published benchmark numbers that can be challenged directly.

This makes Costa the correct dataset for the first phase of work. The thesis should become strong vertically on Costa before claiming broader generalization across plants, climates, or domains.

### 1.3 Task A - Fault Detection

**Task A - Binary anomaly / fault detection:**
$$
f_A : \mathbb{R}^{T \times D} \rightarrow [0, 1]
$$
Output: anomaly score or binary decision. On highly imbalanced datasets this is best treated as an anomaly-detection problem rather than a naive balanced classifier. On Costa, supervised and semi-supervised variants are both worth benchmarking; on La Reunion, one-class formulations remain especially relevant.

**Critical metric choice — PR-AUC, not ROC-AUC:**

With a 97/3 class split, a model that predicts "normal" for every sample achieves 97% accuracy and ROC-AUC ≈ 0.5. These metrics are useless. Precision-Recall AUC directly measures the tradeoff that matters: of all the times I raise an alarm, how often am I right, and of all the actual faults, how many do I catch?

$$
\text{PR-AUC} = \int_0^1 P(r) \, dr
$$

A random baseline achieves PR-AUC equal to class prevalence. Any anomaly model worth reporting must significantly beat that baseline.

### 1.4 Task B - Fault Classification

**Task B - Multi-class fault classification:**
$$
f_B : \mathbb{R}^{T \times D} \rightarrow \Delta^8
$$
Output: probability distribution over fault types. The vertical benchmark is now Costa. Mendeley remains valuable for simulated-source experiments, but no result trained there should be assumed to transfer to real deployments without adaptation.

### 1.5 Forecasting Is Optional Residual Analysis, Not a Core Task

True fault-onset prediction is not scientifically defensible on the currently available public PV fault datasets because the faults are artificially induced and lack reliable pre-fault signatures.

Forecasting is therefore retained only in the following narrow role:

- learn expected normal behavior,
- compute prediction residuals,
- use residual magnitude as a complementary anomaly signal.

This can be useful experimentally, but it is not a primary thesis objective and must not dominate the architecture or evaluation narrative.

### 1.6 Research Sequence

The correct sequence of work is:

1. establish a strong Costa benchmark,
2. choose a deployable winner,
3. complete edge deployment and GUI,
4. then expand to La Reunion, Mendeley, and domain adaptation.

---

## 2. Dataset-Specific Signal Processing

### 2.1 Why Different Processing Per Dataset

The three datasets have fundamentally different sampling rates, noise characteristics, and physical origins. Applying the same feature engineering pipeline to all three is the most common mistake in multi-dataset PV papers. Here is the analysis:

| Property | Mendeley | La Réunion dt2 | Sonalgaz |
|---|---|---|---|
| Sampling rate | 10 kHz | ~7 s | 10 min |
| Physical origin | MATLAB/Simulink simulation | Real inverter output | Real grid monitoring |
| Noise type | None (simulated) | Sensor noise + irradiance variation | Measurement error + missing data |
| High-frequency content | Yes (arc faults, switching transients) | No (7s averaging removes everything) | No |
| Temporal resolution | Sub-millisecond | Sub-minute | 10-minute |

**Consequence:** The same sliding window size means completely different physical durations:
- 60 samples × 0.1 ms = **6 ms window** at Mendeley (appropriate for arc fault transient capture)
- 60 samples × 7 s = **7 minutes window** at La Réunion (appropriate for operational cycle)
- 60 samples × 10 min = **10 hours window** at Sonalgaz (inappropriate — spans multiple operational cycles)

For Sonalgaz, use a maximum window of 6 samples (1 hour) with stride 3 (30 min).

### 2.2 CEEMDAN: Only on Mendeley

Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN) decomposes a signal $x(t)$ into Intrinsic Mode Functions (IMFs):

$$
x(t) = \sum_{k=1}^{K} \text{IMF}_k(t) + r(t)
$$

where each IMF satisfies: (1) the number of extrema and zero-crossings differ by at most 1, and (2) the mean of the upper and lower envelopes is zero.

**Why it is useful for Mendeley:** DC arc faults (F7) and partial shading (F3) produce specific IMF patterns in the 100 Hz–10 kHz range that are not visible in raw time-domain data. CEEMDAN separates these components cleanly.

**Why it is useless for La Réunion:** At 7-second sampling, the Nyquist frequency is 0.07 Hz. There is no high-frequency content. Running CEEMDAN on 7-second electrical data will produce numerically unstable IMFs driven entirely by sampling noise rather than physical processes. Do not run CEEMDAN on La Réunion or Sonalgaz.

**Implementation cost:** CEEMDAN is $O(N \cdot K \cdot E)$ where $K$ is the number of IMFs and $E$ is ensemble size. For a 2.16M row dataset at 10 kHz, this is computationally expensive. Run it once, cache the IMF energy features to Parquet, and never recompute. The `ceemdan_features()` function in `features.py` does this.

### 2.3 Wavelet Denoising: Only on La Réunion

Discrete Wavelet Transform (DWT) with Daubechies-4 (db4):
$$
W_j[n] = \sum_k x[k] \cdot \psi_{j,k}[n]
$$

**Why db4, not db2 or haar:** The Daubechies-$p$ family has $p$ vanishing moments — it is orthogonal to polynomials of degree $p-1$. PV output signals under smooth irradiance variation are well-approximated by cubic polynomials over 5–15 minute windows. db4 (4 vanishing moments) will zero-out this smooth trend in detail coefficients, leaving only the fault-related deviations. db2 would mix trend and fault signal; db6 adds unnecessary computation for no gain.

**Why not on Mendeley:** The Mendeley data is synthetic with no measurement noise. Wavelet denoising on clean synthetic data can remove low-amplitude fault signatures (especially for F2 Partial Shading at light conditions). Do not apply it.

### 2.4 The La Réunion Differential Signal

Dataset dt2 (inverter 1, with faults) and dt3 (inverter 2, healthy) are from the same physical plant with identical orientation and irradiance exposure. This creates a natural control experiment that most papers ignore.

Define the differential power signal:
$$
\Delta P[t] = P_{\text{inv1}}[t] - \alpha \cdot P_{\text{inv2}}[t]
$$

where $\alpha = \text{median}(P_{\text{inv1}} / P_{\text{inv2}})$ over normal operating periods, estimated on the training set only.

**Why this is valuable:** $\Delta P$ removes common-mode disturbances — irradiance drops, temperature transients, morning/evening ramps. These are the dominant source of false positives in PV anomaly detection. A drop in $P_{\text{inv1}}$ due to passing cloud is common-mode and disappears in $\Delta P$. A drop due to a fault is differential and appears amplified.

**Critical implementation note:** $\alpha$ must be estimated on the training split only, then applied as a constant to validation and test splits. If you estimate $\alpha$ on the full dataset and then split, that is label leakage.

**Expected impact on Task A:** PR-AUC improvement of 10–20 percentage points over using raw $P_{\text{inv1}}$ alone.

---

## 3. Feature Engineering Decisions

### 3.1 Physics-Informed Features

The Performance Ratio (PR) is the most dataset-agnostic feature in this project:
$$
\text{PR}[t] = \frac{P_{\text{AC}}[t]}{P_{\text{STC}} \cdot \frac{G[t]}{G_{\text{STC}}}}
$$

where $G_{\text{STC}} = 1000 \, \text{W/m}^2$ and $P_{\text{STC}}$ is rated panel power.

**Why include it:** PR normalizes electrical output by irradiance, removing the dominant confounding variable. Fault-free degradation decreases PR slowly; acute faults cause step changes. Most DL models can learn this relationship implicitly, but providing it explicitly (a) accelerates training convergence, (b) improves sample efficiency (fewer data points needed), and (c) makes the learned representation interpretable.

**The temperature coefficient feature:**
$$
\Delta T_{\text{corrected}}[t] = P[t] \cdot (1 + \gamma \cdot (T_{\text{module}}[t] - 25))
$$

where $\gamma \approx -0.004 \, \%/°C$ for monocrystalline silicon. This corrects for the expected power reduction at elevated temperatures, isolating electrical faults from thermal behavior.

**Rate-of-change features:**
$$
\frac{dP}{dt}[t] = \frac{P[t] - P[t-1]}{\Delta t}
$$

Arc faults (F7 in Mendeley) and line-to-line faults cause rapid current/voltage transients. These show up in first derivatives more clearly than in raw values, especially when the baseline power level is high.

### 3.2 Window Size Selection

**For Mendeley (10 kHz):** A 60-sample window = 6 ms. This is sufficient to capture one full oscillation of 50 Hz AC interference and several switching transients. Arc faults have duration 50–500 ms — multiply windows by step to ensure coverage.

**For La Réunion (~7 s):** A 60-sample window = 7 minutes. This captures typical irradiance ramp-up times and short-duration fault events.

**Statistical justification:** The minimum window size for stationarity assumption in spectral features (FFT, wavelet) requires at least 2× the period of the lowest-frequency component of interest. At 10 kHz, if you care about 100 Hz components, minimum window is 20 samples. 60 samples provides ample margin.

---

## 4. Model Architecture Decisions

### 4.1 TCN as Shared Encoder

A Temporal Convolutional Network uses dilated causal convolutions:
$$
y[t] = (x * f_d)[t] = \sum_{k=0}^{K-1} f[k] \cdot x[t - d \cdot k]
$$

where $d = 2^i$ for layer $i$, exponentially expanding the receptive field.

**Receptive field calculation:** For $L$ layers with kernel size $k$ and max dilation $d_{\max} = 2^{L-1}$:
$$
\text{RF} = 1 + (k - 1) \cdot \sum_{i=0}^{L-1} d_i = 1 + (k-1)(2^L - 1)
$$

For $k=3$, $L=8$: RF = 1 + 2 × 255 = **511 timesteps**. This covers a 60-sample window 8× with context.

**Why TCN over LSTM for the shared encoder:**
1. **Parallelizability:** TCN processes all timesteps in parallel. LSTM is inherently sequential. At 128 hidden units, 60 timesteps, the difference is 12× faster training per epoch on a CPU, 4–8× on GPU.
2. **Fixed computation graph:** TensorRT optimization requires a static computation graph. LSTM hidden state creates dynamic branching that complicates TRT engine building. TCN is fully convolutional — TRT can fuse layers, apply INT8 quantization, and maintain strict latency guarantees.
3. **No vanishing gradient:** TCN with residual connections has gradient paths of $O(1)$ depth regardless of sequence length. With batch size 64 and 60 timesteps, LSTM training is observably unstable without careful gradient clipping.
4. **Interpretability:** TCN layer activations correspond directly to temporal patterns at specific scales (stride $d = 2^i$). LSTM hidden states have no direct physical interpretation.

**Why GRU (not LSTM) for Task C forecasting head:**

GRU equations:
$$
z_t = \sigma(W_z x_t + U_z h_{t-1})
$$
$$
r_t = \sigma(W_r x_t + U_r h_{t-1})
$$
$$
\tilde{h}_t = \tanh(W_h x_t + U_h(r_t \odot h_{t-1}))
$$
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

LSTM adds a separate cell state $c_t$ with an additional forget gate, input gate, and output gate — 4 gates total vs GRU's 2 gates. For $H = 128$ hidden units and $D = 20$ input features:
- LSTM parameters: $4 \times (H \times D + H \times H + H) = 97{,}792$
- GRU parameters: $3 \times (H \times D + H \times H + H) = 73{,}344$

This is a 25% parameter reduction. For sequences ≤ 1000 timesteps (which all three datasets satisfy), empirical evidence (Chung et al. 2014) shows GRU and LSTM achieve comparable accuracy. The GRU advantage is not just compute — it regularizes better on small datasets like La Réunion's fault subset.

**Why not Transformer:**

Self-attention complexity is $O(T^2 D)$ where $T$ is sequence length and $D$ is embedding dimension. For $T = 300$ (La Réunion window at high resolution), $D = 128$: $\approx$ 11.5M operations per forward pass per sample. Compare TCN at $O(T \cdot K \cdot D^2)$ per layer with $K = 3$, $D = 128$, $L = 8$: $\approx$ 4.7M total. Transformer is 2.4× more compute for no accuracy benefit on sequences this short. More critically, Transformer is not causal without explicit masking — a subtle bug that has invalidated published results in several papers. TCN is causal by construction.

### 4.2 LSTM Autoencoder for Task A

For anomaly detection, the architecture is:
```
Encoder: [TCN layers × 4] → h (bottleneck representation)
Decoder: [Transposed TCN / GRU] → x_hat
Loss: MSE(x, x_hat) — trained on NORMAL samples only
Anomaly score: ||x - x_hat||_2 (per-window)
Threshold: 95th percentile of reconstruction errors on validation normal samples
```

**Why reconstruction-based vs softmax classification:**
A softmax classifier for Task A requires fault samples during training. If your fault class is 2.83% of data, the model will see at most ~84K fault windows (from 2.99M La Réunion rows, ~3% × 60-sample windows). This is a severely imbalanced supervised problem. The autoencoder avoids this entirely by defining "normal" as regions of low reconstruction error — no fault labels needed. This also means Task A generalizes to *novel fault modes* that were not in the training set. A softmax classifier will always confidently output one of its trained classes regardless of what it receives.

---

## 5. Multi-Task Learning Design

### 5.1 The Loss Scale Problem

This is the most practically dangerous failure mode in the entire project.

At random initialization, the three task losses have different scales:
- Task A (Focal Loss, $\gamma = 2$): $\mathcal{L}_A \approx 0.2$–$0.5$ (binary)
- Task B (Cross-Entropy, 8 classes): $\mathcal{L}_B \approx \ln(8) = 2.08$ (multiclass)
- Task C (MAE, normalized targets): $\mathcal{L}_C \approx 0.03$–$0.08$

Naive equal weighting $\mathcal{L} = \mathcal{L}_A + \mathcal{L}_B + \mathcal{L}_C$ means Task B dominates by a factor of 4–70×. The shared encoder will optimize entirely for 8-class classification while completely ignoring anomaly detection and forecasting. This has been confirmed empirically in MTL papers (Kendall et al. 2018, Chen et al. 2018).

### 5.2 Kendall Uncertainty Weighting

The principled solution is to learn each task's weight as a function of its aleatoric uncertainty (Kendall et al. 2018):

$$
\mathcal{L}_{\text{total}} = \sum_{k=1}^{K} \frac{1}{2\sigma_k^2} \mathcal{L}_k + \log \sigma_k
$$

where $\sigma_k > 0$ is a learnable parameter per task. The $\log \sigma_k$ term prevents $\sigma_k \rightarrow \infty$ (trivial solution). The $1/\sigma_k^2$ factor automatically reduces the weight of high-uncertainty tasks.

**Implementation:**
```python
class KendallMTLLoss(nn.Module):
    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: list[Tensor]) -> Tensor:
        total = sum(
            torch.exp(-self.log_sigma[k]) * losses[k] + self.log_sigma[k]
            for k in range(len(losses))
        )
        return total
```

**Critical note:** `log_sigma` must be included in the optimizer's parameter group. The scheduler's warmup phase should freeze `log_sigma` for the first 2 epochs to let the task-specific heads stabilize before the weighting adapts — otherwise Task C (smallest initial loss) will get starved immediately.

**Monitor $\sigma_k$ during training:** Log `exp(log_sigma[k])` for each task per epoch. If any $\sigma_k$ diverges to infinity, that task has been abandoned by the multi-task optimizer. This is your diagnostic signal.

---

## 6. Hyperparameter Optimization Protocol

### 6.1 The HPO Leakage Trap

This failure mode is rarely discussed but extremely common in ML papers.

**The trap:** You run 100 HPO trials on your validation set. You select the best configuration by validation performance. Then you report validation performance as the model's performance. This is biased — by an amount approximately $O(\sqrt{\log N_{\text{trials}}})$ above true generalization performance.

**The correct protocol:**
1. Three-way split: Train / Val / Test
2. HPO runs on Val set only
3. Best configuration is retrained on Train+Val combined
4. Final metrics are computed on Test set **once**
5. Statistical significance is assessed on Test set with bootstrap CI

The Test set must be completely invisible to all HPO decisions, including the choice of early stopping epoch.

### 6.2 Two-Stage HPO

**Stage 1 — Structural search (Random Sampler, 20 trials):**
Search space: number of layers (2–8), hidden dimension (32–256, log), kernel size (3–7), dropout (0.0–0.5). Run for 3 epochs only per trial. Purpose: eliminate clearly bad architectural regions cheaply. Random sampler (not TPE) because TPE requires enough observations to build a surrogate model — with 20 trials across 4+ dimensions, TPE adds overhead without benefit.

**Stage 2 — Learning dynamics search (TPE Sampler, 80 trials):**
Fix structure from Stage 1 winner. Search space: learning rate (1e-5 to 1e-2, log), weight decay (1e-6 to 1e-3, log), batch size (32, 64, 128), warmup steps (100–2000), $\gamma$ for focal loss (1.0–4.0). Run for 30 epochs with early stopping (patience 5). TPE is effective here because the search space is lower-dimensional and the surface is smoother.

**Total compute budget:** 20 × 3 + 80 × 30 = 2,460 epoch-equivalents. On your development machine this is approximately 8–12 hours for Mendeley data. Plan accordingly — run overnight.

### 6.3 Stratified Time-Series Cross-Validation

Standard k-fold CV is invalid for time series (future data leaks into training). Use Purged Group Time-Series Split:

```
[──Train──][──Gap──][──Val──]
                              ← single fold
```

The gap (set to 2× maximum autocorrelation lag) prevents serial correlation leakage. For La Réunion at 7s sampling with expected fault duration ~10 minutes: autocorrelation lag ≈ 90 samples → gap = 180 samples = 21 minutes.

For Mendeley (simulated, IID by fault type), standard stratified k-fold is acceptable *within each fault type* because the data generation process is i.i.d. But still split by time within each fault type to avoid any implicit temporal ordering issues in the simulation.

---

## 7. Evaluation Protocol and Anti-Leakage Rules

### 7.1 The Six Leakage Checks (Non-Negotiable)

These are implemented in `src/evaluation/leakage_checks.py`. Run all six before any final result is reported.

**Check 1 — Label shuffle test:** Randomly permute training labels. Retrain model. If accuracy drops by less than 5 percentage points, your model is memorizing features independent of labels (likely temporal autocorrelation leaking through the split).

Expected outcome for a valid model: random labels → near-chance performance (50% for binary, 12.5% for 8-class).

**Check 2 — Duplicate sample check:** Near-duplicate windows across train/test splits (due to overlapping sliding windows) inflate test metrics dramatically. Assert: Levenshtein distance (or L2 distance) between any train window and test window is above threshold $\epsilon = 0.01 \cdot \sigma_{\text{feature}}$.

**Check 3 — Temporal integrity check:** Assert that $\max(t_{\text{train}}) < \min(t_{\text{val}}) < \min(t_{\text{test}})$ for all time-indexed datasets.

**Check 4 — Preprocessor fit scope:** The `FeaturePreprocessor` must be fit exclusively on the training split. Call `preprocessor.fit(X_train)` then `preprocessor.transform(X_val)` and `preprocessor.transform(X_test)` separately. If `fit_transform(X_full)` appears anywhere in the pipeline, it is a bug.

**Check 5 — Feature importance audit:** If any feature achieves importance > 0.5 in a single-feature Random Forest, inspect it. It is likely a target-encoding artifact or a feature derived from the label.

**Check 6 — Performance sanity check:** Any classifier achieving > 99% accuracy on real-world imbalanced data (La Réunion, Sonalgaz) without published precedent is suspect. Run the label shuffle test and check for duplicates first.

### 7.2 Statistical Significance Reporting

Every comparison between Model A and Model B in the thesis must include:
1. Point estimates (mean PR-AUC over N=5 seeds)
2. 95% bootstrap confidence interval (10,000 bootstrap samples)
3. Wilcoxon signed-rank test p-value (paired, non-parametric)
4. Cohen's d effect size (d < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, > 0.8 = large)

A result is worth reporting in the thesis if: p < 0.05 AND d > 0.5. Statistical significance with trivial effect size (d < 0.2) is not scientifically meaningful, even if p < 0.001.

---

## 8. Edge Deployment Architecture

### 8.1 Jetson Nano Hardware Constraints

| Resource | Available | Budget per inference |
|---|---|---|
| RAM | 4 GB shared CPU/GPU | ≤ 512 MB total model |
| GPU | 128-core Maxwell (21.2 GFLOPS FP32) | ≤ 128 MB GPU allocation |
| Latency target | — | ≤ 50 ms per window |
| Storage | 16 GB eMMC | ≤ 500 MB model artifacts |

**Latency budget breakdown (50 ms total):**
- Data acquisition from inverter (Modbus/serial read): 5 ms
- Preprocessing (scaling, window assembly): 3 ms
- ONNX/TensorRT inference: 30 ms
- Post-processing (threshold comparison, alert logic): 2 ms
- GUI update (pyqtgraph plot update): 10 ms

The 30 ms inference budget means: at INT8 precision on the Maxwell GPU, you can afford approximately 10M integer operations. This translates to a maximum TCN encoder of 6 layers × 64 channels × kernel 3 + GRU head 64 units. Do not exceed this unless profiling confirms otherwise.

### 8.2 ONNX Export — Non-Trivialities

**GRU export to ONNX:** PyTorch's `torch.onnx.export()` handles GRU via opset 13+ but requires explicit initial hidden state:
```python
h0 = torch.zeros(1, batch_size, hidden_size)
torch.onnx.export(
    model,
    (input_sequence, h0),  # both inputs required
    "model.onnx",
    opset_version=13,
    input_names=["input", "h0"],
    output_names=["output", "hn"],
    dynamic_axes={"input": {0: "batch"}, "h0": {1: "batch"}}
)
```
If `h0` is created internally in `forward()`, ONNX export will fail with a shape inference error. Refactor the model before attempting export.

**TCN export:** No issues with standard dilated convolutions. Residual connections export cleanly via opset 9+.

**LSTM Autoencoder (Task A):** The decoder uses a transposed operation. Replace any `nn.ConvTranspose1d` with `nn.Upsample + nn.Conv1d` if TensorRT fails — ConvTranspose1d has known quantization issues in TRT 8.x.

### 8.3 TensorRT INT8 Calibration

INT8 quantization requires a calibration dataset of 300–500 representative samples. This is not optional — without calibration, INT8 accuracy degrades catastrophically for the anomaly detection task (reconstruction error thresholds are sensitive to quantization noise).

**Calibration dataset composition:** 400 windows — 300 normal (from training set), 100 fault samples (if available). The calibration set is NOT the test set. It does not affect evaluation metrics.

```python
# TensorRT INT8 calibration
import tensorrt as trt
class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, cache_file):
        ...
    def get_batch(self, names):
        # Must return exactly one batch from calibration data
        ...
```

**Maxwell GPU INT8 throughput:** Maxwell supports INT8 with 2× the FP16 throughput. However, FP16 on Maxwell must be enabled via `builder.fp16_mode = True`. The effective speedup chain: FP32 → FP16 → INT8 = 1× → 1.5× → 2×. Validate latency with `trtexec --int8 --calib=calibration.cache`.

---

## 9. Domain Adaptation — The Research Gap

### 9.1 Why Standard Training Fails Across Datasets

A model trained on Mendeley data and tested on La Réunion data will likely achieve < 50% accuracy on fault classification. This is not a model quality issue — it is a distribution shift issue.

**Covariate shift characterization:**

Define the marginal distributions $P_s(X)$ (source: Mendeley, simulated) and $P_t(X)$ (target: La Réunion, real). The Maximum Mean Discrepancy (MMD) measures the distance between distributions:

$$
\text{MMD}^2(P_s, P_t) = \left\| \frac{1}{n_s} \sum_{i=1}^{n_s} \phi(x_i^s) - \frac{1}{n_t} \sum_{j=1}^{n_t} \phi(x_j^t) \right\|_{\mathcal{H}}^2
$$

where $\phi$ is a kernel feature map (RBF kernel with bandwidth $\sigma$). Compute this *before training* on per-feature basis to quantify which features have high shift.

**Expected high-shift features:** DC bus voltage (Mendeley simulated at fixed 300V vs La Réunion real varies 280–320V), Temperature (Mendeley simulation temperature = fixed vs La Réunion tropical climate average 28°C vs Sonalgaz desert climate average 42°C).

**Expected low-shift features:** Normalized Performance Ratio (dimensionless, physics-derived), $\Delta P / \bar{P}$ (relative change), fault rate-of-change features.

**Design implication:** Your physics-informed features (PR, normalized $\Delta P / \bar{P}$) will transfer across domains better than raw electrical values. This is a strong argument for including them.

### 9.2 Domain-Adversarial Neural Network (DANN)

DANN (Ganin et al. 2016) is the standard approach for domain-invariant feature learning:

```
Input → Shared Encoder F(·) → Task Classifier C(·) → Task Loss
                             ↘ Domain Classifier D(·) → Domain Loss (reversed)
```

The gradient reversal layer negates gradients during backpropagation from the domain classifier:
$$
\mathcal{L}_{\text{DANN}} = \mathcal{L}_{\text{task}} - \lambda \cdot \mathcal{L}_{\text{domain}}
$$

The shared encoder is simultaneously optimized to:
1. Minimize task loss (learn fault-discriminative features)
2. Maximize domain classification loss (make features domain-indistinguishable)

**When to attempt this:** Only after the single-domain baselines are solid and logged in MLflow. DANN requires unlabeled target domain samples during training — La Réunion provides this if training on Mendeley. The $\lambda$ schedule (start 0, anneal to 1 over training) is critical for stable training.

**Expected outcome:** DANN-adapted Mendeley→La Réunion transfer should achieve F1 on fault classes within 15% of fully supervised La Réunion training. If it achieves within 5%, this is publishable as a primary contribution.

### 9.3 Online Drift Detection (Sonalgaz)

For the Sonalgaz deployment scenario (real operational Algeria data, no labels), use ADWIN (ADaptive WINdowing) for distribution shift detection:

```python
from river.drift import ADWIN
detector = ADWIN()
for x in stream:
    detector.update(x)
    if detector.drift_detected:
        # trigger model recalibration
```

ADWIN maintains a sliding window that automatically adjusts its size based on detected changes in the mean. When a drift is detected, the anomaly detection threshold should be re-estimated on recent normal windows. Log drift detections to MLflow as events (not metrics) to build a post-hoc understanding of the deployment environment.

---

## 10. Eliminated Techniques — With Reasons

This section documents what was explicitly decided NOT to use, and why. This prevents wasted time revisiting these decisions.

### 10.1 SMOTE for Class Imbalance
**What it does:** Interpolates synthetic minority-class samples in feature space.  
**Why eliminated:** SMOTE's interpolation assumption — that linear interpolation between two fault samples produces a valid fault signal — is false for time-series electrical signals. A linear interpolation between a DC arc fault window and a partial shading window produces a physically impossible signal. Use Focal Loss + class-balanced batch sampling instead. Focal Loss down-weights easy negatives automatically:
$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log p_t
$$
with $\gamma = 2$ (standard), $\alpha_t$ set inverse-proportional to class frequency.

### 10.2 RBM and Deep Belief Networks
**What they do:** Generative models based on Restricted Boltzmann Machines, stacked into DBNs.  
**Why eliminated:** Training instability (contrastive divergence approximation), no GPU batch parallelism in standard implementations, no integration with modern autograd (requires manual MCMC computation). Superseded by VAEs and reconstruction autoencoders which achieve better anomaly detection with cleaner training. The thesis chapter covering DBNs is literature review only; do not implement.

### 10.3 Swin Transformer
**What it does:** Vision Transformer with shifted windows for 2D inputs.  
**Why eliminated:** Designed for image classification. For time-series, you would need to convert the 1D signal to a 2D image (e.g., GASF), increasing input dimensions by $T^2/T = T$. At $T=60$ this 60× input size expansion dominates all compute advantages of the architecture. Furthermore, Swin Transformer was not pretrained on any electrical signal data — there is no transfer learning benefit. TCN achieves comparable or better accuracy with 50× less compute on 1D time-series.

### 10.4 ARIMA/SARIMA as Deployment Model
**What it does:** Linear time-series forecasting with seasonal components.  
**Why eliminated:** ARIMA is appropriate as a baseline comparison only. As a deployment product, it cannot learn multivariate nonlinear fault dynamics. It is also CPU-only and not exportable to ONNX/TensorRT. Include it in the baseline comparison table for Task C but do not include it in the Jetson Nano deployment pipeline.

### 10.5 Full tsfresh Feature Extraction
**What it does:** Extracts ~800 statistical features from time-series signals.  
**Why eliminated:** 800 features × 2.16M Mendeley rows × 60-sample windows = significant compute + memory. Many features are redundant (multiple autocorrelation lag values, multiple quantiles). Use tsfresh in `minimal` mode (15 features) for the ML baselines only. For DL models, the raw window is the input — feature extraction is handled by the learned convolutional filters.

### 10.6 Prophet
**Why eliminated:** Designed for daily/weekly seasonality in business time-series (sales, traffic). PV data has sub-minute dynamics and fault patterns that are non-seasonal. Prophet's additive model is fundamentally mismatched to the fault forecasting task (Task C). No computation budget should be spent on it.

### 10.7 Muon Optimizer as Primary
**What it does:** Orthogonalizes weight updates using Newton-Schulz iteration for faster convergence.  
**Why de-prioritized:** Excellent theoretical properties but requires careful tuning of the Newton-Schulz iteration count (typically 5–10 steps). The matmul overhead per update step is non-trivial and incompatible with TensorRT deployment (optimizer state is training-only, but the additional tuning time competes with other Week 3–4 priorities). Use AdamW as primary. Add Muon as an optional late-stage experiment only if training time permits.

---

## 11. Predicted Experimental Outcomes

These are falsifiable predictions based on dataset characteristics and algorithm properties. Track actual vs predicted outcomes in MLflow. Deviations from predictions are scientifically interesting and should be analyzed.

### 11.1 Task A — Anomaly Detection on La Réunion

| Model | Predicted PR-AUC | Rationale |
|---|---|---|
| Isolation Forest (raw features) | 0.45–0.55 | Does not model temporal structure; irradiance variation creates false positives |
| Isolation Forest (with PR feature) | 0.58–0.65 | PR normalizes irradiance; reduces false positives substantially |
| LSTM Autoencoder (raw) | 0.65–0.72 | Temporal modeling helps; some normal variation captured |
| LSTM Autoencoder (with $\Delta P$ feature) | 0.75–0.85 | Differential inverter signal removes common-mode noise |
| MTL-TCN (full system) | 0.78–0.88 | Shared encoder benefits from fault classification signal |

### 11.2 Task B — Fault Classification on Mendeley

| Model | Predicted F1-macro | Rationale |
|---|---|---|
| Random Forest (tsfresh minimal) | 0.82–0.88 | Mendeley is simulated, features are clean and separable |
| XGBoost (physics + window stats) | 0.88–0.93 | Tree ensembles perform near-optimally on tabular data without temporal structure |
| 1D-CNN | 0.90–0.95 | Learns fault-specific spectral patterns from raw signal |
| TCN + CEEMDAN features | 0.92–0.97 | CEEMDAN IMFs expose high-frequency arc/transient structure |
| MTL-TCN | 0.90–0.96 | Slight MTL regularization benefit; shared encoder adds noise if La Réunion diverges |

**Note:** Very high accuracy on Mendeley (> 0.97) is achievable but may indicate overfitting to simulation artifacts. Cross-validate the top model on La Réunion fault subset to check generalization.

### 11.3 Task B — Cross-Domain Transfer (Mendeley → La Réunion)

| Model | Predicted F1-macro | Rationale |
|---|---|---|
| Direct transfer (no adaptation) | 0.15–0.30 | Distribution shift between simulated and real; voltage/current scales differ |
| Transfer + physics features only | 0.35–0.50 | Physics features (PR, ΔP/P) are domain-invariant |
| DANN adaptation | 0.55–0.70 | Domain-adversarial training closes majority of distribution gap |
| Full supervision on La Réunion | 0.65–0.80 | Limited by small fault sample count (~84K windows at 3% fault rate) |

### 11.4 Task C — 5-minute Fault Forecasting

| Model | Predicted MAE (normalized) | Rationale |
|---|---|---|
| Persistence baseline (y_hat = y_t) | 0.08–0.12 | PV output is autocorrelated; persistence is a strong baseline |
| ARIMA (univariate) | 0.06–0.09 | Captures linear autocorrelation; limited by univariate assumption |
| GRU (multivariate) | 0.04–0.07 | Multivariate context (irradiance, temperature) helps 5-min horizon |
| TCN-GRU head (MTL) | 0.03–0.06 | Shared encoder leverages fault classification knowledge for anomaly trajectory |

**Critical check:** If the GRU forecasting model achieves MAE < 0.02 (normalized), verify for leakage. A model that perfectly forecasts anomaly scores has likely seen future data in its input window.

---

## 12. Implementation Checklist

Before calling any result "final":
- [ ] All 6 leakage checks pass (`run_leakage_report()`)
- [ ] Every metric reported has ±bootstrap CI in MLflow
- [ ] Train/val/test split is time-based (not random) for La Réunion and Sonalgaz
- [ ] `FeaturePreprocessor.fit()` called only on training split
- [ ] Differential signal $\alpha$ coefficient estimated on training split only
- [ ] CEEMDAN features computed only for Mendeley
- [ ] Wavelet features computed only for La Réunion
- [ ] All 5 random seeds evaluated (report mean ± std)
- [ ] MTL $\sigma_k$ values logged per epoch (check no task abandoned)
- [ ] ONNX model validated: `torch_output ≈ onnx_output` (L2 < 1e-4)
- [ ] TensorRT INT8 model validated against ONNX output
- [ ] Latency profiled on Jetson Nano (target: ≤ 50 ms per window)
- [ ] Statistical test (Wilcoxon p < 0.05 + Cohen's d > 0.5) for every model comparison claimed in thesis

---

*End of Technical Design Document*
