# Compute Strategy — Local PC vs Google Colab Pro+

**Context:** Local machine has no GPU. Colab Pro+ provides T4/A100 with **600 compute units/month (1800 total over 3 months)**. That's a solid budget — use GPU freely when it genuinely accelerates work, just don't burn units on tasks that get zero GPU benefit.

---

## Decision Rule (Simple)

| If the task is... | Run on | Why |
|---|---|---|
| **CPU-bound or I/O-bound** | Local | GPU won't help. Don't waste units. |
| **GPU-accelerated AND takes > 2 min on CPU** | Colab | Worth the overhead of uploading data. |
| **Interactive / exploratory** | Local | Colab's session management adds friction. |

---

## Per-Stage Breakdown

### Stage 1 — Ingestion & EDA

| Task | Where | Rationale |
|---|---|---|
| CSV → Parquet conversion | **Local** | I/O-bound (disk read/write). Polars is fast on CPU. |
| EDA notebook (stats, plots) | **Local** | pandas/matplotlib are CPU-only. No GPU benefit. |
| Missing value analysis | **Local** | Pure pandas operations. |
| Correlation / VIF / MI | **Local** | scikit-learn MI and statsmodels VIF are CPU-only. |
| ADF stationarity tests | **Local** | statsmodels, single-threaded. |
| Autocorrelation (ACF) | **Local** | statsmodels, fast even on 3M rows. |

**Colab needed: Never** for this stage.

---

### Stage 2 — Feature Engineering

| Task | Where | Rationale |
|---|---|---|
| Physics features (PR, dP/dt, ΔP) | **Local** | Vectorized pandas/numpy. Fast. |
| Sliding window creation | **Local** | NumPy array operations. CPU-efficient. |
| Window statistics (mean, std, skew...) | **Local** | scipy.stats on numpy arrays. |
| Wavelet denoising (La Réunion) | **Local** | PyWavelets is CPU-only. ~3M rows takes a few minutes. |
| Wavelet energy features | **Local** | Same as above. |
| CEEMDAN (Mendeley, later) | **Local** | EMD-signal is CPU-only. Slow but GPU won't help. Cache result to Parquet. |
| tsfresh minimal features | **Local** | CPU-only. Use `n_jobs=-1` for parallelism. |
| Feature selection (filter methods) | **Local** | MI, correlation — all CPU. |
| GASF image generation | **Local** | NumPy matrix ops. Generate once, save as .npy. |

**Colab needed: Never** for this stage. Everything is CPU or I/O bound.

---

### Stage 3 — Splitting & Preprocessing

| Task | Where | Rationale |
|---|---|---|
| Time-based train/val/test split | **Local** | Index operations. Instant. |
| RobustScaler fit + transform | **Local** | scikit-learn, CPU. |
| Leakage checks (pre-model) | **Local** | Assertions and pandas ops. |
| Preprocessor serialization (.pkl) | **Local** | pickle.dump. |

**Colab needed: Never.**

---

### Stage 4 — ML Baselines (Round 1)

| Task | Where | Rationale |
|---|---|---|
| Isolation Forest | **Local** | scikit-learn, CPU-only. Fast even on 100K windows. |
| One-Class SVM | **Local** | CPU-only. May be slow on large N — subsample if needed. |
| LightGBM | **Local** | CPU training is fast (LightGBM is optimized for CPU). |
| CatBoost | **Local** | Has GPU mode BUT overhead of Colab setup > time saved for ≤500K rows. |
| XGBoost | **Local** | Same reasoning as CatBoost. |
| Extra Trees / Random Forest | **Local** | CPU-only. Use `n_jobs=-1`. |
| Optuna HPO (20-50 trials, ML models) | **Local** | Each trial is seconds. Total: minutes. |
| Post-model leakage checks | **Local** | Label shuffle test retrains ML models — still fast on CPU. |
| Bootstrap CI + Wilcoxon tests | **Local** | Pure numpy. |

**Colab needed: Never** for ML baselines. Tree models and sklearn don't benefit meaningfully from GPU at this dataset size.

---

### Stage 5 — Deep Learning (Round 2) ← **THIS IS WHERE COLAB MATTERS**

| Task | Where | Rationale |
|---|---|---|
| Single DL model training (debug run, 3-5 epochs) | **Local** | Verify code works. CPU is fine for a few epochs. |
| Full DL training (30-100 epochs) | **Colab (T4)** | 10-50× speedup over CPU. |
| Optuna HPO Stage 1 (20 trials × 3 epochs) | **Colab (T4)** | 60 epoch-equivalents. ~30 min on T4 vs hours on CPU. |
| Optuna HPO Stage 2 (80 trials × 30 epochs) | **Colab (A100)** | 2400 epoch-equivalents. Use A100 — this is the biggest compute block. |
| Multi-seed evaluation (5 seeds × best config) | **Colab (T4)** | 5 full training runs. |
| LSTM Autoencoder (Task A) | **Colab (T4)** | Sequence models benefit from GPU. |
| TCN / GRU / CNN-1D training | **Colab (T4)** | Convolutions and RNNs are GPU-accelerated. |
| Multi-Task Learning (shared encoder) | **Colab (T4/A100)** | Larger model, 3 heads. |
| DANN domain adaptation (later) | **Colab (A100)** | Two-domain training, adversarial loss. Needs fast iteration. |

**Colab compute budget estimate (Stage 5):**

| Sub-task | GPU | Hours (est.) | ~Units |
|---|---|---|---|
| HPO Stage 1 (structural) | T4 | 0.5–1h | ~2–4 |
| HPO Stage 2 (learning dynamics) | A100 | 2–4h | ~15–30 |
| Best model × 5 seeds × 3 tasks | T4 | 1–2h | ~4–8 |
| MTL training + tuning | T4 | 2–3h | ~8–12 |
| DANN (if attempted) | A100 | 2–3h | ~15–20 |
| Iterative experiments / reruns | T4 | 3–5h | ~12–20 |
| **Total GPU estimate** | | **~12–18h** | **~55–95 units** |

With 600 units/month this is well within budget. You have room for experimentation — don't stress about using GPU when it helps. Just avoid the anti-patterns below.

---

### Stage 6 — Edge Deployment

| Task | Where | Rationale |
|---|---|---|
| ONNX export | **Local** | `torch.onnx.export()` is CPU-only. |
| ONNX validation (output parity check) | **Local** | onnxruntime CPU inference. |
| TensorRT conversion | **Jetson Nano** | TRT must run on target hardware (ARM + Maxwell GPU). |
| INT8 calibration | **Jetson Nano** | Calibration requires the target GPU architecture. |
| Latency profiling | **Jetson Nano** | Must measure on actual deployment hardware. |
| GUI development (PySide6) | **Local** | Desktop GUI. CPU. |

**Colab needed: Never** for deployment. TRT requires Jetson.

---

## Colab Session Discipline

### Before opening a Colab session:
1. **All data is preprocessed** — Parquet files ready, no raw CSV loading on GPU.
2. **Code is tested locally** — Run 2-3 epochs on CPU to catch bugs. Don't debug on GPU.
3. **HPO search space is defined** — Don't explore interactively on Colab.
4. **Know exactly what you'll run** — Write the training script locally, upload, execute.

### During a Colab session:
- Upload only processed Parquet files (small) — not raw CSVs (large).
- Use `%%time` on training cells to track real GPU utilization.
- Save checkpoints to Google Drive every N epochs (Colab sessions can disconnect).
- Log everything to MLflow/DagsHub — don't rely on Colab's runtime memory.

### After a Colab session:
- Download: best model checkpoint (.pt), MLflow metrics, any generated plots.
- Push to DagsHub/DVC if appropriate.
- **Disconnect runtime immediately** — idle GPU still burns compute units.

---

## Anti-Patterns (Actual Wastes)

| Mistake | Why it wastes units |
|---|---|
| Running EDA / pandas on Colab GPU | Zero GPU benefit. These are CPU-only ops. |
| Loading raw CSVs on Colab | Upload the processed Parquet, not the 2GB CSV. |
| Leaving Colab idle for 2+ hours | Idle runtime still burns units. Disconnect when done. |
| Debugging broken code on A100 | Quick sanity check locally (2 CPU epochs) catches most bugs for free. |

---

## File Transfer Strategy

```
LOCAL (preprocessing)              COLAB (GPU training)
─────────────────────              ────────────────────
data/processed/*.parquet    ──→    /content/drive/data/
configs/*.yaml              ──→    /content/drive/configs/
src/models/*.py             ──→    /content/drive/src/
src/training/*.py           ──→    /content/drive/src/

                            ←──    experiments/checkpoints/*.pt
                            ←──    experiments/metrics/*.json
                            ←──    MLflow logs (auto-sync to DagsHub)
```

**Best practice:** Mount Google Drive in Colab. Keep processed data + training scripts in Drive. This avoids re-uploading every session.

---

## GPU Selection Guide

| Task | GPU | Why |
|---|---|---|
| Quick experiments, single model training | **T4** (16GB) | Cheapest. Sufficient for batch_size=256, hidden_dim=128. |
| HPO Stage 2 (80 trials) | **A100** (40GB) | Faster per-trial → fewer total units burned despite higher per-hour cost. |
| MTL with 3 task heads | **T4** | Model fits in 16GB. |
| DANN (two-domain batches) | **A100** | Double batch memory. A100's memory bandwidth helps. |
| Inference benchmarking | **T4** | Closer to Jetson Nano's Maxwell arch than A100 (not perfect but better). |

**Rule of thumb:** If total GPU time for a task is < 2h, use T4. If > 2h, A100 saves total compute units because fewer wall-clock hours × higher throughput = less total cost even at higher per-hour rate.
