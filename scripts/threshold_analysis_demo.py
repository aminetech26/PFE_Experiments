"""Measure blindfolded threshold strategies for affine PC-Flow on one fold.

Trains a short affine PC-Flow run on the default Costa split and reports:
  1. Temporal-persistence sweep (N-consecutive debounce) at the GPD threshold.
  2. Operating-point F1 spectrum: GPD (deployable, blindfolded) vs val-tuned
     (supervised, transferred to test) vs test-oracle (unachievable upper bound).
  3. Normal-score vs (irr, pvt) diagnostic — would a conditional threshold help?

This is a measurement harness, not a training entrypoint: no MLflow, no artifacts.
Epochs are capped so it runs on CPU in a few minutes; numbers are indicative of
the operating-point behaviour, which is robust to small changes in model quality.

Run:
    uv run python -m scripts.threshold_analysis_demo --max-epochs 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.evaluation.threshold_analysis import (
    apply_persistence_filter,
    f1_optimal_threshold,
    metrics_at_threshold,
    operating_point_nll_diagnostic,
    persistence_sweep,
)
from src.modeling.anomaly_detection.dl.dataset import TimeSeriesDataset
from src.modeling.anomaly_detection.dl.pc_flow.model import PCFlowModel
from src.modeling.anomaly_detection.dl.pc_flow.trainer import PCFlowLightningModule
from src.modeling.common.feature_loader import load_features_for_task
from src.modeling.common.threshold_calibration import calibrate_threshold, load_threshold_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    with (PROJECT_ROOT / "configs" / "model_config.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="costa")
    p.add_argument("--profile", default="plus_physics")
    p.add_argument("--task", default="anomaly_semisup")
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)
    config = _load_config()
    cfg = config["anomaly_detection"]["dl"]["models"]["pc_flow"]

    train_df, val_df, test_df, manifest, _ = load_features_for_task(
        task=args.task, profile=args.profile, dataset=args.dataset, split_path="path_a",
    )
    features: list[str] = manifest.get("final_features", [])
    label_col = str(manifest.get("label_column", "label"))
    feature_idx = {n: i for i, n in enumerate(features)}

    ctx_names = cfg.get("context_features", ["irr", "pvt"])
    ctx_idx = [feature_idx[f] for f in ctx_names if f in feature_idx]
    n_non_context = len(features) - len(ctx_idx)

    # Capture raw irr/pvt for the diagnostic BEFORE scaling (physical units).
    irr_raw = test_df["irr"].to_numpy() if "irr" in test_df else None
    pvt_raw = test_df["pvt"].to_numpy() if "pvt" in test_df else None

    scaler = StandardScaler()
    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])
    mean_t = torch.tensor(scaler.mean_, dtype=torch.float32)
    scale_t = torch.tensor(scaler.scale_, dtype=torch.float32)

    bs = int(cfg.get("batch_size", 4096))

    def ds(df, normal_only):
        return TimeSeriesDataset(df, features, label_col, 1, 1, normal_only=normal_only,
                                 return_original_label=True, return_group_id=True)
    train_dl = DataLoader(ds(train_df, True), batch_size=bs, shuffle=True)
    val_dl = DataLoader(ds(val_df, False), batch_size=bs, shuffle=False)
    test_dl = DataLoader(ds(test_df, False), batch_size=bs, shuffle=False)
    train_dl_seq = DataLoader(ds(train_df, True), batch_size=bs, shuffle=False)

    model = PCFlowModel(n_features=n_non_context, n_context=len(ctx_idx),
                        n_coupling_layers=int(cfg.get("n_coupling_layers", 4)),
                        hidden_dim=int(cfg.get("hidden_dim", 32)),
                        dropout=float(cfg.get("dropout", 0.0)),
                        coupling_type="affine")  # affine = the chosen model
    logger.info("Affine PC-Flow params: {:,}", model.n_params)

    lit = PCFlowLightningModule(
        model=model, scaler_mean=mean_t, scaler_scale=scale_t,
        feature_idx=feature_idx, context_feature_indices=ctx_idx,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        max_epochs=args.max_epochs, total_steps=max(1, len(train_dl) * args.max_epochs),
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator="cpu",
        callbacks=[EarlyStopping(monitor="val_macro_per_class_pr_auc_monitor", patience=6, mode="max")],
        enable_progress_bar=False, enable_model_summary=False, logger=False,
        enable_checkpointing=False, gradient_clip_val=1.0,
    )
    trainer.fit(lit, train_dataloaders=train_dl, val_dataloaders=val_dl)
    trainer.validate(lit, dataloaders=val_dl)
    trainer.test(lit, dataloaders=test_dl)

    thr_cfg = load_threshold_config(config, cfg)
    train_scores = lit.collect_train_scores(train_dl_seq)
    gpd_thr, _ = calibrate_threshold(train_scores, **thr_cfg)

    val_s, val_l = lit._val_scores_np, lit._val_labels_np
    test_s, test_l = lit._test_scores_np, lit._test_labels_np
    test_g = lit._test_group_ids_np

    # ── 1. Persistence sweep at the GPD (blindfolded) threshold ──────────────
    sweep = persistence_sweep(test_s, test_l, test_g, gpd_thr, n_values=(1, 3, 5, 10))
    print("\n=== 1. TEMPORAL PERSISTENCE SWEEP (blindfolded, GPD threshold) ===")
    print(f"{'N':>3} {'precision':>10} {'recall':>9} {'F1':>8} {'alarm_rate':>11} {'latency_s':>10}")
    for r in sweep:
        print(f"{r['n_consecutive']:>3} {r['precision']:>10.4f} {r['recall']:>9.4f} "
              f"{r['f1']:>8.4f} {r['alarm_rate']:>11.4f} {r['detection_latency_samples']:>10}")

    # ── 2. Operating-point F1 spectrum ───────────────────────────────────────
    gpd = metrics_at_threshold(test_s, test_l, gpd_thr)
    val_thr, _ = f1_optimal_threshold(val_s, val_l)          # tuned on val faults
    val_tuned = metrics_at_threshold(test_s, test_l, val_thr)  # applied to test
    oracle_thr, oracle_f1 = f1_optimal_threshold(test_s, test_l)  # upper bound
    print("\n=== 2. OPERATING-POINT F1 SPECTRUM (price of deployability) ===")
    print(f"{'strategy':>26} {'threshold':>10} {'precision':>10} {'recall':>9} {'F1':>8}")
    print(f"{'GPD (deployable, blind)':>26} {gpd['threshold']:>10.3f} {gpd['precision']:>10.4f} {gpd['recall']:>9.4f} {gpd['f1']:>8.4f}")
    print(f"{'val-tuned -> test':>26} {val_tuned['threshold']:>10.3f} {val_tuned['precision']:>10.4f} {val_tuned['recall']:>9.4f} {val_tuned['f1']:>8.4f}")
    print(f"{'test-oracle (upper bd)':>26} {oracle_thr:>10.3f} {'':>10} {'':>9} {oracle_f1:>8.4f}")
    transfers = abs(val_tuned["f1"] - oracle_f1) < 0.02
    print(f"  -> val-tuned threshold {'TRANSFERS' if transfers else 'does NOT transfer'} "
          f"(test F1 {val_tuned['f1']:.4f} vs oracle {oracle_f1:.4f})")

    # ── 3. Normal-score vs operating-point diagnostic ────────────────────────
    if irr_raw is not None and pvt_raw is not None and len(test_s) == len(irr_raw):
        normal_mask = (test_l == 0)
        diag = operating_point_nll_diagnostic(
            test_s[normal_mask], irr_raw[normal_mask], pvt_raw[normal_mask],
        )
        print("\n=== 3. NORMAL-SCORE vs OPERATING-POINT (conditional-threshold need) ===")
        for name, d in diag.items():
            print(f"  {name}: pearson_corr={d['pearson_corr']:+.3f}  bin_spread_ratio={d['bin_spread_ratio']:.3f}")
        max_spread = max(d["bin_spread_ratio"] for d in diag.values())
        verdict = ("conditioning INCOMPLETE -> conditional threshold would help"
                   if max_spread > 0.30 else
                   "conditioning OK -> global threshold sufficient")
        print(f"  -> {verdict} (max bin_spread_ratio={max_spread:.3f})")
    else:
        print("\n[3] skipped: irr/pvt unavailable or length mismatch")

    # ── 4. SENSITIVE THRESHOLD + HYSTERESIS, selected on VAL, reported on TEST ─
    # Honesty protocol: threshold candidates are blindfolded (GPD at increasing
    # target FPR = more sensitive) plus the val-F1-optimal (val labels, disclosed);
    # N is a latency budget. The (threshold, N) pair is SELECTED BY VAL F1, then
    # evaluated once on TEST. No test labels touch the selection.
    val_g = lit._val_group_ids_np
    candidates: dict[str, float] = {}
    for tp in (0.05, 0.10, 0.20):  # higher target FPR -> lower (more sensitive) threshold
        t, _ = calibrate_threshold(train_scores, strategy="gpd", pot_quantile=0.90, target_tail_prob=tp)
        candidates[f"gpd_fpr{tp:.2f}"] = t
    candidates["val_f1_opt"] = val_thr  # supervised operating point (disclosed)

    print("\n=== 4. SENSITIVE THRESHOLD + HYSTERESIS (select on VAL, report TEST) ===")
    print(f"{'thr_strategy':>13} {'thr':>7} {'N':>3} | {'val_F1':>7} | {'test_F1':>7} {'test_P':>7} {'test_R':>7}")
    best: dict | None = None
    baseline_test_f1 = sweep[0]["f1"]  # GPD, N=1 (raw deployable baseline)
    for name, t in candidates.items():
        for n in (1, 3, 5, 10):
            vp = apply_persistence_filter((val_s >= t).astype(int), val_g, n)
            tp_ = apply_persistence_filter((test_s >= t).astype(int), test_g, n)
            vf1 = float(f1_score(val_l, vp, zero_division=0))
            tf1 = float(f1_score(test_l, tp_, zero_division=0))
            t_prec = float(precision_score(test_l, tp_, zero_division=0))
            t_rec = float(recall_score(test_l, tp_, zero_division=0))
            print(f"{name:>13} {t:>7.2f} {n:>3} | {vf1:>7.4f} | {tf1:>7.4f} {t_prec:>7.4f} {t_rec:>7.4f}")
            if best is None or vf1 > best["val_f1"]:
                best = {"name": name, "thr": t, "n": n, "val_f1": vf1,
                        "test_f1": tf1, "test_p": t_prec, "test_r": t_rec}
    print(f"\n  SELECTED on val: {best['name']} thr={best['thr']:.2f} N={best['n']} "
          f"(val F1={best['val_f1']:.4f})")
    print(f"  -> TEST: F1={best['test_f1']:.4f}  P={best['test_p']:.4f}  R={best['test_r']:.4f}")
    print(f"  vs GPD/N=1 deployable baseline: F1={baseline_test_f1:.4f}  "
          f"(delta {best['test_f1'] - baseline_test_f1:+.4f})")


if __name__ == "__main__":
    main()
