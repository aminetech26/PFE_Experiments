"""
Leakage Prevention & Validation Suite

Implements all checks from the leakage prevention protocol:
  1. Label Shuffle Test
  2. Duplicate Sample Check
  3. Time-Split Validation
  4. Feature Importance Audit
  5. Performance Sanity Check
  6. Bootstrap / Resample Validation

Run this after every major training stage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit


# ============================================================================
# 1. LABEL SHUFFLE TEST (Target Leakage)
# ============================================================================

def label_shuffle_test(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_shuffles: int = 5,
    metric: str = "f1_weighted",
) -> dict:
    """
    Train model with shuffled labels. Score should drop to near-random.
    If it doesn't → target leakage present.

    Returns dict with real_score, shuffle_scores, is_leaking.
    """
    import copy
    from functools import partial
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    metric_fns: dict = {
        "f1_weighted": partial(f1_score, average="weighted", zero_division=0),
        "f1_macro": partial(f1_score, average="macro", zero_division=0),
        "f1_binary": partial(f1_score, average="binary", zero_division=0),
        "accuracy": accuracy_score,
    }
    score_fn = metric_fns.get(metric, partial(f1_score, average="weighted", zero_division=0))

    # Real score
    real_model = copy.deepcopy(model)
    real_model.fit(X_train, y_train)
    y_pred_real = real_model.predict(X_val)
    real_score = float(score_fn(y_val, y_pred_real))

    # Shuffle scores
    shuffle_scores = []
    for _ in range(n_shuffles):
        shuffled_model = copy.deepcopy(model)
        y_shuffled = np.random.permutation(y_train)
        shuffled_model.fit(X_train, y_shuffled)
        y_pred_shuffle = shuffled_model.predict(X_val)
        shuffle_scores.append(float(score_fn(y_val, y_pred_shuffle)))

    mean_shuffle = float(np.mean(shuffle_scores))
    is_leaking = mean_shuffle > 0.5 * real_score

    result = {
        "metric": metric,
        "real_score": real_score,
        "mean_shuffle_score": mean_shuffle,
        "is_leaking": is_leaking,
    }
    if is_leaking:
        logger.warning(
            "LEAKAGE ALERT (Label Shuffle): real={:.3f}, shuffle={:.3f} (unexpectedly high!)",
            real_score,
            mean_shuffle,
        )
    else:
        logger.success(
            "Label Shuffle OK: real={:.3f}, shuffle={:.3f}", real_score, mean_shuffle
        )
    return result


# ============================================================================
# 2. DUPLICATE SAMPLE CHECK (Data Leakage)
# ============================================================================

def duplicate_sample_check(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Check if any feature rows in val are exact duplicates of train rows.
    This indicates data from the future was used during training.
    """
    train_set = set(df_train[feature_cols].apply(tuple, axis=1))
    val_set = set(df_val[feature_cols].apply(tuple, axis=1))
    overlap = train_set & val_set
    overlap_count = len(overlap)
    overlap_pct = overlap_count / len(val_set) * 100

    result = {
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "overlapping_samples": overlap_count,
        "overlap_pct": overlap_pct,
        "is_leaking": overlap_pct > 1.0,  # >1% is suspicious
    }
    if result["is_leaking"]:
        logger.warning(f"LEAKAGE ALERT (Duplicates): {overlap_pct:.1f}% of val samples "
                       f"are exact duplicates of train samples!")
    else:
        logger.success(f"Duplicate Check OK: {overlap_pct:.2f}% overlap")
    return result


def exact_duplicate_overlap_check(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Strict exact-duplicate overlap check with zero-tolerance policy."""
    train_set = set(df_train[feature_cols].apply(tuple, axis=1))
    val_set = set(df_val[feature_cols].apply(tuple, axis=1))
    overlap = train_set & val_set
    overlap_count = len(overlap)
    overlap_pct = (overlap_count / len(val_set) * 100.0) if len(val_set) > 0 else 0.0

    result = {
        "train_unique_samples": len(train_set),
        "val_unique_samples": len(val_set),
        "overlapping_samples": overlap_count,
        "overlap_pct": overlap_pct,
        "is_clean": overlap_count == 0,
    }
    if result["is_clean"]:
        logger.success(f"Exact Duplicate Pre-Check OK: {overlap_count} overlaps")
    else:
        logger.warning(
            "LEAKAGE ALERT (Exact Duplicates): {} overlapping rows ({:.4f}%) between train/val",
            overlap_count,
            overlap_pct,
        )
    return result


# ============================================================================
# 3. TIME-SPLIT VALIDATION (Temporal Leakage)
# ============================================================================

def validate_time_split(
    df: pd.DataFrame,
    time_col: str,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
) -> dict:
    """
    Verify that the train/val split respects temporal ordering.
    """
    train_max = df[df[time_col] <= train_end][time_col].max()
    val_min = df[df[time_col] >= val_start][time_col].min()

    gap = (val_min - train_max).total_seconds() if pd.notna(train_max) and pd.notna(val_min) else np.nan
    is_valid = train_max < val_min if pd.notna(train_max) and pd.notna(val_min) else False

    result = {
        "train_max_time": str(train_max),
        "val_min_time": str(val_min),
        "gap_seconds": gap,
        "is_valid": is_valid,
    }
    if not is_valid:
        logger.error(f"LEAKAGE ALERT (Time Split): train extends past val start!")
    else:
        logger.success(f"Time Split OK: {gap:.0f}s gap between train end and val start")
    return result


# ============================================================================
# 4. FEATURE IMPORTANCE AUDIT (Target Leakage via Proxy Features)
# ============================================================================

def feature_importance_audit(
    model,
    feature_names: list[str],
    importance_threshold: float = 0.6,
) -> dict:
    """
    Flag features with disproportionately high importance.
    A single feature explaining >60% of the model → potential leakage.
    """
    if not hasattr(model, "feature_importances_"):
        logger.info("Model has no feature_importances_ (non-tree model). Skipping audit.")
        return {}

    importances = model.feature_importances_
    total = importances.sum()
    relative = importances / (total + 1e-10)
    top_feature = feature_names[np.argmax(relative)]
    top_importance = relative.max()

    flags = {name: float(imp) for name, imp in zip(feature_names, relative)
             if imp > importance_threshold}

    result = {
        "top_feature": top_feature,
        "top_importance": float(top_importance),
        "flagged_features": flags,
        "is_suspicious": top_importance > importance_threshold,
    }
    if result["is_suspicious"]:
        logger.warning(f"LEAKAGE ALERT (Feature Importance): '{top_feature}' "
                       f"has {top_importance:.1%} importance — investigate for leakage!")
    else:
        logger.success(f"Feature Importance OK: top feature '{top_feature}' at {top_importance:.1%}")
    return result


# ============================================================================
# 5. PERFORMANCE SANITY CHECK
# ============================================================================

def performance_sanity_check(
    metric_name: str,
    score: float,
    warning_threshold: float = 0.99,
    task: str = "classification",
) -> dict:
    """
    Flag suspiciously perfect scores.
    >99% accuracy on a multi-class time-series task is almost always leakage.
    """
    is_suspicious = score >= warning_threshold
    result = {
        "metric": metric_name,
        "score": score,
        "warning_threshold": warning_threshold,
        "is_suspicious": is_suspicious,
    }
    if is_suspicious:
        logger.warning(f"SANITY ALERT: {metric_name}={score:.4f} >= {warning_threshold:.2f}. "
                       f"This is suspiciously high — verify for leakage before reporting!")
    else:
        logger.success(f"Sanity Check OK: {metric_name}={score:.4f}")
    return result


# ============================================================================
# 6. BOOTSTRAP VALIDATION (Reproducibility)
# ============================================================================

def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    metric: str = "f1_weighted",
) -> dict:
    """
    Compute bootstrap confidence interval for a classification metric.
    Report mean ± std and CI bounds.
    """
    from functools import partial
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

    metric_fns = {
        "f1_weighted": partial(f1_score, average="weighted"),
        "f1_macro": partial(f1_score, average="macro"),
        "f1_binary": partial(f1_score, average="binary"),
    }
    fn = metric_fns.get(metric, partial(f1_score, average="weighted"))

    scores = []
    n = len(y_true)
    rng = np.random.default_rng(seed=42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            scores.append(fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    result = {
        "metric": metric,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "ci_lower": float(np.quantile(scores, alpha)),
        "ci_upper": float(np.quantile(scores, 1 - alpha)),
        "n_bootstrap": n_bootstrap,
    }
    logger.info(f"Bootstrap {metric}: {result['mean']:.4f} ± {result['std']:.4f} "
                f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}] {int(ci*100)}% CI")
    return result


# ============================================================================
# FULL LEAKAGE REPORT
# ============================================================================

def run_leakage_report(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    metric: str = "f1_weighted",
) -> dict:
    """
    Run leakage checks and return a combined report suitable for MLflow logging.

    X_eval / y_eval: the held-out set used for sanity_check and bootstrap_ci.
    Pass the TRUE test set here when the final model was trained on train+val.
    Falls back to X_val / y_val if not provided (only correct when the model
    was NOT trained on val data).
    """
    logger.info("=" * 60)
    logger.info("LEAKAGE PREVENTION REPORT")
    logger.info("=" * 60)

    x_eval_eff = X_eval if X_eval is not None else X_val
    y_eval_eff = y_eval if y_eval is not None else y_val

    report = {}

    # 1. Label shuffle — retrains a copy on X_train only, evaluates on X_val
    report["label_shuffle"] = label_shuffle_test(
        model, X_train, y_train, X_val, y_val, metric=metric
    )

    # 2. Feature importance
    report["feature_importance"] = feature_importance_audit(model, feature_names)

    # 3. Sanity check + bootstrap on the true eval set
    y_pred_eval = model.predict(x_eval_eff)
    eval_f1 = f1_score(y_eval_eff, y_pred_eval, average="weighted", zero_division=0)
    report["sanity_check"] = performance_sanity_check("eval_f1_weighted", eval_f1)
    report["bootstrap_ci"] = bootstrap_confidence_interval(y_eval_eff, y_pred_eval, metric=metric)

    flags = [k for k, v in report.items() if isinstance(v, dict) and v.get("is_leaking", False)]
    flags += [k for k, v in report.items() if isinstance(v, dict) and v.get("is_suspicious", False)]
    report["leakage_flags"] = flags
    report["is_clean"] = len(flags) == 0

    if report["is_clean"]:
        logger.success("NO LEAKAGE DETECTED — results are credible")
    else:
        logger.error("LEAKAGE FLAGS RAISED: {}", flags)

    return report
