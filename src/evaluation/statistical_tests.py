"""
Statistical significance testing for model comparison.
All claims must be backed by at least one test below.

Protocol:
  1. Wilcoxon signed-rank test (paired, non-parametric) — primary
  2. Two-way ANOVA — model × dataset interaction
  3. Effect size (Cohen's d) alongside p-value
  4. Report as: mean ± std (95% CI) [p=X.XXX, d=X.XX]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


# ============================================================================
# WILCOXON SIGNED-RANK TEST (paired models)
# ============================================================================

def wilcoxon_test(
    scores_a: list[float],
    scores_b: list[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    alpha: float = 0.05,
) -> dict:
    """
    Paired Wilcoxon signed-rank test.
    Use when comparing two models on the same CV folds.

    scores_a, scores_b: list of per-fold metric values (same length)
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must be same length (paired folds)")

    stat, p_value = stats.wilcoxon(scores_a, scores_b)
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    better = model_a_name if mean_a > mean_b else model_b_name

    result = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "better_model": better,
    }

    if result["significant"]:
        logger.success(f"Wilcoxon: {better} is significantly better "
                       f"(p={p_value:.4f} < {alpha}): {mean_a:.4f} vs {mean_b:.4f}")
    else:
        logger.info(f"Wilcoxon: no significant difference (p={p_value:.4f} >= {alpha}): "
                    f"{mean_a:.4f} vs {mean_b:.4f}")
    return result


# ============================================================================
# COHEN'S D (effect size)
# ============================================================================

def cohen_d(scores_a: list[float], scores_b: list[float]) -> float:
    """
    Compute Cohen's d effect size for two score lists.
    Interpretation: 0.2=small, 0.5=medium, 0.8=large
    """
    a, b = np.array(scores_a), np.array(scores_b)
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


# ============================================================================
# TWO-WAY ANOVA (model × dataset)
# ============================================================================

def two_way_anova(results_df: pd.DataFrame) -> dict:
    """
    Two-way ANOVA: tests if model improvements are consistent across datasets.

    results_df must have columns:
      - 'model': model name (factor A)
      - 'dataset': dataset name (factor B)
      - 'score': metric value (dependent variable)

    Returns p-values for model effect, dataset effect, and interaction.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    model = ols("score ~ C(model) + C(dataset) + C(model):C(dataset)", data=results_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    result = {
        "anova_table": anova_table.to_dict(),
        "model_p_value": float(anova_table.loc["C(model)", "PR(>F)"]),
        "dataset_p_value": float(anova_table.loc["C(dataset)", "PR(>F)"]),
        "interaction_p_value": float(anova_table.loc["C(model):C(dataset)", "PR(>F)"]),
    }

    logger.info(f"Two-way ANOVA:")
    logger.info(f"  Model effect:       p={result['model_p_value']:.4f}")
    logger.info(f"  Dataset effect:     p={result['dataset_p_value']:.4f}")
    logger.info(f"  Interaction effect: p={result['interaction_p_value']:.4f}")

    if result["interaction_p_value"] < 0.05:
        logger.warning("Significant interaction: model improvements are NOT consistent "
                       "across datasets. Investigate per-dataset results.")
    else:
        logger.success("No significant interaction: model improvements are consistent "
                       "across datasets.")
    return result


# ============================================================================
# SUMMARY REPORT (format for thesis / paper)
# ============================================================================

def format_metric_report(
    scores: list[float],
    metric_name: str = "F1",
    n_bootstrap: int = 1000,
) -> str:
    """
    Format a metric as: 'mean ± std (95% CI: [lo, hi])'
    Ready to paste into thesis or paper table.
    """
    arr = np.array(scores)
    mean = arr.mean()
    std = arr.std()

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    ci_lo = np.quantile(boot, 0.025)
    ci_hi = np.quantile(boot, 0.975)

    return f"{metric_name}: {mean:.4f} ± {std:.4f} (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])"
