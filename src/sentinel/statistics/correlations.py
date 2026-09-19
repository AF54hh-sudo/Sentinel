"""Pearson and Spearman correlation reports with inference boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from sentinel.statistics.hypothesis_tests import StatisticalTestReport


def correlation_test(
    first: pd.Series | np.ndarray | list[float],
    second: pd.Series | np.ndarray | list[float],
    *,
    first_label: str,
    second_label: str,
    method: str = "spearman",
    alpha: float = 0.05,
    business_question: str | None = None,
    limitations: list[str] | None = None,
) -> StatisticalTestReport:
    paired = pd.DataFrame({first_label: first, second_label: second}).apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(paired) < 3:
        raise ValueError("correlation requires at least three complete pairs")
    if paired[first_label].nunique() < 2 or paired[second_label].nunique() < 2:
        raise ValueError("correlation requires variation in both variables")
    if method == "pearson":
        result = stats.pearsonr(paired[first_label], paired[second_label])
        test_name = "Pearson product-moment correlation"
        effect_name = "Pearson r"
        reason = "Measures linear association between two numeric variables."
        assumptions = [
            "Pairs are independent.",
            "The relationship is approximately linear.",
            "Inference is sensitive to influential outliers and assumes bivariate normality.",
        ]
    elif method == "spearman":
        result = stats.spearmanr(paired[first_label], paired[second_label])
        test_name = "Spearman rank correlation"
        effect_name = "Spearman rho"
        reason = (
            "Measures monotonic association without requiring normally distributed values; "
            "appropriate for skewed support metrics and bounded satisfaction scores."
        )
        assumptions = [
            "Pairs are independent.",
            "Variables are at least ordinal.",
            "The relationship is interpreted as monotonic, not necessarily linear.",
        ]
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    coefficient, p_value = float(result.statistic), float(result.pvalue)
    significant = p_value < alpha
    direction = "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero"
    return StatisticalTestReport(
        business_question=business_question
        or f"Are {first_label} and {second_label} associated?",
        null_hypothesis=f"There is no {method} association between {first_label} and {second_label}.",
        alternative_hypothesis=f"There is a non-zero {method} association between {first_label} and {second_label}.",
        test_used=test_name,
        reason_for_test=reason,
        sample_size={"complete_pairs": len(paired)},
        statistic=coefficient,
        p_value=p_value,
        effect_size_name=effect_name,
        effect_size=coefficient,
        alpha=alpha,
        result="Reject the null hypothesis" if significant else "Fail to reject the null hypothesis",
        business_interpretation=(
            f"The estimated association is {direction} ({effect_name}={coefficient:.3f}) and "
            f"{'statistically distinguishable from zero' if significant else 'not statistically distinguishable from zero'} "
            f"at alpha={alpha:.2f}. This is not evidence of causation."
        ),
        assumptions=assumptions,
        limitations=list(limitations or []) + ["Correlation alone cannot establish causation."],
    )
