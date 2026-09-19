"""Statistical inference with effect sizes and business-readable reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


@dataclass(frozen=True)
class StatisticalTestReport:
    business_question: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_used: str
    reason_for_test: str
    sample_size: dict[str, int]
    statistic: float
    p_value: float
    effect_size_name: str
    effect_size: float
    alpha: float
    result: str
    business_interpretation: str
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric(values: pd.Series | np.ndarray | list[float], name: str) -> np.ndarray:
    cleaned = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if cleaned.size < 2:
        raise ValueError(f"{name} must contain at least two finite observations")
    return cleaned


def descriptive_statistics(values: pd.Series | np.ndarray | list[float]) -> dict[str, float | int]:
    sample = _numeric(values, "values")
    return {
        "n": int(sample.size),
        "mean": float(np.mean(sample)),
        "median": float(np.median(sample)),
        "std": float(np.std(sample, ddof=1)),
        "min": float(np.min(sample)),
        "p25": float(np.quantile(sample, 0.25)),
        "p75": float(np.quantile(sample, 0.75)),
        "max": float(np.max(sample)),
        "skewness": float(stats.skew(sample, bias=False)),
    }


def mean_confidence_interval(
    values: pd.Series | np.ndarray | list[float], confidence: float = 0.95
) -> dict[str, float | int]:
    sample = _numeric(values, "values")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    mean = float(np.mean(sample))
    standard_error = float(stats.sem(sample))
    critical = float(stats.t.ppf((1 + confidence) / 2, df=sample.size - 1))
    margin = critical * standard_error
    return {
        "n": int(sample.size),
        "mean": mean,
        "confidence": confidence,
        "lower": mean - margin,
        "upper": mean + margin,
        "standard_error": standard_error,
    }


def proportion_confidence_interval(
    successes: int, observations: int, confidence: float = 0.95
) -> dict[str, float | int]:
    if observations <= 0 or not 0 <= successes <= observations:
        raise ValueError("successes must be between zero and positive observations")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    lower, upper = proportion_confint(
        successes,
        observations,
        alpha=1 - confidence,
        method="wilson",
    )
    return {
        "successes": successes,
        "observations": observations,
        "proportion": successes / observations,
        "confidence": confidence,
        "lower": float(lower),
        "upper": float(upper),
        "method": "Wilson score",
    }


def _hedges_g(first: np.ndarray, second: np.ndarray) -> float:
    degrees_freedom = first.size + second.size - 2
    pooled_variance = (
        (first.size - 1) * np.var(first, ddof=1)
        + (second.size - 1) * np.var(second, ddof=1)
    ) / degrees_freedom
    if pooled_variance <= 0:
        return 0.0
    cohens_d = (np.mean(first) - np.mean(second)) / sqrt(pooled_variance)
    correction = 1 - 3 / (4 * degrees_freedom - 1)
    return float(cohens_d * correction)


def welch_t_test(
    first: pd.Series | np.ndarray | list[float],
    second: pd.Series | np.ndarray | list[float],
    *,
    first_label: str,
    second_label: str,
    business_question: str,
    null_hypothesis: str,
    alternative_hypothesis: str,
    alpha: float = 0.05,
    interpretation_template: str | None = None,
    limitations: list[str] | None = None,
) -> StatisticalTestReport:
    first_values = _numeric(first, first_label)
    second_values = _numeric(second, second_label)
    statistic, p_value = stats.ttest_ind(
        first_values, second_values, equal_var=False, nan_policy="omit"
    )
    effect = _hedges_g(first_values, second_values)
    significant = bool(p_value < alpha)
    default_interpretation = (
        f"{first_label} averaged {np.mean(first_values):.2f}, compared with "
        f"{np.mean(second_values):.2f} for {second_label}. The difference "
        f"{'is statistically distinguishable' if significant else 'is not statistically distinguishable'} "
        f"at alpha={alpha:.2f}; Hedges' g={effect:.3f}."
    )
    return StatisticalTestReport(
        business_question=business_question,
        null_hypothesis=null_hypothesis,
        alternative_hypothesis=alternative_hypothesis,
        test_used="Welch's independent-samples t-test",
        reason_for_test=(
            "Compares two independent group means without assuming equal variances; "
            "the mean is the business quantity of interest."
        ),
        sample_size={first_label: int(first_values.size), second_label: int(second_values.size)},
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size_name="Hedges' g",
        effect_size=effect,
        alpha=alpha,
        result="Reject the null hypothesis" if significant else "Fail to reject the null hypothesis",
        business_interpretation=interpretation_template or default_interpretation,
        assumptions=[
            "Groups are independent at the selected unit of analysis.",
            "Observations are numeric and the sample means are estimable.",
            "Welch's test does not require equal group variances.",
        ],
        limitations=list(limitations or []),
    )


def mann_whitney_test(
    first: pd.Series | np.ndarray | list[float],
    second: pd.Series | np.ndarray | list[float],
    *,
    first_label: str,
    second_label: str,
    business_question: str,
    null_hypothesis: str,
    alternative_hypothesis: str,
    alpha: float = 0.05,
    limitations: list[str] | None = None,
) -> StatisticalTestReport:
    first_values = _numeric(first, first_label)
    second_values = _numeric(second, second_label)
    statistic, p_value = stats.mannwhitneyu(
        first_values, second_values, alternative="two-sided", method="auto"
    )
    effect = float(2 * statistic / (first_values.size * second_values.size) - 1)
    significant = bool(p_value < alpha)
    return StatisticalTestReport(
        business_question=business_question,
        null_hypothesis=null_hypothesis,
        alternative_hypothesis=alternative_hypothesis,
        test_used="Mann–Whitney U test (two-sided)",
        reason_for_test=(
            "Nonparametric comparison of two independent distributions, used because support "
            "resolution is right-skewed and may contain influential high values."
        ),
        sample_size={first_label: int(first_values.size), second_label: int(second_values.size)},
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size_name="rank-biserial correlation",
        effect_size=effect,
        alpha=alpha,
        result="Reject the null hypothesis" if significant else "Fail to reject the null hypothesis",
        business_interpretation=(
            f"The {first_label} median is {np.median(first_values):.2f}, versus "
            f"{np.median(second_values):.2f} for {second_label}. The distributional difference "
            f"{'is statistically distinguishable' if significant else 'is not statistically distinguishable'}; "
            f"rank-biserial correlation={effect:.3f}."
        ),
        assumptions=[
            "Groups and customer-level observations are independent.",
            "The response is at least ordinal.",
            "A location-shift interpretation requires similarly shaped distributions.",
        ],
        limitations=list(limitations or []),
    )


def chi_square_test(
    contingency: pd.DataFrame,
    *,
    business_question: str,
    row_variable: str,
    column_variable: str,
    alpha: float = 0.05,
    limitations: list[str] | None = None,
) -> StatisticalTestReport:
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise ValueError("contingency table must have at least two rows and two columns")
    values = contingency.to_numpy(dtype=float)
    if (values < 0).any() or not np.allclose(values, np.round(values)):
        raise ValueError("contingency counts must be non-negative integers")
    statistic, p_value, _, expected = stats.chi2_contingency(values, correction=False)
    total = values.sum()
    denominator = min(values.shape[0] - 1, values.shape[1] - 1)
    effect = float(sqrt(statistic / (total * denominator))) if total and denominator else 0.0
    significant = bool(p_value < alpha)
    low_expected = int((expected < 5).sum())
    return StatisticalTestReport(
        business_question=business_question,
        null_hypothesis=f"{row_variable} and {column_variable} are independent.",
        alternative_hypothesis=f"{row_variable} and {column_variable} are associated.",
        test_used="Pearson chi-square test of independence",
        reason_for_test="Both variables are categorical and observations form frequency counts.",
        sample_size={"observations": int(total), "cells": int(values.size)},
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size_name="Cramér's V",
        effect_size=effect,
        alpha=alpha,
        result="Reject the null hypothesis" if significant else "Fail to reject the null hypothesis",
        business_interpretation=(
            f"{row_variable} and {column_variable} "
            f"{'show a statistically detectable association' if significant else 'do not show a statistically detectable association'}; "
            f"Cramér's V={effect:.3f}."
        ),
        assumptions=[
            "Each observation contributes to one contingency-table cell.",
            "Expected cell counts should generally be at least five.",
            f"Cells with expected count below five: {low_expected}.",
        ],
        limitations=list(limitations or []),
    )
