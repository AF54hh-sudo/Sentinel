import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.generator import AMXTechDataGenerator, GenerationConfig
from sentinel.eda.cleaning import clean_dataset
from sentinel.statistics.business_analysis import (
    pre_outcome_customer_support,
    run_business_statistics,
)
from sentinel.statistics.correlations import correlation_test
from sentinel.statistics.hypothesis_tests import (
    chi_square_test,
    descriptive_statistics,
    mann_whitney_test,
    mean_confidence_interval,
    proportion_confidence_interval,
    welch_t_test,
)


def test_descriptive_statistics_known_values():
    result = descriptive_statistics([1, 2, 3, 4, 5])
    assert result["n"] == 5
    assert result["mean"] == 3
    assert result["median"] == 3
    assert result["std"] == pytest.approx(np.sqrt(2.5))


def test_mean_confidence_interval_contains_known_mean_and_shrinks():
    narrow = mean_confidence_interval(np.arange(1, 101))
    wide = mean_confidence_interval(np.arange(1, 11))
    assert narrow["lower"] < 50.5 < narrow["upper"]
    assert (narrow["upper"] - narrow["lower"]) < (wide["upper"] - wide["lower"]) * 4


def test_wilson_proportion_interval_known_example():
    result = proportion_confidence_interval(50, 100)
    assert result["proportion"] == 0.5
    assert result["lower"] == pytest.approx(0.4038, abs=0.001)
    assert result["upper"] == pytest.approx(0.5962, abs=0.001)


def test_welch_report_has_complete_contract_and_detects_difference():
    first = [10, 11, 12, 13, 14, 15]
    second = [1, 2, 3, 4, 5, 6]
    report = welch_t_test(
        first,
        second,
        first_label="high",
        second_label="low",
        business_question="Are group means different?",
        null_hypothesis="Means are equal.",
        alternative_hypothesis="Means differ.",
    )
    values = report.to_dict()
    required = {
        "business_question",
        "null_hypothesis",
        "alternative_hypothesis",
        "test_used",
        "reason_for_test",
        "sample_size",
        "statistic",
        "p_value",
        "effect_size_name",
        "effect_size",
        "result",
        "business_interpretation",
    }
    assert required.issubset(values)
    assert report.p_value < 0.05
    assert report.effect_size > 0
    assert report.result == "Reject the null hypothesis"


def test_mann_whitney_effect_direction():
    report = mann_whitney_test(
        [8, 9, 10, 11],
        [1, 2, 3, 4],
        first_label="high",
        second_label="low",
        business_question="Are distributions different?",
        null_hypothesis="Distributions are equal.",
        alternative_hypothesis="Distributions differ.",
    )
    assert report.effect_size > 0
    assert report.effect_size_name == "rank-biserial correlation"


def test_chi_square_known_association_and_effect_size():
    table = pd.DataFrame([[90, 10], [20, 80]], index=["A", "B"], columns=["no", "yes"])
    report = chi_square_test(
        table,
        business_question="Are variables associated?",
        row_variable="group",
        column_variable="outcome",
    )
    assert report.p_value < 0.001
    assert 0 < report.effect_size <= 1
    assert report.effect_size_name == "Cramér's V"


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlation_known_relationship(method):
    report = correlation_test(
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
        first_label="x",
        second_label="y",
        method=method,
    )
    assert report.effect_size == pytest.approx(1)
    assert report.p_value < 0.01
    assert "not evidence of causation" in report.business_interpretation


def test_invalid_statistical_inputs_are_rejected():
    with pytest.raises(ValueError):
        mean_confidence_interval([1])
    with pytest.raises(ValueError):
        proportion_confidence_interval(11, 10)
    with pytest.raises(ValueError):
        chi_square_test(pd.DataFrame([[1, 2]]), business_question="x", row_variable="a", column_variable="b")
    with pytest.raises(ValueError):
        correlation_test([1, 1, 1], [1, 2, 3], first_label="x", second_label="y")


@pytest.fixture(scope="module")
def small_tables():
    raw = AMXTechDataGenerator(
        GenerationConfig(n_customers=500, n_subscriptions=800, n_tickets=1_000)
    ).generate()
    return clean_dataset(raw)[0]


def test_customer_support_frame_has_one_row_per_customer_and_censors_outcomes(small_tables):
    frame = pre_outcome_customer_support(small_tables)
    assert len(frame) == len(small_tables["customers"])
    assert frame.customer_id.is_unique
    assert frame.churned.isin([0, 1]).all()
    assert (frame.ticket_count >= 0).all()


def test_business_analysis_emits_all_required_tests(small_tables):
    report = run_business_statistics(small_tables)
    assert set(report["tests"]) == {
        "support_resolution_welch",
        "support_resolution_mann_whitney",
        "support_satisfaction_spearman",
        "support_satisfaction_pearson_sensitivity",
        "plan_cancellation_chi_square",
    }
    for test in report["tests"].values():
        assert test["reason_for_test"]
        assert test["effect_size_name"]
        assert test["business_interpretation"]
        assert test["assumptions"]
    assert report["assumption_diagnostics"]["support_analysis_unique_customers"]
    assert report["assumption_diagnostics"]["primary_correlation_method"] == "Spearman"


def test_statistics_notebook_is_valid_and_nonempty():
    notebook = json.loads(Path("notebooks/03_statistics.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
