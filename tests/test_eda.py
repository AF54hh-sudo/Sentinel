import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.generator import AMXTechDataGenerator, GenerationConfig
from sentinel.eda.cleaning import clean_dataset
from sentinel.eda.exploration import (
    business_metrics,
    customer_behavior_correlations,
    dimension_metrics,
    monthly_churn_trend,
    monthly_costs,
    monthly_revenue,
    monthly_ticket_trend,
)
from sentinel.eda.summary import (
    categorical_summary,
    dataset_summary,
    iqr_outlier_summary,
    missing_value_summary,
    numerical_summary,
)


@pytest.fixture(scope="module")
def raw_tables():
    return AMXTechDataGenerator(
        GenerationConfig(n_customers=200, n_subscriptions=300, n_tickets=400)
    ).generate()


@pytest.fixture(scope="module")
def cleaned_tables(raw_tables):
    return clean_dataset(raw_tables)[0]


def test_cleaning_policy_is_explicit_and_non_mutating(raw_tables):
    original_customers = raw_tables["customers"].copy(deep=True)
    original_tickets = raw_tables["support_tickets"].copy(deep=True)
    cleaned, report = clean_dataset(raw_tables)
    pd.testing.assert_frame_equal(raw_tables["customers"], original_customers)
    pd.testing.assert_frame_equal(raw_tables["support_tickets"], original_tickets)
    assert cleaned["customers"].industry.isna().sum() == 0
    assert report.industries_filled == original_customers.industry.isna().sum()
    assert report.duplicate_events_removed == 3
    assert len(cleaned["support_tickets"]) == len(original_tickets) - 3
    assert report.satisfaction_values_retained_missing > 0
    assert report.validation and report.validation.is_valid


def test_summary_functions_cover_required_profile(cleaned_tables):
    sales = cleaned_tables["sales"]
    summary = dataset_summary(sales)
    assert summary["rows"] == len(sales)
    assert summary["columns"] == len(sales.columns)
    assert "net_revenue" in numerical_summary(sales).index
    assert "plan_type" in categorical_summary(sales)
    assert set(missing_value_summary(sales)) == {"missing_count", "missing_percent"}
    assert "net_revenue" in iqr_outlier_summary(sales).index


def test_business_metrics_reconcile(cleaned_tables):
    metrics = business_metrics(cleaned_tables)
    assert metrics["net_revenue"] == pytest.approx(
        cleaned_tables["sales"].net_revenue.sum()
    )
    assert metrics["estimated_gross_contribution"] == pytest.approx(
        metrics["net_revenue"] - metrics["infrastructure_cost"]
    )
    assert 0 <= metrics["churn_rate"] <= 1
    assert 0 <= metrics["renewal_rate"] <= 1


def test_trends_and_dimensions_are_ordered_and_finite(cleaned_tables):
    monthly_frames = [
        monthly_revenue(cleaned_tables["sales"]),
        monthly_costs(cleaned_tables["cloud_costs"]),
        monthly_ticket_trend(cleaned_tables["support_tickets"]),
        monthly_churn_trend(cleaned_tables["subscriptions"]),
    ]
    for frame in monthly_frames:
        assert frame.month.is_monotonic_increasing
        assert frame.month.is_unique
    for dimension in ("region", "plan_type"):
        result = dimension_metrics(cleaned_tables, dimension)
        assert result[dimension].notna().all()
        assert result.churn_rate.between(0, 1).all()


def test_customer_correlations_are_valid(cleaned_tables):
    correlations = customer_behavior_correlations(cleaned_tables)
    assert np.allclose(np.diag(correlations), 1)
    assert correlations.equals(correlations.T)
    assert correlations.loc["average_resolution_hours", "average_satisfaction"] < 0


def test_phase4_notebooks_are_valid_and_nonempty():
    for name in ("01_data_validation.ipynb", "02_eda.ipynb"):
        notebook = json.loads((Path("notebooks") / name).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
