from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError

from sentinel.config import get_settings
from sentinel.database.connection import create_database_engine
from sentinel.database.loader import read_database_tables
from sentinel.database.queries import (
    AnalyticsFilter,
    compare_periods,
    contribution_summary,
    cost_summary,
    europe_churn_investigation,
    monthly_cost_trend,
    monthly_revenue_trend,
    revenue_by_dimension,
    revenue_summary,
    subscription_summary,
    support_summary,
)
from sentinel.eda.cleaning import clean_dataset


@pytest.fixture(scope="module")
def live_engine():
    engine = create_database_engine(get_settings().database_url, connect_timeout=2)
    try:
        with engine.connect():
            pass
    except OperationalError:
        engine.dispose()
        pytest.skip("Local PostgreSQL is unavailable")
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def tables(live_engine):
    return clean_dataset(read_database_tables(live_engine))[0]


def _sales(tables, filters):
    frame = tables["sales"]
    dates = pd.to_datetime(frame.sale_date)
    selected = frame[
        dates.between(pd.Timestamp(filters.start_date), pd.Timestamp(filters.end_date))
    ]
    if filters.region:
        selected = selected[selected.region.eq(filters.region)]
    if filters.plan_type:
        selected = selected[selected.plan_type.eq(filters.plan_type)]
    return selected


def _costs(tables, filters):
    frame = tables["cloud_costs"]
    dates = pd.to_datetime(frame.date)
    selected = frame[
        dates.between(pd.Timestamp(filters.start_date), pd.Timestamp(filters.end_date))
    ]
    if filters.region:
        selected = selected[selected.region.eq(filters.region)]
    return selected


@pytest.mark.parametrize(
    "filters",
    [
        AnalyticsFilter("2023-01-01", "2024-12-31"),
        AnalyticsFilter("2024-04-01", "2024-06-30", region="Europe"),
        AnalyticsFilter("2024-01-01", "2024-03-31", plan_type="Enterprise"),
    ],
)
def test_revenue_summary_reconciles_with_pandas(live_engine, tables, filters):
    actual = revenue_summary(live_engine, filters)
    expected = _sales(tables, filters)
    assert actual["gross_revenue"] == pytest.approx(float(expected.gross_revenue.sum()))
    assert actual["discount_amount"] == pytest.approx(float(expected.discount_amount.sum()))
    assert actual["net_revenue"] == pytest.approx(float(expected.net_revenue.sum()))
    assert actual["revenue_customers"] == expected.customer_id.nunique()
    assert actual["sales_observations"] == len(expected)


def test_cost_and_contribution_reconcile_with_pandas(live_engine, tables):
    filters = AnalyticsFilter("2024-04-01", "2024-06-30", region="Europe")
    revenue = _sales(tables, filters)
    costs = _costs(tables, filters)
    actual_costs = cost_summary(live_engine, filters)
    actual = contribution_summary(live_engine, filters)
    assert actual_costs["gpu_cost"] == pytest.approx(float(costs.gpu_cost.sum()))
    assert actual_costs["total_cost"] == pytest.approx(float(costs.total_cost.sum()))
    assert actual["estimated_gross_contribution"] == pytest.approx(
        float(revenue.net_revenue.sum() - costs.total_cost.sum())
    )


def test_subscription_summary_reconciles_with_pandas(live_engine, tables):
    filters = AnalyticsFilter("2024-04-01", "2024-06-30", region="Europe")
    frame = tables["subscriptions"].merge(
        tables["customers"][["customer_id", "region"]], on="customer_id", how="left"
    )
    frame = frame[frame.region.eq("Europe")]
    start = pd.to_datetime(frame.start_date)
    end = pd.to_datetime(frame.end_date)
    renewal = pd.to_datetime(frame.renewal_date)
    at_risk = (start <= pd.Timestamp(filters.end_date)) & (
        end.isna() | (end >= pd.Timestamp(filters.start_date))
    )
    cancelled = frame.subscription_status.eq("cancelled") & end.between(
        pd.Timestamp(filters.start_date), pd.Timestamp(filters.end_date)
    )
    due = renewal.between(pd.Timestamp(filters.start_date), pd.Timestamp(filters.end_date))
    renewed = due & (end.isna() | (end >= renewal))
    actual = subscription_summary(live_engine, filters)
    assert actual["subscriptions_at_risk"] == int(at_risk.sum())
    assert actual["cancellations"] == int(cancelled.sum())
    assert actual["renewals_due"] == int(due.sum())
    assert actual["renewals_retained"] == int(renewed.sum())


def test_support_summary_reconciles_with_cleaned_pandas(live_engine, tables):
    filters = AnalyticsFilter("2024-04-01", "2024-06-30", region="Europe")
    frame = tables["support_tickets"].merge(
        tables["customers"][["customer_id", "region"]], on="customer_id", how="left"
    )
    dates = pd.to_datetime(frame.created_date)
    expected = frame[
        dates.between(pd.Timestamp(filters.start_date), pd.Timestamp(filters.end_date))
        & frame.region.eq("Europe")
    ]
    actual = support_summary(live_engine, filters)
    assert actual["ticket_count"] == len(expected)
    assert actual["customers_with_tickets"] == expected.customer_id.nunique()
    assert actual["average_resolution_hours"] == pytest.approx(
        float(expected.resolution_hours.mean())
    )
    assert actual["average_satisfaction"] == pytest.approx(
        float(expected.satisfaction_score.mean())
    )


def test_monthly_trends_reconcile_with_pandas(live_engine, tables):
    filters = AnalyticsFilter("2024-01-01", "2024-12-31")
    revenue = pd.DataFrame(monthly_revenue_trend(live_engine, filters))
    costs = pd.DataFrame(monthly_cost_trend(live_engine, filters))
    expected_revenue = _sales(tables, filters).copy()
    expected_revenue["month"] = pd.to_datetime(expected_revenue.sale_date).dt.to_period("M")
    expected_costs = _costs(tables, filters).copy()
    expected_costs["month"] = pd.to_datetime(expected_costs.date).dt.to_period("M")
    assert len(revenue) == 12
    assert len(costs) == 12
    assert revenue.net_revenue.sum() == pytest.approx(
        float(expected_revenue.net_revenue.sum())
    )
    assert costs.total_cost.sum() == pytest.approx(float(expected_costs.total_cost.sum()))


def test_dimensions_and_demo_investigations(live_engine):
    whole_period = AnalyticsFilter("2023-01-01", "2024-12-31")
    regions = revenue_by_dimension(live_engine, whole_period, "region")
    plans = revenue_by_dimension(live_engine, whole_period, "plan_type")
    assert {row["region"] for row in regions} == {"India", "North America", "Europe", "APAC"}
    assert {row["plan_type"] for row in plans} == {"Basic", "Professional", "Enterprise"}

    q1 = AnalyticsFilter("2024-01-01", "2024-03-31")
    q2 = AnalyticsFilter("2024-04-01", "2024-06-30")
    comparison = compare_periods(live_engine, q1, q2)
    assert comparison["changes"]["net_revenue_growth"] > 0
    assert comparison["changes"]["estimated_gross_contribution_growth"] < 0

    europe = europe_churn_investigation(
        live_engine,
        AnalyticsFilter(q1.start_date, q1.end_date, region="Europe"),
        AnalyticsFilter(q2.start_date, q2.end_date, region="Europe"),
    )
    assert (
        europe["second_period"]["support"]["average_resolution_hours"]
        > europe["first_period"]["support"]["average_resolution_hours"]
    )
    assert "does not establish causation" in europe["interpretation_boundary"]


def test_filter_validation_and_unsupported_allocations(live_engine):
    assert AnalyticsFilter("2024-01-01", "2024-01-31").start_date == date(2024, 1, 1)
    with pytest.raises(ValueError):
        AnalyticsFilter("2024-02-01", "2024-01-01")
    with pytest.raises(ValueError):
        AnalyticsFilter("2024-01-01", "2024-01-31", region="Mars")
    with pytest.raises(ValueError):
        AnalyticsFilter("2024-01-01", "2024-01-31", plan_type="Unlimited")
    plan_filter = AnalyticsFilter("2024-01-01", "2024-01-31", plan_type="Basic")
    with pytest.raises(ValueError, match="not allocated by plan"):
        cost_summary(live_engine, plan_filter)
    with pytest.raises(ValueError, match="cannot be safely allocated by plan"):
        support_summary(live_engine, plan_filter)
