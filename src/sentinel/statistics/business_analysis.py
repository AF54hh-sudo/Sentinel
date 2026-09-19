"""AMX Tech Phase 6 analyses assembled from cleaned business tables."""

from __future__ import annotations

from typing import Any

import pandas as pd

from sentinel.statistics.correlations import correlation_test
from sentinel.statistics.hypothesis_tests import (
    chi_square_test,
    descriptive_statistics,
    mann_whitney_test,
    mean_confidence_interval,
    proportion_confidence_interval,
    welch_t_test,
)


def pre_outcome_customer_support(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create one independent row per customer using support known by outcome time.

    For customers with a cancellation, support events are censored at their first cancellation
    date. Other customers are observed through the dataset end. Customers without a support
    event in their observation window are retained with ticket_count=0 and missing averages.
    """
    subscriptions = tables["subscriptions"].copy()
    subscriptions["end_date"] = pd.to_datetime(subscriptions.end_date)
    cancellation_dates = (
        subscriptions[subscriptions.subscription_status.eq("cancelled")]
        .groupby("customer_id", as_index=False)
        .agg(first_cancellation_date=("end_date", "min"))
    )
    customers = tables["customers"][["customer_id", "region"]].merge(
        cancellation_dates, on="customer_id", how="left"
    )
    observation_end = max(
        pd.to_datetime(tables["sales"].sale_date).max(),
        pd.to_datetime(tables["support_tickets"].created_date).max(),
    )
    customers["outcome_date"] = customers.first_cancellation_date.fillna(observation_end)
    customers["churned"] = customers.first_cancellation_date.notna().astype(int)

    tickets = tables["support_tickets"].copy()
    tickets["created_date"] = pd.to_datetime(tickets.created_date)
    observed = tickets.merge(
        customers[["customer_id", "outcome_date"]], on="customer_id", how="inner"
    )
    observed = observed[observed.created_date <= observed.outcome_date]
    support = observed.groupby("customer_id", as_index=False).agg(
        ticket_count=("ticket_id", "count"),
        average_resolution_hours=("resolution_hours", "mean"),
        average_satisfaction=("satisfaction_score", "mean"),
    )
    result = customers.merge(support, on="customer_id", how="left")
    result["ticket_count"] = result.ticket_count.fillna(0).astype(int)
    return result


def run_business_statistics(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    customer_support = pre_outcome_customer_support(tables)
    supported = customer_support.dropna(subset=["average_resolution_hours"])
    churned = supported[supported.churned.eq(1)]
    retained = supported[supported.churned.eq(0)]
    limitations = [
        "This synthetic observational comparison demonstrates association, not causation.",
        "Customers without support interactions are absent from resolution-time comparisons.",
        "Support observation ends at first cancellation for churned customers and dataset end for others.",
    ]

    welch = welch_t_test(
        churned.average_resolution_hours,
        retained.average_resolution_hours,
        first_label="churned customers",
        second_label="retained customers",
        business_question=(
            "Do customers who later cancel have a different mean pre-outcome support resolution time?"
        ),
        null_hypothesis="Mean pre-outcome resolution time is equal for churned and retained customers.",
        alternative_hypothesis="Mean pre-outcome resolution time differs between churned and retained customers.",
        limitations=limitations,
    )
    mann_whitney = mann_whitney_test(
        churned.average_resolution_hours,
        retained.average_resolution_hours,
        first_label="churned customers",
        second_label="retained customers",
        business_question=(
            "Do pre-outcome support-resolution distributions differ for churned and retained customers?"
        ),
        null_hypothesis=(
            "Pre-outcome resolution-time distributions are the same for churned and retained customers."
        ),
        alternative_hypothesis=(
            "Pre-outcome resolution-time distributions differ for churned and retained customers."
        ),
        limitations=limitations,
    )
    support_correlation = correlation_test(
        supported.average_resolution_hours,
        supported.average_satisfaction,
        first_label="average_resolution_hours",
        second_label="average_satisfaction",
        method="spearman",
        business_question=(
            "Is customer-level support resolution time monotonically associated with satisfaction?"
        ),
        limitations=[
            "Only customers with non-missing satisfaction and a support event contribute complete pairs."
        ],
    )
    support_pearson_sensitivity = correlation_test(
        supported.average_resolution_hours,
        supported.average_satisfaction,
        first_label="average_resolution_hours",
        second_label="average_satisfaction",
        method="pearson",
        business_question=(
            "As a sensitivity analysis, is customer-level resolution time linearly associated with satisfaction?"
        ),
        limitations=[
            "This is secondary to Spearman because resolution is right-skewed and satisfaction is bounded.",
            "Only complete customer-level pairs contribute to the calculation.",
        ],
    )

    subscriptions = tables["subscriptions"].copy()
    subscriptions["cancelled"] = subscriptions.subscription_status.eq("cancelled")
    contingency = pd.crosstab(subscriptions.plan_type, subscriptions.cancelled)
    contingency = contingency.reindex(columns=[False, True], fill_value=0)
    contingency.columns = ["not_cancelled", "cancelled"]
    plan_test = chi_square_test(
        contingency,
        business_question="Is AMX Tech subscription cancellation associated with plan type?",
        row_variable="plan type",
        column_variable="cancellation status",
        limitations=[
            "Some customers hold multiple subscriptions, so strict observation independence may be imperfect.",
            "The test does not adjust for region, tenure, support quality, or company size.",
        ],
    )

    europe = subscriptions.merge(
        tables["customers"][["customer_id", "region"]], on="customer_id", how="left"
    )
    europe = europe[europe.region.eq("Europe")]
    europe["start_date"] = pd.to_datetime(europe.start_date)
    europe["end_date"] = pd.to_datetime(europe.end_date)
    period_intervals = {
        "q1_2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31")),
        "q2_2024": (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30")),
    }
    churn_intervals: dict[str, dict[str, float | int | str]] = {}
    for label, (start, end) in period_intervals.items():
        at_risk = (europe.start_date <= end) & (
            europe.end_date.isna() | (europe.end_date >= start)
        )
        cancelled = (
            europe.subscription_status.eq("cancelled")
            & europe.end_date.between(start, end)
        )
        churn_intervals[label] = proportion_confidence_interval(
            int(cancelled.sum()), int(at_risk.sum())
        )

    return {
        "analysis_unit": "customer for support analyses; subscription for plan analysis",
        "descriptive_statistics": {
            "churned_customer_resolution": descriptive_statistics(
                churned.average_resolution_hours
            ),
            "retained_customer_resolution": descriptive_statistics(
                retained.average_resolution_hours
            ),
        },
        "assumption_diagnostics": {
            "support_analysis_unique_customers": bool(customer_support.customer_id.is_unique),
            "churned_resolution_skewness": descriptive_statistics(
                churned.average_resolution_hours
            )["skewness"],
            "retained_resolution_skewness": descriptive_statistics(
                retained.average_resolution_hours
            )["skewness"],
            "variance_ratio_larger_to_smaller": float(
                max(churned.average_resolution_hours.var(), retained.average_resolution_hours.var())
                / min(churned.average_resolution_hours.var(), retained.average_resolution_hours.var())
            ),
            "primary_correlation_method": "Spearman",
            "pearson_role": "secondary linear sensitivity analysis",
        },
        "confidence_intervals": {
            "churned_mean_resolution": mean_confidence_interval(
                churned.average_resolution_hours
            ),
            "retained_mean_resolution": mean_confidence_interval(
                retained.average_resolution_hours
            ),
            "europe_churn": churn_intervals,
        },
        "tests": {
            "support_resolution_welch": welch.to_dict(),
            "support_resolution_mann_whitney": mann_whitney.to_dict(),
            "support_satisfaction_spearman": support_correlation.to_dict(),
            "support_satisfaction_pearson_sensitivity": (
                support_pearson_sensitivity.to_dict()
            ),
            "plan_cancellation_chi_square": plan_test.to_dict(),
        },
        "contingency_table": contingency.to_dict(orient="index"),
    }
