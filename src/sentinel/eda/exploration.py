"""Business metrics and time-based exploratory views computed with Pandas."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def business_metrics(tables: dict[str, pd.DataFrame]) -> dict[str, float | int]:
    customers = tables["customers"]
    subscriptions = tables["subscriptions"]
    sales = tables["sales"]
    costs = tables["cloud_costs"]
    tickets = tables["support_tickets"]
    total_subscriptions = len(subscriptions)
    cancelled = int(subscriptions.subscription_status.eq("cancelled").sum())
    due = subscriptions[pd.to_datetime(subscriptions.renewal_date) <= pd.to_datetime(sales.sale_date).max()]
    renewed = due.end_date.isna() | (
        pd.to_datetime(due.end_date) >= pd.to_datetime(due.renewal_date)
    )
    net_revenue = float(sales.net_revenue.sum())
    infrastructure_cost = float(costs.total_cost.sum())
    return {
        "total_revenue": float(sales.gross_revenue.sum()),
        "net_revenue": net_revenue,
        "average_revenue_per_customer": net_revenue / sales.customer_id.nunique(),
        "churn_rate": cancelled / total_subscriptions,
        "renewal_rate": float(renewed.mean()) if len(due) else np.nan,
        "active_subscriptions": int(subscriptions.subscription_status.eq("active").sum()),
        "average_discount_rate": float(
            sales.discount_amount.sum() / sales.gross_revenue.sum()
        ),
        "infrastructure_cost": infrastructure_cost,
        "cost_to_revenue_ratio": infrastructure_cost / net_revenue,
        "estimated_gross_contribution": net_revenue - infrastructure_cost,
        "average_support_resolution_hours": float(tickets.resolution_hours.mean()),
        "average_satisfaction_score": float(tickets.satisfaction_score.mean()),
        "customers": int(customers.customer_id.nunique()),
    }


def monthly_revenue(sales: pd.DataFrame) -> pd.DataFrame:
    frame = sales.copy()
    frame["month"] = _month_start(frame.sale_date)
    result = frame.groupby("month", as_index=False).agg(
        gross_revenue=("gross_revenue", "sum"),
        discount_amount=("discount_amount", "sum"),
        net_revenue=("net_revenue", "sum"),
        active_customers=("customer_id", "nunique"),
    )
    result["net_revenue_growth"] = result.net_revenue.pct_change()
    result["discount_rate"] = result.discount_amount / result.gross_revenue
    return result


def monthly_costs(cloud_costs: pd.DataFrame) -> pd.DataFrame:
    frame = cloud_costs.copy()
    frame["month"] = _month_start(frame.date)
    return frame.groupby("month", as_index=False).agg(
        compute_cost=("compute_cost", "sum"),
        storage_cost=("storage_cost", "sum"),
        network_cost=("network_cost", "sum"),
        gpu_cost=("gpu_cost", "sum"),
        total_cost=("total_cost", "sum"),
    )


def monthly_ticket_trend(tickets: pd.DataFrame) -> pd.DataFrame:
    frame = tickets.copy()
    frame["month"] = _month_start(frame.created_date)
    return frame.groupby("month", as_index=False).agg(
        ticket_count=("ticket_id", "count"),
        customers_with_tickets=("customer_id", "nunique"),
        average_resolution_hours=("resolution_hours", "mean"),
        average_satisfaction=("satisfaction_score", "mean"),
    )


def monthly_churn_trend(subscriptions: pd.DataFrame) -> pd.DataFrame:
    frame = subscriptions.copy()
    frame["start_date"] = pd.to_datetime(frame.start_date)
    frame["end_date"] = pd.to_datetime(frame.end_date)
    last_observation = max(frame.start_date.max(), frame.end_date.max())
    months = pd.date_range(frame.start_date.min().to_period("M").to_timestamp(), last_observation.to_period("M").to_timestamp(), freq="MS")
    rows: list[dict[str, Any]] = []
    for month in months:
        next_month = month + pd.offsets.MonthBegin(1)
        at_risk = (frame.start_date < next_month) & (frame.end_date.isna() | (frame.end_date >= month))
        cancellations = (
            frame.subscription_status.eq("cancelled")
            & frame.end_date.ge(month)
            & frame.end_date.lt(next_month)
        )
        denominator = int(at_risk.sum())
        rows.append(
            {
                "month": month,
                "subscriptions_at_risk": denominator,
                "cancellations": int(cancellations.sum()),
                "churn_rate": float(cancellations.sum() / denominator) if denominator else np.nan,
            }
        )
    return pd.DataFrame(rows)


def dimension_metrics(tables: dict[str, pd.DataFrame], dimension: str) -> pd.DataFrame:
    if dimension not in {"region", "plan_type"}:
        raise ValueError("dimension must be 'region' or 'plan_type'")
    sales = tables["sales"].groupby(dimension, as_index=False).agg(
        gross_revenue=("gross_revenue", "sum"),
        discount_amount=("discount_amount", "sum"),
        net_revenue=("net_revenue", "sum"),
        revenue_customers=("customer_id", "nunique"),
    )
    subscriptions = tables["subscriptions"]
    if dimension == "region":
        subscriptions = subscriptions.merge(
            tables["customers"][["customer_id", "region"]], on="customer_id", how="left"
        )
    churn = subscriptions.groupby(dimension, as_index=False).agg(
        subscriptions=("subscription_id", "count"),
        cancellations=("subscription_status", lambda values: int(values.eq("cancelled").sum())),
    )
    churn["churn_rate"] = churn.cancellations / churn.subscriptions
    result = sales.merge(churn, on=dimension, how="outer")
    result["discount_rate"] = result.discount_amount / result.gross_revenue
    return result.sort_values("net_revenue", ascending=False).reset_index(drop=True)


def customer_behavior_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sales = tables["sales"].groupby("customer_id", as_index=False).agg(
        total_net_revenue=("net_revenue", "sum"),
        average_discount=("discount_amount", "mean"),
    )
    tickets = tables["support_tickets"].groupby("customer_id", as_index=False).agg(
        ticket_count=("ticket_id", "count"),
        average_resolution_hours=("resolution_hours", "mean"),
        average_satisfaction=("satisfaction_score", "mean"),
    )
    subs = tables["subscriptions"].groupby("customer_id", as_index=False).agg(
        subscription_count=("subscription_id", "count"),
        churned=("subscription_status", lambda values: int(values.eq("cancelled").any())),
    )
    return subs.merge(sales, on="customer_id", how="left").merge(
        tickets, on="customer_id", how="left"
    ).fillna({"total_net_revenue": 0, "average_discount": 0, "ticket_count": 0})


def customer_behavior_correlations(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return customer_behavior_frame(tables).drop(columns="customer_id").corr(method="spearman")
