"""Leakage-safe customer snapshot features for churn prediction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChurnSnapshotConfig:
    snapshot_date: str | pd.Timestamp = "2024-03-31"
    prediction_end_date: str | pd.Timestamp = "2024-12-31"
    trailing_window_days: int = 180

    def __post_init__(self) -> None:
        snapshot = pd.Timestamp(self.snapshot_date)
        prediction_end = pd.Timestamp(self.prediction_end_date)
        if snapshot >= prediction_end:
            raise ValueError("snapshot_date must be before prediction_end_date")
        if self.trailing_window_days < 1:
            raise ValueError("trailing_window_days must be positive")
        object.__setattr__(self, "snapshot_date", snapshot)
        object.__setattr__(self, "prediction_end_date", prediction_end)


CATEGORICAL_FEATURES = [
    "region",
    "industry",
    "company_size",
    "acquisition_channel",
    "primary_plan",
]

NUMERIC_FEATURES = [
    "customer_tenure_days",
    "active_subscription_count",
    "monthly_contract_value",
    "total_seats",
    "historical_renewals",
    "days_to_renewal",
    "historical_net_revenue",
    "trailing_net_revenue",
    "average_historical_revenue",
    "historical_discount_rate",
    "sales_observation_count",
    "total_support_tickets",
    "trailing_support_tickets",
    "average_resolution_hours",
    "average_satisfaction",
    "satisfaction_response_rate",
    "has_support_history",
    "is_enterprise",
]

TARGET_COLUMN = "churned_in_horizon"


def _date_columns(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    prepared = {name: frame.copy() for name, frame in tables.items()}
    for frame in prepared.values():
        frame.columns = ["".join(column) for column in frame.columns]
    for column in ("signup_date",):
        prepared["customers"][column] = pd.to_datetime(prepared["customers"][column])
    for column in ("start_date", "renewal_date", "end_date"):
        prepared["subscriptions"][column] = pd.to_datetime(
            prepared["subscriptions"][column]
        )
    prepared["sales"]["sale_date"] = pd.to_datetime(prepared["sales"].sale_date)
    for column in ("created_date", "resolved_date"):
        prepared["support_tickets"][column] = pd.to_datetime(
            prepared["support_tickets"][column]
        )
    return prepared


def _primary_plan(values: pd.Series) -> str:
    priority = {"Basic": 0, "Professional": 1, "Enterprise": 2}
    return max(values, key=priority.__getitem__)


def build_churn_snapshot(
    tables: dict[str, pd.DataFrame],
    config: ChurnSnapshotConfig | None = None,
) -> pd.DataFrame:
    """Build one row per eligible customer using only data known at the snapshot."""
    config = config or ChurnSnapshotConfig()
    prepared = _date_columns(tables)
    customers = prepared["customers"]
    subscriptions = prepared["subscriptions"]
    sales = prepared["sales"]
    tickets = prepared["support_tickets"]
    snapshot = config.snapshot_date
    prediction_end = config.prediction_end_date
    trailing_start = snapshot - pd.Timedelta(config.trailing_window_days, unit="D")

    active_at_snapshot = subscriptions[
        subscriptions.start_date.le(snapshot)
        & (subscriptions.end_date.isna() | subscriptions.end_date.gt(snapshot))
    ].copy()
    if active_at_snapshot.empty:
        raise ValueError("No customers are eligible at the requested snapshot")

    active_at_snapshot["completed_renewals"] = (
        (snapshot - active_at_snapshot.start_date).dt.days.clip(lower=0) // 365
    )
    active_at_snapshot["days_to_next_renewal"] = 365 - (
        (snapshot - active_at_snapshot.start_date).dt.days.clip(lower=0) % 365
    )
    subscription_features = active_at_snapshot.groupby("customer_id", as_index=False).agg(
        active_subscription_count=("subscription_id", "count"),
        primary_plan=("plan_type", _primary_plan),
        monthly_contract_value=("monthly_price", "sum"),
        total_seats=("seats", "sum"),
        historical_renewals=("completed_renewals", "sum"),
        days_to_renewal=("days_to_next_renewal", "min"),
        is_enterprise=("plan_type", lambda values: int(values.eq("Enterprise").any())),
    )

    historical_sales = sales[sales.sale_date.le(snapshot)].copy()
    historical_sales["is_trailing"] = historical_sales.sale_date.ge(trailing_start)
    sales_features = historical_sales.groupby("customer_id", as_index=False).agg(
        historical_net_revenue=("net_revenue", "sum"),
        trailing_net_revenue=(
            "net_revenue",
            lambda values: float(
                values[historical_sales.loc[values.index, "is_trailing"]].sum()
            ),
        ),
        average_historical_revenue=("net_revenue", "mean"),
        total_gross_revenue=("gross_revenue", "sum"),
        total_discount=("discount_amount", "sum"),
        sales_observation_count=("sale_id", "count"),
    )
    sales_features["historical_discount_rate"] = (
        sales_features.total_discount / sales_features.total_gross_revenue
    )
    sales_features = sales_features.drop(columns=["total_gross_revenue", "total_discount"])

    historical_tickets = tickets[tickets.created_date.le(snapshot)].copy()
    historical_tickets["is_trailing"] = historical_tickets.created_date.ge(trailing_start)
    ticket_features = historical_tickets.groupby("customer_id", as_index=False).agg(
        total_support_tickets=("ticket_id", "count"),
        trailing_support_tickets=(
            "ticket_id",
            lambda values: int(
                historical_tickets.loc[values.index, "is_trailing"].sum()
            ),
        ),
        average_resolution_hours=("resolution_hours", "mean"),
        average_satisfaction=("satisfaction_score", "mean"),
        satisfaction_responses=("satisfaction_score", "count"),
    )
    ticket_features["satisfaction_response_rate"] = (
        ticket_features.satisfaction_responses / ticket_features.total_support_tickets
    )
    ticket_features["has_support_history"] = 1
    ticket_features = ticket_features.drop(columns="satisfaction_responses")

    future_churners = set(
        active_at_snapshot.loc[
            active_at_snapshot.subscription_status.eq("cancelled")
            & active_at_snapshot.end_date.gt(snapshot)
            & active_at_snapshot.end_date.le(prediction_end),
            "customer_id",
        ]
    )
    cohort = customers.merge(subscription_features, on="customer_id", how="inner")
    cohort["customer_tenure_days"] = (snapshot - cohort.signup_date).dt.days
    cohort = cohort.merge(sales_features, on="customer_id", how="left").merge(
        ticket_features, on="customer_id", how="left"
    )
    zero_fill = [
        "historical_net_revenue",
        "trailing_net_revenue",
        "sales_observation_count",
        "total_support_tickets",
        "trailing_support_tickets",
        "has_support_history",
    ]
    cohort[zero_fill] = cohort[zero_fill].fillna(0)
    cohort[TARGET_COLUMN] = cohort.customer_id.isin(future_churners).astype(int)
    keep = ["customer_id", *CATEGORICAL_FEATURES, *NUMERIC_FEATURES, TARGET_COLUMN]
    result = cohort[keep].copy()
    result.columns = ["".join(column) for column in result.columns]
    if result.customer_id.duplicated().any():
        raise ValueError("Churn snapshot must contain one row per customer")
    if result[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Churn snapshot must contain both target classes")
    return result.sort_values("customer_id").reset_index(drop=True)
