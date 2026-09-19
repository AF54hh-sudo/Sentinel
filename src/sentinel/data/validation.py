"""Data-contract and cross-table validation for generated AMX Tech data."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.data.model import PRIMARY_KEYS, TABLE_COLUMNS


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Dataset validation failed:\n- " + "\n- ".join(self.errors))


def validate_dataset(tables: dict[str, pd.DataFrame]) -> ValidationReport:
    report = ValidationReport()
    for table, expected in TABLE_COLUMNS.items():
        if table not in tables:
            report.errors.append(f"Missing table: {table}")
            continue
        frame = tables[table]
        report.row_counts[table] = len(frame)
        missing_columns = set(expected) - set(frame.columns)
        if missing_columns:
            report.errors.append(f"{table}: missing columns {sorted(missing_columns)}")
        key = PRIMARY_KEYS[table]
        if key in frame and (frame[key].isna().any() or frame[key].duplicated().any()):
            report.errors.append(f"{table}: primary key {key} is null or duplicated")

    if report.errors:
        return report

    customers, subscriptions, sales, costs, tickets = (
        tables["customers"], tables["subscriptions"], tables["sales"],
        tables["cloud_costs"], tables["support_tickets"],
    )
    allowed_values = {
        ("customers", "region"): {"India", "North America", "Europe", "APAC"},
        ("customers", "company_size"): {"Small", "Medium", "Enterprise"},
        ("customers", "customer_status"): {"active", "inactive"},
        ("subscriptions", "plan_type"): {"Basic", "Professional", "Enterprise"},
        ("subscriptions", "subscription_status"): {"active", "cancelled", "expired"},
        ("support_tickets", "severity"): {"Low", "Medium", "High", "Critical"},
        ("support_tickets", "status"): {"resolved", "open"},
    }
    for (table_name, column), allowed in allowed_values.items():
        unexpected = set(tables[table_name][column].dropna().unique()) - allowed
        if unexpected:
            report.errors.append(
                f"{table_name}.{column} contains unexpected values: {sorted(unexpected)}"
            )
    customer_ids = set(customers.customer_id)
    subscription_ids = set(subscriptions.subscription_id)
    if not set(subscriptions.customer_id).issubset(customer_ids):
        report.errors.append("subscriptions contains orphan customer_id values")
    if not set(sales.customer_id).issubset(customer_ids):
        report.errors.append("sales contains orphan customer_id values")
    if not set(sales.subscription_id).issubset(subscription_ids):
        report.errors.append("sales contains orphan subscription_id values")
    if not set(tickets.customer_id).issubset(customer_ids):
        report.errors.append("support_tickets contains orphan customer_id values")

    subscription_dates = subscriptions.assign(
        start_date=pd.to_datetime(subscriptions.start_date, errors="coerce"),
        renewal_date=pd.to_datetime(subscriptions.renewal_date, errors="coerce"),
        end_date=pd.to_datetime(subscriptions.end_date, errors="coerce"),
    )
    invalid_end_state = (
        subscription_dates.subscription_status.eq("active")
        ^ subscription_dates.end_date.isna()
    )
    if invalid_end_state.any():
        report.errors.append("subscriptions end_date is inconsistent with active status")
    if (subscription_dates.renewal_date < subscription_dates.start_date).any():
        report.errors.append("subscriptions has renewal_date before start_date")
    if (
        subscription_dates.end_date.notna()
        & (subscription_dates.end_date < subscription_dates.start_date)
    ).any():
        report.errors.append("subscriptions has end_date before start_date")

    ticket_dates = tickets.assign(
        created_date=pd.to_datetime(tickets.created_date, errors="coerce"),
        resolved_date=pd.to_datetime(tickets.resolved_date, errors="coerce"),
    )
    invalid_ticket_state = ticket_dates.status.eq("open") ^ ticket_dates.resolved_date.isna()
    if invalid_ticket_state.any():
        report.errors.append("support_tickets resolved_date is inconsistent with status")
    if (
        ticket_dates.resolved_date.notna()
        & (ticket_dates.resolved_date < ticket_dates.created_date)
    ).any():
        report.errors.append("support_tickets has resolved_date before created_date")

    expected_customer_status = customers.customer_id.map(
        subscriptions.groupby("customer_id").subscription_status.apply(
            lambda values: "active" if values.eq("active").any() else "inactive"
        )
    )
    if not expected_customer_status.equals(customers.customer_status):
        report.errors.append("customers.customer_status is inconsistent with subscriptions")

    sale_dimensions = sales.merge(
        subscriptions[["subscription_id", "customer_id", "plan_type"]],
        on="subscription_id",
        suffixes=("", "_subscription"),
        how="left",
    ).merge(
        customers[["customer_id", "region"]],
        on="customer_id",
        suffixes=("", "_customer"),
        how="left",
    )
    if not sale_dimensions.customer_id.eq(sale_dimensions.customer_id_subscription).all():
        report.errors.append("sales.customer_id does not match its subscription")
    if not sale_dimensions.plan_type.eq(sale_dimensions.plan_type_subscription).all():
        report.errors.append("sales.plan_type does not match its subscription")
    if not sale_dimensions.region.eq(sale_dimensions.region_customer).all():
        report.errors.append("sales.region does not match its customer")
    if (subscriptions.monthly_price <= 0).any() or (subscriptions.seats <= 0).any():
        report.errors.append("subscriptions has non-positive price or seats")
    if (sales[["gross_revenue", "discount_amount", "net_revenue"]].lt(0).any()).any():
        report.errors.append("sales has negative monetary values")
    if not np.allclose(sales.gross_revenue - sales.discount_amount, sales.net_revenue, atol=0.02):
        report.errors.append("sales net_revenue does not reconcile")
    components = costs.compute_cost + costs.storage_cost + costs.network_cost + costs.gpu_cost
    if not np.allclose(components, costs.total_cost, atol=0.03):
        report.errors.append("cloud_costs total_cost does not reconcile")
    scores = tickets.satisfaction_score.dropna()
    if not scores.between(1, 5).all():
        report.errors.append("support_tickets satisfaction_score is outside 1-5")
    if (tickets.resolution_hours <= 0).any():
        report.errors.append("support_tickets has non-positive resolution_hours")

    business_cols = [column for column in tickets.columns if column != "ticket_id"]
    duplicates = int(tickets.duplicated(business_cols).sum())
    if duplicates:
        report.warnings.append(f"support_tickets contains {duplicates} duplicate business records")
    for table, frame in tables.items():
        missing = int(frame.isna().sum().sum())
        if missing:
            report.warnings.append(f"{table} contains {missing} missing values (including valid nullable dates)")
    return report
