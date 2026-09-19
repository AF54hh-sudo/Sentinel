"""Phase 3 database smoke queries and conservative read-only execution."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Connection, Engine, Select, and_, case, func, or_, select, text

from sentinel.database.schema import (
    TABLES_IN_LOAD_ORDER,
    cloud_costs,
    customers,
    sales,
    subscriptions,
    support_tickets,
)

VALID_REGIONS = frozenset({"India", "North America", "Europe", "APAC"})
VALID_PLANS = frozenset({"Basic", "Professional", "Enterprise"})

FORBIDDEN_SQL = re.compile(
    r"\b(ALTER|ANALYZE|CALL|COMMENT|COPY|CREATE|DELETE|DO|DROP|GRANT|INSERT|INTO|LOCK|"
    r"MERGE|REFRESH|REINDEX|RESET|REVOKE|SET|TRUNCATE|UPDATE|VACUUM)\b",
    flags=re.IGNORECASE,
)


def _date_value(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid ISO date: {value!r}") from error


@dataclass(frozen=True)
class AnalyticsFilter:
    """Validated, inclusive filters shared by Phase 5 analytical queries."""

    start_date: date | str
    end_date: date | str
    region: str | None = None
    plan_type: str | None = None

    def __post_init__(self) -> None:
        start = _date_value(self.start_date)
        end = _date_value(self.end_date)
        if start > end:
            raise ValueError("start_date must be on or before end_date")
        if self.region is not None and self.region not in VALID_REGIONS:
            raise ValueError(f"Unknown region: {self.region!r}")
        if self.plan_type is not None and self.plan_type not in VALID_PLANS:
            raise ValueError(f"Unknown plan_type: {self.plan_type!r}")
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)


def _number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return value


def _mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _number(value) for key, value in row.items()}


def _sales_clauses(filters: AnalyticsFilter) -> list[Any]:
    clauses = [sales.c.sale_date.between(filters.start_date, filters.end_date)]
    if filters.region:
        clauses.append(sales.c.region == filters.region)
    if filters.plan_type:
        clauses.append(sales.c.plan_type == filters.plan_type)
    return clauses


def _subscription_clauses(filters: AnalyticsFilter) -> list[Any]:
    clauses: list[Any] = []
    if filters.region:
        clauses.append(customers.c.region == filters.region)
    if filters.plan_type:
        clauses.append(subscriptions.c.plan_type == filters.plan_type)
    return clauses


def _reject_plan_cost_allocation(filters: AnalyticsFilter) -> None:
    if filters.plan_type:
        raise ValueError(
            "Infrastructure costs are not allocated by plan; plan-filtered cost or "
            "contribution would be misleading"
        )


def validate_read_only_sql(statement: str) -> str:
    """Allow one plain SELECT statement; reject mutations and multi-statements."""
    normalized = statement.strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    if not normalized or ";" in normalized:
        raise ValueError("Exactly one SQL statement is allowed")
    if not re.match(r"^SELECT\b", normalized, flags=re.IGNORECASE):
        raise ValueError("Only SELECT statements are allowed")
    if FORBIDDEN_SQL.search(normalized):
        raise ValueError("Potentially mutating SQL is blocked")
    return normalized


def execute_read_only(
    connection: Connection,
    statement: str,
    parameters: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    safe_statement = validate_read_only_sql(statement)
    result = connection.execute(text(safe_statement), dict(parameters or {}))
    return [dict(row) for row in result.mappings().all()]


def revenue_summary(engine: Engine, filters: AnalyticsFilter) -> dict[str, Any]:
    """Revenue, discounts, customer count, and growth-ready totals for one period."""
    statement = select(
        func.coalesce(func.sum(sales.c.gross_revenue), 0).label("gross_revenue"),
        func.coalesce(func.sum(sales.c.discount_amount), 0).label("discount_amount"),
        func.coalesce(func.sum(sales.c.net_revenue), 0).label("net_revenue"),
        func.count(func.distinct(sales.c.customer_id)).label("revenue_customers"),
        func.count(sales.c.sale_id).label("sales_observations"),
    ).where(*_sales_clauses(filters))
    with engine.connect() as connection:
        result = _mapping(connection.execute(statement).mappings().one())
    gross = float(result["gross_revenue"] or 0)
    net = float(result["net_revenue"] or 0)
    customers_count = int(result["revenue_customers"] or 0)
    result["discount_rate"] = float(result["discount_amount"] or 0) / gross if gross else None
    result["average_revenue_per_customer"] = net / customers_count if customers_count else None
    return result


def cost_summary(engine: Engine, filters: AnalyticsFilter) -> dict[str, Any]:
    """Infrastructure component totals. Plan-level allocation is intentionally forbidden."""
    _reject_plan_cost_allocation(filters)
    clauses = [cloud_costs.c.date.between(filters.start_date, filters.end_date)]
    if filters.region:
        clauses.append(cloud_costs.c.region == filters.region)
    statement = select(
        func.coalesce(func.sum(cloud_costs.c.compute_cost), 0).label("compute_cost"),
        func.coalesce(func.sum(cloud_costs.c.storage_cost), 0).label("storage_cost"),
        func.coalesce(func.sum(cloud_costs.c.network_cost), 0).label("network_cost"),
        func.coalesce(func.sum(cloud_costs.c.gpu_cost), 0).label("gpu_cost"),
        func.coalesce(func.sum(cloud_costs.c.total_cost), 0).label("total_cost"),
        func.count(cloud_costs.c.cost_id).label("cost_observations"),
    ).where(*clauses)
    with engine.connect() as connection:
        return _mapping(connection.execute(statement).mappings().one())


def contribution_summary(engine: Engine, filters: AnalyticsFilter) -> dict[str, Any]:
    """Net revenue minus recorded infrastructure cost, not accounting profit."""
    _reject_plan_cost_allocation(filters)
    revenue = revenue_summary(engine, filters)
    costs = cost_summary(engine, filters)
    net = float(revenue["net_revenue"] or 0)
    total_cost = float(costs["total_cost"] or 0)
    return {
        **revenue,
        **costs,
        "estimated_gross_contribution": net - total_cost,
        "cost_to_revenue_ratio": total_cost / net if net else None,
    }


def subscription_summary(engine: Engine, filters: AnalyticsFilter) -> dict[str, Any]:
    """Period churn and renewal metrics with explicit exposure denominators."""
    dimension_clauses = _subscription_clauses(filters)
    at_risk = and_(
        subscriptions.c.start_date <= filters.end_date,
        or_(subscriptions.c.end_date.is_(None), subscriptions.c.end_date >= filters.start_date),
    )
    cancelled_in_period = and_(
        subscriptions.c.subscription_status == "cancelled",
        subscriptions.c.end_date.between(filters.start_date, filters.end_date),
    )
    due_for_renewal = subscriptions.c.renewal_date.between(
        filters.start_date, filters.end_date
    )
    renewed = and_(
        due_for_renewal,
        or_(subscriptions.c.end_date.is_(None), subscriptions.c.end_date >= subscriptions.c.renewal_date),
    )
    statement = (
        select(
            func.sum(case((at_risk, 1), else_=0)).label("subscriptions_at_risk"),
            func.sum(case((cancelled_in_period, 1), else_=0)).label("cancellations"),
            func.sum(case((due_for_renewal, 1), else_=0)).label("renewals_due"),
            func.sum(case((renewed, 1), else_=0)).label("renewals_retained"),
            func.sum(
                case((subscriptions.c.subscription_status == "active", 1), else_=0)
            ).label("currently_active_subscriptions"),
        )
        .select_from(subscriptions.join(customers))
        .where(*dimension_clauses)
    )
    with engine.connect() as connection:
        result = _mapping(connection.execute(statement).mappings().one())
    at_risk_count = int(result["subscriptions_at_risk"] or 0)
    renewals_due = int(result["renewals_due"] or 0)
    result["churn_rate"] = (
        int(result["cancellations"] or 0) / at_risk_count if at_risk_count else None
    )
    result["renewal_rate"] = (
        int(result["renewals_retained"] or 0) / renewals_due if renewals_due else None
    )
    return result


def support_summary(engine: Engine, filters: AnalyticsFilter) -> dict[str, Any]:
    """Deduplicated ticket metrics for the inclusive created-date period."""
    if filters.plan_type:
        raise ValueError(
            "A customer may have multiple plans; ticket metrics cannot be safely allocated by plan"
        )
    business_columns = [
        support_tickets.c.customer_id,
        support_tickets.c.created_date,
        support_tickets.c.resolved_date,
        support_tickets.c.ticket_category,
        support_tickets.c.severity,
        support_tickets.c.resolution_hours,
        support_tickets.c.satisfaction_score,
        support_tickets.c.status,
    ]
    ranked = (
        select(
            support_tickets,
            func.row_number()
            .over(partition_by=business_columns, order_by=support_tickets.c.ticket_id)
            .label("duplicate_rank"),
        )
        .subquery("ranked_tickets")
    )
    clauses = [
        ranked.c.created_date.between(filters.start_date, filters.end_date),
        ranked.c.duplicate_rank == 1,
    ]
    from_clause = ranked
    if filters.region:
        from_clause = ranked.join(customers, ranked.c.customer_id == customers.c.customer_id)
        clauses.append(customers.c.region == filters.region)
    statement = select(
        func.count(ranked.c.ticket_id).label("ticket_count"),
        func.count(func.distinct(ranked.c.customer_id)).label("customers_with_tickets"),
        func.avg(ranked.c.resolution_hours).label("average_resolution_hours"),
        func.avg(ranked.c.satisfaction_score).label("average_satisfaction"),
        func.sum(case((ranked.c.status == "open", 1), else_=0)).label("open_tickets"),
    ).select_from(from_clause).where(*clauses)
    with engine.connect() as connection:
        return _mapping(connection.execute(statement).mappings().one())


def monthly_revenue_trend(engine: Engine, filters: AnalyticsFilter) -> list[dict[str, Any]]:
    month = func.date_trunc("month", sales.c.sale_date).cast(sales.c.sale_date.type).label("month")
    statement = (
        select(
            month,
            func.sum(sales.c.gross_revenue).label("gross_revenue"),
            func.sum(sales.c.discount_amount).label("discount_amount"),
            func.sum(sales.c.net_revenue).label("net_revenue"),
            func.count(func.distinct(sales.c.customer_id)).label("active_customers"),
        )
        .where(*_sales_clauses(filters))
        .group_by(month)
        .order_by(month)
    )
    with engine.connect() as connection:
        rows = [_mapping(row) for row in connection.execute(statement).mappings()]
    previous: float | None = None
    for row in rows:
        net = float(row["net_revenue"])
        row["net_revenue_growth"] = (net / previous - 1) if previous else None
        row["discount_rate"] = (
            float(row["discount_amount"]) / float(row["gross_revenue"])
            if row["gross_revenue"]
            else None
        )
        previous = net
    return rows


def monthly_cost_trend(engine: Engine, filters: AnalyticsFilter) -> list[dict[str, Any]]:
    _reject_plan_cost_allocation(filters)
    month = func.date_trunc("month", cloud_costs.c.date).cast(cloud_costs.c.date.type).label("month")
    clauses = [cloud_costs.c.date.between(filters.start_date, filters.end_date)]
    if filters.region:
        clauses.append(cloud_costs.c.region == filters.region)
    statement = (
        select(
            month,
            func.sum(cloud_costs.c.compute_cost).label("compute_cost"),
            func.sum(cloud_costs.c.storage_cost).label("storage_cost"),
            func.sum(cloud_costs.c.network_cost).label("network_cost"),
            func.sum(cloud_costs.c.gpu_cost).label("gpu_cost"),
            func.sum(cloud_costs.c.total_cost).label("total_cost"),
        )
        .where(*clauses)
        .group_by(month)
        .order_by(month)
    )
    with engine.connect() as connection:
        return [_mapping(row) for row in connection.execute(statement).mappings()]


def revenue_by_dimension(
    engine: Engine,
    filters: AnalyticsFilter,
    dimension: str,
) -> list[dict[str, Any]]:
    """Revenue and period churn grouped by the allowlisted region or plan dimension."""
    if dimension not in {"region", "plan_type"}:
        raise ValueError("dimension must be 'region' or 'plan_type'")
    dimension_column = sales.c.region if dimension == "region" else sales.c.plan_type
    revenue_statement = (
        select(
            dimension_column.label(dimension),
            func.sum(sales.c.gross_revenue).label("gross_revenue"),
            func.sum(sales.c.discount_amount).label("discount_amount"),
            func.sum(sales.c.net_revenue).label("net_revenue"),
            func.count(func.distinct(sales.c.customer_id)).label("revenue_customers"),
        )
        .where(*_sales_clauses(filters))
        .group_by(dimension_column)
    )
    sub_dimension = customers.c.region if dimension == "region" else subscriptions.c.plan_type
    at_risk = and_(
        subscriptions.c.start_date <= filters.end_date,
        or_(subscriptions.c.end_date.is_(None), subscriptions.c.end_date >= filters.start_date),
    )
    cancelled = and_(
        subscriptions.c.subscription_status == "cancelled",
        subscriptions.c.end_date.between(filters.start_date, filters.end_date),
    )
    churn_statement = (
        select(
            sub_dimension.label(dimension),
            func.sum(case((at_risk, 1), else_=0)).label("subscriptions_at_risk"),
            func.sum(case((cancelled, 1), else_=0)).label("cancellations"),
        )
        .select_from(subscriptions.join(customers))
        .where(*_subscription_clauses(filters))
        .group_by(sub_dimension)
    )
    with engine.connect() as connection:
        revenue_rows = {
            row[dimension]: _mapping(row)
            for row in connection.execute(revenue_statement).mappings()
        }
        churn_rows = {
            row[dimension]: _mapping(row)
            for row in connection.execute(churn_statement).mappings()
        }
    results: list[dict[str, Any]] = []
    for key in sorted(revenue_rows.keys() | churn_rows.keys()):
        row = {**revenue_rows.get(key, {dimension: key}), **churn_rows.get(key, {})}
        gross = float(row.get("gross_revenue") or 0)
        at_risk_count = int(row.get("subscriptions_at_risk") or 0)
        row["discount_rate"] = float(row.get("discount_amount") or 0) / gross if gross else None
        row["churn_rate"] = (
            int(row.get("cancellations") or 0) / at_risk_count if at_risk_count else None
        )
        results.append(row)
    return sorted(results, key=lambda row: float(row.get("net_revenue") or 0), reverse=True)


def compare_periods(
    engine: Engine,
    first: AnalyticsFilter,
    second: AnalyticsFilter,
) -> dict[str, Any]:
    """Compare revenue, cost, and estimated contribution for two periods."""
    if first.region != second.region or first.plan_type != second.plan_type:
        raise ValueError("Period comparisons require identical region and plan filters")
    first_values = contribution_summary(engine, first)
    second_values = contribution_summary(engine, second)
    changes: dict[str, float | None] = {}
    for metric in (
        "gross_revenue",
        "discount_amount",
        "net_revenue",
        "gpu_cost",
        "total_cost",
        "estimated_gross_contribution",
    ):
        first_value = float(first_values[metric] or 0)
        second_value = float(second_values[metric] or 0)
        changes[f"{metric}_change"] = second_value - first_value
        changes[f"{metric}_growth"] = (
            second_value / first_value - 1 if first_value else None
        )
    return {"first_period": first_values, "second_period": second_values, "changes": changes}


def europe_churn_investigation(
    engine: Engine,
    first: AnalyticsFilter,
    second: AnalyticsFilter,
) -> dict[str, Any]:
    """Compare Europe subscription and support signals without implying causation."""
    if first.region != "Europe" or second.region != "Europe":
        raise ValueError("Europe investigation filters must use region='Europe'")
    return {
        "first_period": {
            "subscriptions": subscription_summary(engine, first),
            "support": support_summary(engine, first),
        },
        "second_period": {
            "subscriptions": subscription_summary(engine, second),
            "support": support_summary(engine, second),
        },
        "interpretation_boundary": (
            "Support quality and cancellation are compared as associations; this query does "
            "not establish causation."
        ),
    }


def table_count_statements() -> dict[str, Select[Any]]:
    return {
        table.name: select(func.count()).select_from(table)
        for table in TABLES_IN_LOAD_ORDER
    }


def table_row_counts(engine: Engine) -> dict[str, int]:
    with engine.connect() as connection:
        return {
            name: int(connection.scalar(statement) or 0)
            for name, statement in table_count_statements().items()
        }


def foreign_key_violation_counts(engine: Engine) -> dict[str, int]:
    """Return orphan counts for every business foreign-key relationship."""
    statements = {
        "subscriptions_customer": """
            SELECT COUNT(*) FROM subscriptions s
            LEFT JOIN customers c ON c.customer_id = s.customer_id
            WHERE c.customer_id IS NULL
        """,
        "sales_customer": """
            SELECT COUNT(*) FROM sales s
            LEFT JOIN customers c ON c.customer_id = s.customer_id
            WHERE c.customer_id IS NULL
        """,
        "sales_subscription": """
            SELECT COUNT(*) FROM sales s
            LEFT JOIN subscriptions sub ON sub.subscription_id = s.subscription_id
            WHERE sub.subscription_id IS NULL
        """,
        "tickets_customer": """
            SELECT COUNT(*) FROM support_tickets t
            LEFT JOIN customers c ON c.customer_id = t.customer_id
            WHERE c.customer_id IS NULL
        """,
    }
    with engine.connect() as connection:
        return {
            name: int(connection.execute(text(statement)).scalar_one())
            for name, statement in statements.items()
        }
