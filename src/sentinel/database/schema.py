"""SQLAlchemy schema for Sentinel's five AMX Tech business tables."""

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
)
from sqlalchemy.engine import Engine

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=NAMING_CONVENTION)

customers = Table(
    "customers",
    metadata,
    Column("customer_id", String(9), primary_key=True),
    Column("customer_name", String(160), nullable=False),
    Column("region", String(32), nullable=False),
    Column("country", String(64), nullable=False),
    Column("industry", String(32)),
    Column("company_size", String(16), nullable=False),
    Column("acquisition_channel", String(32), nullable=False),
    Column("signup_date", Date, nullable=False),
    Column("customer_status", String(16), nullable=False),
    CheckConstraint(
        "region IN ('India', 'North America', 'Europe', 'APAC')",
        name="customers_region",
    ),
    CheckConstraint(
        "company_size IN ('Small', 'Medium', 'Enterprise')",
        name="customers_company_size",
    ),
    CheckConstraint(
        "customer_status IN ('active', 'inactive')",
        name="customers_status",
    ),
)

subscriptions = Table(
    "subscriptions",
    metadata,
    Column("subscription_id", String(9), primary_key=True),
    Column(
        "customer_id",
        String(9),
        ForeignKey("customers.customer_id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("plan_type", String(20), nullable=False),
    Column("start_date", Date, nullable=False),
    Column("renewal_date", Date, nullable=False),
    Column("end_date", Date),
    Column("monthly_price", Numeric(14, 2), nullable=False),
    Column("seats", Integer, nullable=False),
    Column("subscription_status", String(16), nullable=False),
    CheckConstraint(
        "plan_type IN ('Basic', 'Professional', 'Enterprise')",
        name="subscriptions_plan",
    ),
    CheckConstraint(
        "subscription_status IN ('active', 'cancelled', 'expired')",
        name="subscriptions_status",
    ),
    CheckConstraint("monthly_price > 0", name="subscriptions_price_positive"),
    CheckConstraint("seats > 0", name="subscriptions_seats_positive"),
    CheckConstraint("renewal_date >= start_date", name="subscriptions_renewal_after_start"),
    CheckConstraint(
        "(subscription_status = 'active' AND end_date IS NULL) OR "
        "(subscription_status <> 'active' AND end_date IS NOT NULL)",
        name="subscriptions_end_date_status",
    ),
)

sales = Table(
    "sales",
    metadata,
    Column("sale_id", String(10), primary_key=True),
    Column(
        "customer_id",
        String(9),
        ForeignKey("customers.customer_id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column(
        "subscription_id",
        String(9),
        ForeignKey("subscriptions.subscription_id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("sale_date", Date, nullable=False),
    Column("region", String(32), nullable=False),
    Column("plan_type", String(20), nullable=False),
    Column("gross_revenue", Numeric(14, 2), nullable=False),
    Column("discount_amount", Numeric(14, 2), nullable=False),
    Column("net_revenue", Numeric(14, 2), nullable=False),
    Column("sales_channel", String(32), nullable=False),
    CheckConstraint("gross_revenue >= 0", name="sales_gross_nonnegative"),
    CheckConstraint("discount_amount >= 0", name="sales_discount_nonnegative"),
    CheckConstraint("net_revenue >= 0", name="sales_net_nonnegative"),
    CheckConstraint("discount_amount <= gross_revenue", name="sales_discount_bounded"),
    CheckConstraint(
        "abs(gross_revenue - discount_amount - net_revenue) <= 0.02",
        name="sales_reconciliation",
    ),
)

cloud_costs = Table(
    "cloud_costs",
    metadata,
    Column("cost_id", String(9), primary_key=True),
    Column("date", Date, nullable=False),
    Column("region", String(32), nullable=False),
    Column("service_type", String(48), nullable=False),
    Column("compute_cost", Numeric(14, 2), nullable=False),
    Column("storage_cost", Numeric(14, 2), nullable=False),
    Column("network_cost", Numeric(14, 2), nullable=False),
    Column("gpu_cost", Numeric(14, 2), nullable=False),
    Column("total_cost", Numeric(14, 2), nullable=False),
    CheckConstraint(
        "compute_cost >= 0 AND storage_cost >= 0 AND network_cost >= 0 "
        "AND gpu_cost >= 0 AND total_cost >= 0",
        name="cloud_costs_nonnegative",
    ),
    CheckConstraint(
        "abs(compute_cost + storage_cost + network_cost + gpu_cost - total_cost) <= 0.03",
        name="cloud_costs_reconciliation",
    ),
)

support_tickets = Table(
    "support_tickets",
    metadata,
    Column("ticket_id", String(10), primary_key=True),
    Column(
        "customer_id",
        String(9),
        ForeignKey("customers.customer_id", ondelete="RESTRICT"),
        nullable=False,
    ),
    Column("created_date", Date, nullable=False),
    Column("resolved_date", Date),
    Column("ticket_category", String(32), nullable=False),
    Column("severity", String(16), nullable=False),
    Column("resolution_hours", Numeric(12, 2), nullable=False),
    Column("satisfaction_score", Numeric(2, 1)),
    Column("status", String(16), nullable=False),
    CheckConstraint("resolution_hours > 0", name="tickets_resolution_positive"),
    CheckConstraint(
        "satisfaction_score IS NULL OR satisfaction_score BETWEEN 1 AND 5",
        name="tickets_satisfaction_range",
    ),
    CheckConstraint("status IN ('resolved', 'open')", name="tickets_status"),
    CheckConstraint(
        "(status = 'resolved' AND resolved_date IS NOT NULL) OR "
        "(status = 'open' AND resolved_date IS NULL)",
        name="tickets_resolved_date_status",
    ),
    CheckConstraint(
        "resolved_date IS NULL OR resolved_date >= created_date",
        name="tickets_resolution_after_creation",
    ),
)

Index("ix_customers_region_industry", customers.c.region, customers.c.industry)
Index("ix_subscriptions_customer_status", subscriptions.c.customer_id, subscriptions.c.subscription_status)
Index("ix_subscriptions_plan_status", subscriptions.c.plan_type, subscriptions.c.subscription_status)
Index("ix_sales_date_region", sales.c.sale_date, sales.c.region)
Index("ix_sales_customer_date", sales.c.customer_id, sales.c.sale_date)
Index("ix_sales_subscription", sales.c.subscription_id)
Index("ix_cloud_costs_date_region", cloud_costs.c.date, cloud_costs.c.region)
Index("ix_tickets_customer_created", support_tickets.c.customer_id, support_tickets.c.created_date)
Index("ix_tickets_created_severity", support_tickets.c.created_date, support_tickets.c.severity)

TABLES_IN_LOAD_ORDER = (customers, subscriptions, sales, cloud_costs, support_tickets)
TABLES_IN_DELETE_ORDER = tuple(reversed(TABLES_IN_LOAD_ORDER))


def create_schema(engine: Engine) -> None:
    """Create missing business tables, constraints, and indexes."""
    metadata.create_all(engine, checkfirst=True)
