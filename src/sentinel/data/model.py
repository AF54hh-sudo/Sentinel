"""Canonical five-table AMX Tech data model."""

TABLE_COLUMNS: dict[str, list[str]] = {
    "customers": [
        "customer_id", "customer_name", "region", "country", "industry",
        "company_size", "acquisition_channel", "signup_date", "customer_status",
    ],
    "subscriptions": [
        "subscription_id", "customer_id", "plan_type", "start_date", "renewal_date",
        "end_date", "monthly_price", "seats", "subscription_status",
    ],
    "sales": [
        "sale_id", "customer_id", "subscription_id", "sale_date", "region",
        "plan_type", "gross_revenue", "discount_amount", "net_revenue", "sales_channel",
    ],
    "cloud_costs": [
        "cost_id", "date", "region", "service_type", "compute_cost", "storage_cost",
        "network_cost", "gpu_cost", "total_cost",
    ],
    "support_tickets": [
        "ticket_id", "customer_id", "created_date", "resolved_date", "ticket_category",
        "severity", "resolution_hours", "satisfaction_score", "status",
    ],
}

PRIMARY_KEYS = {
    "customers": "customer_id",
    "subscriptions": "subscription_id",
    "sales": "sale_id",
    "cloud_costs": "cost_id",
    "support_tickets": "ticket_id",
}
