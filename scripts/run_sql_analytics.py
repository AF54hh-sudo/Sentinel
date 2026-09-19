"""Run the Phase 5 AMX Tech SQL analytics demonstration."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from sentinel.config import get_settings
from sentinel.database.connection import create_database_engine
from sentinel.database.queries import (
    AnalyticsFilter,
    compare_periods,
    europe_churn_investigation,
    monthly_cost_trend,
    monthly_revenue_trend,
    revenue_by_dimension,
)


def _json_default(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> int:
    settings = get_settings()
    engine = create_database_engine(settings.database_url)
    try:
        q1 = AnalyticsFilter("2024-01-01", "2024-03-31")
        q2 = AnalyticsFilter("2024-04-01", "2024-06-30")
        full = AnalyticsFilter("2023-01-01", "2024-12-31")
        report = {
            "company": "AMX Tech",
            "q1_q2_profitability": compare_periods(engine, q1, q2),
            "europe_churn_investigation": europe_churn_investigation(
                engine,
                AnalyticsFilter(q1.start_date, q1.end_date, region="Europe"),
                AnalyticsFilter(q2.start_date, q2.end_date, region="Europe"),
            ),
            "region_metrics": revenue_by_dimension(engine, full, "region"),
            "plan_metrics": revenue_by_dimension(engine, full, "plan_type"),
            "monthly_revenue": monthly_revenue_trend(engine, full),
            "monthly_cloud_costs": monthly_cost_trend(engine, full),
            "limitations": [
                "Estimated gross contribution excludes non-infrastructure operating costs.",
                "Support and cancellation patterns are associations, not proof of causation.",
                "Infrastructure cost is not allocated to subscription plans.",
            ],
        }
    finally:
        engine.dispose()
    output = Path("data/generated/sql_analytics_report.json")
    output.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"output": str(output), "company": "AMX Tech"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
