"""Run Phase 4 validation, cleaning, profiling, and exploratory summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinel.config import get_settings
from sentinel.data.loader import read_csv_dataset
from sentinel.data.validation import validate_dataset
from sentinel.database.connection import create_database_engine
from sentinel.database.loader import read_database_tables
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
from sentinel.eda.summary import profile_dataset


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats before strict JSON serialization."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reproducible Sentinel Phase 4 EDA.")
    parser.add_argument("--source", choices=["csv", "database"], default="csv")
    parser.add_argument("--output", type=Path, default=Path("data/generated/eda_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if args.source == "database":
        engine = create_database_engine(settings.database_url)
        try:
            raw = read_database_tables(engine)
        finally:
            engine.dispose()
    else:
        raw = read_csv_dataset(settings.data_dir)

    raw_validation = validate_dataset(raw)
    raw_validation.raise_for_errors()
    cleaned, cleaning = clean_dataset(raw)
    report = {
        "source": args.source,
        "raw_validation": {
            "row_counts": raw_validation.row_counts,
            "warnings": raw_validation.warnings,
        },
        "cleaning": {
            "rows_before": cleaning.rows_before,
            "rows_after": cleaning.rows_after,
            "duplicate_events_removed": cleaning.duplicate_events_removed,
            "industries_filled": cleaning.industries_filled,
            "satisfaction_values_retained_missing": (
                cleaning.satisfaction_values_retained_missing
            ),
        },
        "profile": profile_dataset(cleaned),
        "business_metrics": business_metrics(cleaned),
        "revenue_by_region": dimension_metrics(cleaned, "region").to_dict("records"),
        "revenue_by_plan": dimension_metrics(cleaned, "plan_type").to_dict("records"),
        "monthly_revenue": monthly_revenue(cleaned["sales"]).to_dict("records"),
        "monthly_costs": monthly_costs(cleaned["cloud_costs"]).to_dict("records"),
        "monthly_churn": monthly_churn_trend(cleaned["subscriptions"]).to_dict("records"),
        "monthly_tickets": monthly_ticket_trend(cleaned["support_tickets"]).to_dict("records"),
        "customer_behavior_spearman": customer_behavior_correlations(cleaned).to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(report), indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"source": args.source, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
