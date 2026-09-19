"""Run the reproducible AMX Tech Phase 6 statistical analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.config import get_settings
from sentinel.database.connection import create_database_engine
from sentinel.database.loader import read_database_tables
from sentinel.eda.cleaning import clean_dataset
from sentinel.statistics.business_analysis import run_business_statistics


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    engine = create_database_engine(get_settings().database_url)
    try:
        tables, cleaning = clean_dataset(read_database_tables(engine))
    finally:
        engine.dispose()
    report = {
        "company": "AMX Tech",
        "cleaning": {
            "duplicate_events_removed": cleaning.duplicate_events_removed,
            "industries_filled": cleaning.industries_filled,
            "satisfaction_values_retained_missing": cleaning.satisfaction_values_retained_missing,
        },
        **run_business_statistics(tables),
        "global_limitations": [
            "The data is synthetic and conclusions do not generalize to a real company.",
            "Multiple analyses are reported exploratorily without multiplicity adjustment.",
            "Statistical significance does not imply business importance or causality.",
        ],
    }
    output = Path("data/generated/statistics_report.json")
    output.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"company": "AMX Tech", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
