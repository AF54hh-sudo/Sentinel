"""Run and persist the AMX Tech Phase 9 monthly net-revenue forecast."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sentinel.config import get_settings
from sentinel.data.loader import read_csv_dataset
from sentinel.database.connection import create_database_engine
from sentinel.database.loader import read_database_tables
from sentinel.eda.cleaning import clean_dataset
from sentinel.forecasting import RevenueForecastConfig, run_revenue_forecast


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


def _load_tables(source: str) -> dict[str, Any]:
    if source == "csv":
        return read_csv_dataset(Path("data/generated"))
    engine = create_database_engine(get_settings().database_url)
    try:
        return read_database_tables(engine)
    finally:
        engine.dispose()


def main(source: str = "database") -> int:
    tables, cleaning = clean_dataset(_load_tables(source))
    config = RevenueForecastConfig()
    future, backtest, fitted_model, report = run_revenue_forecast(
        tables["sales"], config
    )

    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_schema_version": 1,
        "company": "AMX Tech",
        "model": fitted_model,
        "selected_method": report["selected_method"],
        "configuration": asdict(config),
        "series_contract": report["series"],
        "intended_use": (
            "Forecast AMX Tech monthly net revenue from a complete month-start history."
        ),
    }
    joblib.dump(artifact, output_dir / "revenue_forecast_model.joblib")
    (output_dir / "revenue_forecast_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    future.to_csv(output_dir / "revenue_forecast.csv", index=False)
    backtest.to_csv(output_dir / "revenue_forecast_backtest.csv", index=False)

    print(
        json.dumps(
            {
                "company": "AMX Tech",
                "selected_method": report["selected_method"],
                "test_mae": report["selected_test_metrics"]["mae"],
                "test_rmse": report["selected_test_metrics"]["rmse"],
                "test_wape": report["selected_test_metrics"]["wape"],
                "test_wape_improvement_over_naive": report[
                    "test_wape_improvement_over_naive"
                ],
                "next_month_forecast": report["future_forecast"][0],
                "source": source,
                "output_dir": str(output_dir),
                "duplicates_removed": cleaning.duplicate_events_removed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("database", "csv"), default="database")
    raise SystemExit(main(parser.parse_args().source))
