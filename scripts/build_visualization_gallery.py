"""Build the AMX Tech Phase 10 reusable Plotly chart gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from sentinel.config import get_settings
from sentinel.data.loader import read_csv_dataset
from sentinel.database.connection import create_database_engine
from sentinel.database.loader import read_database_tables
from sentinel.eda.cleaning import clean_dataset
from sentinel.eda.exploration import (
    dimension_metrics,
    monthly_churn_trend,
    monthly_costs,
    monthly_revenue,
)
from sentinel.forecasting import build_monthly_net_revenue
from sentinel.visualization import (
    anomaly_evidence_chart,
    churn_model_evaluation_chart,
    churn_trend_chart,
    cloud_cost_chart,
    export_chart_gallery,
    forecast_evidence_chart,
    revenue_trend_chart,
    segment_performance_chart,
)


def _load_tables(source: str) -> dict[str, Any]:
    if source == "csv":
        return read_csv_dataset(Path("data/generated"))
    engine = create_database_engine(get_settings().database_url)
    try:
        return read_database_tables(engine)
    finally:
        engine.dispose()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required analytical artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(source: str = "database") -> int:
    tables, cleaning = clean_dataset(_load_tables(source))
    artifact_dir = Path("models")
    required_csvs = {
        "anomaly": artifact_dir / "cloud_cost_anomaly_scores.csv",
        "future": artifact_dir / "revenue_forecast.csv",
        "backtest": artifact_dir / "revenue_forecast_backtest.csv",
    }
    missing = [path for path in required_csvs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Generate Phase 8 and Phase 9 artifacts first: "
            + ", ".join(str(path) for path in missing)
        )

    anomaly_scores = pd.read_csv(required_csvs["anomaly"])
    future = pd.read_csv(required_csvs["future"])
    backtest = pd.read_csv(required_csvs["backtest"])
    churn_report = _read_json(artifact_dir / "churn_model_report.json")
    forecast_report = _read_json(artifact_dir / "revenue_forecast_report.json")
    history = build_monthly_net_revenue(tables["sales"])

    figures = {
        "monthly-revenue": revenue_trend_chart(monthly_revenue(tables["sales"])),
        "monthly-churn": churn_trend_chart(monthly_churn_trend(tables["subscriptions"])),
        "cloud-cost-composition": cloud_cost_chart(monthly_costs(tables["cloud_costs"])),
        "region-performance": segment_performance_chart(
            dimension_metrics(tables, "region"), "region"
        ),
        "plan-performance": segment_performance_chart(
            dimension_metrics(tables, "plan_type"), "plan_type"
        ),
        "gpu-anomaly-evidence": anomaly_evidence_chart(anomaly_scores),
        "churn-model-evaluation": churn_model_evaluation_chart(churn_report),
        "revenue-forecast": forecast_evidence_chart(
            history,
            backtest,
            future,
            selected_method=forecast_report["selected_method"],
        ),
    }
    output_dir = artifact_dir / "visualizations"
    outputs = export_chart_gallery(figures, output_dir)
    manifest = {
        "company": "AMX Tech",
        "source": source,
        "chart_count": len(figures),
        "charts": {name: str(path) for name, path in outputs.items()},
        "duplicates_removed": cleaning.duplicate_events_removed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("database", "csv"), default="database")
    raise SystemExit(main(parser.parse_args().source))
