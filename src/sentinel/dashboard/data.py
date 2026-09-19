"""Validated data boundary for the Phase 11 Streamlit dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sentinel.config.settings import PROJECT_ROOT, get_settings
from sentinel.data.loader import read_csv_dataset
from sentinel.eda.cleaning import CleaningReport, clean_dataset
from sentinel.eda.exploration import (
    dimension_metrics,
    monthly_churn_trend,
    monthly_costs,
    monthly_revenue,
)
from sentinel.forecasting import build_monthly_net_revenue

ARTIFACT_FILES = {
    "churn_report": "churn_model_report.json",
    "churn_predictions": "churn_test_predictions.csv",
    "forecast_report": "revenue_forecast_report.json",
    "forecast_backtest": "revenue_forecast_backtest.csv",
    "future_forecast": "revenue_forecast.csv",
    "anomaly_report": "cloud_cost_anomaly_report.json",
    "anomaly_scores": "cloud_cost_anomaly_scores.csv",
}


@dataclass(frozen=True)
class DashboardBundle:
    """Clean business tables and versioned analytical artifacts used by the UI."""

    source: str
    tables: dict[str, pd.DataFrame]
    cleaning: CleaningReport
    churn_report: dict[str, Any]
    churn_predictions: pd.DataFrame
    forecast_report: dict[str, Any]
    forecast_backtest: pd.DataFrame
    future_forecast: pd.DataFrame
    anomaly_report: dict[str, Any]
    anomaly_scores: pd.DataFrame
    revenue_history: pd.Series

    @property
    def regions(self) -> tuple[str, ...]:
        return tuple(sorted(self.tables["customers"].region.dropna().unique()))

    @property
    def plans(self) -> tuple[str, ...]:
        return tuple(sorted(self.tables["subscriptions"].plan_type.dropna().unique()))

    @property
    def date_bounds(self) -> tuple[date, date]:
        sales_dates = pd.to_datetime(self.tables["sales"].sale_date)
        final_month_end = sales_dates.max() + pd.offsets.MonthEnd(0)
        return sales_dates.min().date(), final_month_end.date()


@dataclass(frozen=True)
class DashboardView:
    """Filtered, chart-ready frames and headline metrics for one UI state."""

    revenue: pd.DataFrame
    churn: pd.DataFrame
    costs: pd.DataFrame
    region_metrics: pd.DataFrame
    plan_metrics: pd.DataFrame
    net_revenue: float
    revenue_growth: float | None
    latest_churn_rate: float | None
    infrastructure_cost: float
    contribution: float
    cost_to_revenue_ratio: float | None
    revenue_customers: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required dashboard artifact is missing: {path}") from error
    if not isinstance(content, dict):
        raise TypeError(f"Dashboard JSON artifact must contain an object: {path}")
    return content


def _require_artifacts(artifact_dir: Path) -> dict[str, Path]:
    paths = {name: artifact_dir / filename for name, filename in ARTIFACT_FILES.items()}
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            "Dashboard artifacts are incomplete. Run Phases 7-9 first; missing: " + names
        )
    return paths


def _load_source_tables(source: str, project_root: Path) -> dict[str, pd.DataFrame]:
    if source == "csv":
        return read_csv_dataset(project_root / "data" / "generated")
    if source != "database":
        raise ValueError("source must be 'csv' or 'database'")

    from sqlalchemy.exc import SQLAlchemyError

    from sentinel.database.connection import create_database_engine
    from sentinel.database.loader import read_database_tables

    engine = create_database_engine(get_settings().database_url, connect_timeout=3)
    try:
        try:
            return read_database_tables(engine)
        except SQLAlchemyError as error:
            raise RuntimeError("PostgreSQL is unavailable or has not been seeded") from error
    finally:
        engine.dispose()


def load_dashboard_bundle(
    source: str = "csv",
    *,
    project_root: Path = PROJECT_ROOT,
) -> DashboardBundle:
    """Load and validate every source needed by the Phase 11 dashboard."""
    root = Path(project_root)
    tables, cleaning = clean_dataset(_load_source_tables(source, root))
    paths = _require_artifacts(root / "models")
    return DashboardBundle(
        source=source,
        tables=tables,
        cleaning=cleaning,
        churn_report=_read_json(paths["churn_report"]),
        churn_predictions=pd.read_csv(paths["churn_predictions"]),
        forecast_report=_read_json(paths["forecast_report"]),
        forecast_backtest=pd.read_csv(paths["forecast_backtest"]),
        future_forecast=pd.read_csv(paths["future_forecast"]),
        anomaly_report=_read_json(paths["anomaly_report"]),
        anomaly_scores=pd.read_csv(paths["anomaly_scores"]),
        revenue_history=build_monthly_net_revenue(tables["sales"]),
    )


def _as_timestamp(value: date | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _filter_sales(
    sales: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    regions: tuple[str, ...],
    plans: tuple[str, ...],
) -> pd.DataFrame:
    dates = pd.to_datetime(sales.sale_date)
    mask = dates.between(start, end)
    if regions:
        mask &= sales.region.isin(regions)
    if plans:
        mask &= sales.plan_type.isin(plans)
    return sales.loc[mask].copy()


def _filter_subscriptions(
    tables: dict[str, pd.DataFrame],
    regions: tuple[str, ...],
    plans: tuple[str, ...],
) -> pd.DataFrame:
    subscriptions = tables["subscriptions"].copy()
    if regions:
        customer_ids = tables["customers"].loc[
            tables["customers"].region.isin(regions), "customer_id"
        ]
        subscriptions = subscriptions[subscriptions.customer_id.isin(customer_ids)]
    if plans:
        subscriptions = subscriptions[subscriptions.plan_type.isin(plans)]
    return subscriptions.copy()


def _filter_costs(
    costs: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    regions: tuple[str, ...],
) -> pd.DataFrame:
    dates = pd.to_datetime(costs.date)
    mask = dates.between(start, end)
    if regions:
        mask &= costs.region.isin(regions)
    return costs.loc[mask].copy()


def _growth(frame: pd.DataFrame) -> float | None:
    if len(frame) < 2 or frame.net_revenue.iloc[0] == 0:
        return None
    return float(frame.net_revenue.iloc[-1] / frame.net_revenue.iloc[0] - 1)


def _latest_rate(frame: pd.DataFrame) -> float | None:
    values = frame.churn_rate.dropna()
    return float(values.iloc[-1]) if not values.empty else None


def prepare_dashboard_view(
    bundle: DashboardBundle,
    *,
    start_date: date | datetime | pd.Timestamp,
    end_date: date | datetime | pd.Timestamp,
    regions: tuple[str, ...] = (),
    plans: tuple[str, ...] = (),
) -> DashboardView:
    """Build filter-aware metrics without mutating the validated source bundle."""
    start, end = _as_timestamp(start_date), _as_timestamp(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")
    unknown_regions = set(regions) - set(bundle.regions)
    unknown_plans = set(plans) - set(bundle.plans)
    if unknown_regions:
        raise ValueError(f"Unknown regions: {sorted(unknown_regions)}")
    if unknown_plans:
        raise ValueError(f"Unknown plans: {sorted(unknown_plans)}")

    sales = _filter_sales(bundle.tables["sales"], start, end, regions, plans)
    subscriptions = _filter_subscriptions(bundle.tables, regions, plans)
    costs = _filter_costs(bundle.tables["cloud_costs"], start, end, regions)
    if sales.empty or subscriptions.empty or costs.empty:
        raise ValueError("The selected filters do not contain enough data for the dashboard")

    revenue = monthly_revenue(sales)
    churn = monthly_churn_trend(subscriptions)
    month_start = start.to_period("M").to_timestamp()
    month_end = end.to_period("M").to_timestamp()
    churn = churn[churn.month.between(month_start, month_end)].reset_index(drop=True)
    monthly_infrastructure = monthly_costs(costs)

    segment_tables = {name: frame.copy() for name, frame in bundle.tables.items()}
    segment_tables["sales"] = sales
    segment_tables["subscriptions"] = subscriptions
    region_metrics = dimension_metrics(segment_tables, "region")
    plan_metrics = dimension_metrics(segment_tables, "plan_type")

    net_revenue = float(sales.net_revenue.sum())
    infrastructure_cost = float(costs.total_cost.sum())
    return DashboardView(
        revenue=revenue,
        churn=churn,
        costs=monthly_infrastructure,
        region_metrics=region_metrics,
        plan_metrics=plan_metrics,
        net_revenue=net_revenue,
        revenue_growth=_growth(revenue),
        latest_churn_rate=_latest_rate(churn),
        infrastructure_cost=infrastructure_cost,
        contribution=net_revenue - infrastructure_cost,
        cost_to_revenue_ratio=(infrastructure_cost / net_revenue if net_revenue else None),
        revenue_customers=int(sales.customer_id.nunique()),
    )


def filter_churn_predictions(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    regions: tuple[str, ...] = (),
    plans: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return the selected held-out cohort ranked by predicted churn risk."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    required = {"customer_id", "region", "primary_plan", "churn_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Churn predictions are missing columns: {sorted(missing)}")
    frame = predictions.copy()
    if regions:
        frame = frame[frame.region.isin(regions)]
    if plans:
        frame = frame[frame.primary_plan.isin(plans)]
    frame["predicted_churn"] = frame.churn_probability.ge(threshold)
    return frame.sort_values("churn_probability", ascending=False).reset_index(drop=True)
