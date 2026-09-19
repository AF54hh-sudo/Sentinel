"""Deterministic analytical tools available to the Phase 13 graph."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from sentinel.agents.state import AnalysisEvidence, EvidenceMetric, ToolName
from sentinel.config.settings import PROJECT_ROOT
from sentinel.dashboard import DashboardBundle, filter_churn_predictions, prepare_dashboard_view
from sentinel.eda.exploration import business_metrics
from sentinel.llm import BusinessMetric, QuestionIntent


def _metric(name: str, value: float | str | bool, unit: str) -> EvidenceMetric:
    return EvidenceMetric(name=name, value=value, unit=unit)


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


class AnalyticsToolkit:
    """Expose approved analytical outputs without arbitrary SQL or Python execution."""

    def __init__(
        self,
        bundle: DashboardBundle,
        *,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        self.bundle = bundle
        self.project_root = Path(project_root)

    def run(self, tool: ToolName, intent: QuestionIntent) -> AnalysisEvidence:
        methods = {
            ToolName.BUSINESS: self._business_analysis,
            ToolName.STATISTICS: self._statistical_analysis,
            ToolName.ML: self._ml_analysis,
            ToolName.FORECAST: self._forecast_analysis,
            ToolName.ANOMALY: self._anomaly_analysis,
            ToolName.DATA_QUALITY: self._data_quality_analysis,
        }
        if tool not in methods:
            raise ValueError(f"No analytical implementation for tool: {tool.value}")
        return methods[tool](intent)

    def _period(self, intent: QuestionIntent) -> tuple[pd.Timestamp, pd.Timestamp]:
        minimum, maximum = self.bundle.date_bounds
        period = intent.time_period
        return (
            pd.Timestamp(period.start_date or minimum),
            pd.Timestamp(period.end_date or maximum),
        )

    def _scoped_customer_ids(self, intent: QuestionIntent) -> set[str] | None:
        if not intent.filters.industries and not intent.filters.customer_ids:
            return None
        customers = self.bundle.tables["customers"]
        known_ids = set(customers.customer_id)
        requested_ids = set(intent.filters.customer_ids)
        unknown_ids = requested_ids - known_ids
        if unknown_ids:
            raise ValueError(f"Unknown customer IDs: {sorted(unknown_ids)}")
        selected = known_ids
        if intent.filters.industries:
            selected &= set(
                customers.loc[
                    customers.industry.isin(intent.filters.industries),
                    "customer_id",
                ]
            )
        if requested_ids:
            selected &= requested_ids
        if not selected:
            raise ValueError("The requested industry/customer filters contain no customers")
        return selected

    def _business_bundle(self, intent: QuestionIntent) -> DashboardBundle:
        customer_ids = self._scoped_customer_ids(intent)
        if customer_ids is None:
            return self.bundle
        tables = {name: frame.copy() for name, frame in self.bundle.tables.items()}
        tables["customers"] = tables["customers"][
            tables["customers"].customer_id.isin(customer_ids)
        ].copy()
        for name in ("subscriptions", "sales", "support_tickets"):
            tables[name] = tables[name][tables[name].customer_id.isin(customer_ids)].copy()
        return replace(self.bundle, tables=tables)

    def _business_analysis(self, intent: QuestionIntent) -> AnalysisEvidence:
        start, end = self._period(intent)
        analysis_bundle = self._business_bundle(intent)
        view = prepare_dashboard_view(
            analysis_bundle,
            start_date=start,
            end_date=end,
            regions=tuple(intent.filters.regions),
            plans=tuple(intent.filters.plans),
        )
        requested = set(intent.metrics)
        weighted_discount = float(
            view.revenue.discount_amount.sum() / view.revenue.gross_revenue.sum()
        )
        values: dict[BusinessMetric, tuple[int | float | str | bool, str]] = {
            BusinessMetric.GROSS_REVENUE: (float(view.revenue.gross_revenue.sum()), "currency"),
            BusinessMetric.NET_REVENUE: (view.net_revenue, "currency"),
            BusinessMetric.REVENUE_GROWTH: (view.revenue_growth or 0.0, "ratio"),
            BusinessMetric.DISCOUNT_RATE: (weighted_discount, "ratio"),
            BusinessMetric.INFRASTRUCTURE_COST: (view.infrastructure_cost, "currency"),
            BusinessMetric.GPU_COST: (float(view.costs.gpu_cost.sum()), "currency"),
            BusinessMetric.COST_TO_REVENUE: (view.cost_to_revenue_ratio or 0.0, "ratio"),
            BusinessMetric.GROSS_CONTRIBUTION: (view.contribution, "currency"),
            BusinessMetric.CHURN_RATE: (view.latest_churn_rate or 0.0, "ratio"),
            BusinessMetric.CUSTOMER_COUNT: (view.revenue_customers, "customers"),
            BusinessMetric.SUBSCRIPTION_COUNT: (
                int(view.churn.subscriptions_at_risk.max()),
                "subscriptions",
            ),
        }
        full_metrics = business_metrics(analysis_bundle.tables)
        values[BusinessMetric.RENEWAL_RATE] = (float(full_metrics["renewal_rate"]), "ratio")

        tickets = analysis_bundle.tables["support_tickets"].copy()
        customer_ids = analysis_bundle.tables["customers"].customer_id
        if intent.filters.regions:
            customer_ids = analysis_bundle.tables["customers"].loc[
                analysis_bundle.tables["customers"].region.isin(intent.filters.regions),
                "customer_id",
            ]
        if intent.filters.plans:
            plan_customers = analysis_bundle.tables["subscriptions"].loc[
                analysis_bundle.tables["subscriptions"].plan_type.isin(intent.filters.plans),
                "customer_id",
            ]
            customer_ids = customer_ids[customer_ids.isin(plan_customers)]
        ticket_dates = pd.to_datetime(tickets.created_date)
        tickets = tickets[
            tickets.customer_id.isin(customer_ids) & ticket_dates.between(start, end)
        ]
        if not tickets.empty:
            values[BusinessMetric.SUPPORT_RESOLUTION] = (
                float(tickets.resolution_hours.mean()),
                "hours",
            )
            satisfaction = tickets.satisfaction_score.dropna()
            if not satisfaction.empty:
                values[BusinessMetric.SATISFACTION] = (float(satisfaction.mean()), "score_1_to_5")

        metrics = [
            _metric(metric.value, *values[metric])
            for metric in intent.metrics
            if metric in values
        ]
        if not metrics:
            raise ValueError("The business tool cannot compute the requested metrics")

        records: list[dict[str, Any]] = []
        if not intent.filters.regions:
            records.extend(
                {
                    "dimension": "region",
                    "segment": row.region,
                    "net_revenue": float(row.net_revenue),
                    "churn_rate": float(row.churn_rate),
                }
                for row in view.region_metrics.itertuples(index=False)
            )
        if not intent.filters.plans:
            records.extend(
                {
                    "dimension": "plan_type",
                    "segment": row.plan_type,
                    "net_revenue": float(row.net_revenue),
                    "churn_rate": float(row.churn_rate),
                }
                for row in view.plan_metrics.itertuples(index=False)
            )
        limitations = [
            "The dataset is synthetic and does not establish real-world business performance.",
            "Estimated gross contribution subtracts infrastructure cost only, not every operating expense.",
        ]
        if BusinessMetric.RENEWAL_RATE in requested and (
            intent.filters.regions or intent.filters.plans or intent.time_period.start_date
        ):
            limitations.append(
                "Renewal rate uses the full observed subscription outcome window, not the selected filters."
            )
        return AnalysisEvidence(
            tool=ToolName.BUSINESS,
            title="Validated business analytics",
            summary=(
                f"Computed {len(metrics)} requested metrics for {start.date()} through {end.date()} "
                "from cleaned AMX Tech business tables."
            ),
            metrics=metrics,
            records=records,
            sources=[f"{self.bundle.source}: cleaned AMX Tech relational tables"],
            limitations=limitations,
        )

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _statistical_analysis(self, intent: QuestionIntent) -> AnalysisEvidence:
        report = self._read_json(self.project_root / "data" / "generated" / "statistics_report.json")
        support_metrics = {
            BusinessMetric.SUPPORT_RESOLUTION,
            BusinessMetric.SATISFACTION,
        }
        if set(intent.metrics) & support_metrics:
            names = [
                "support_resolution_welch",
                "support_resolution_mann_whitney",
                "support_satisfaction_spearman",
            ]
        else:
            names = ["plan_cancellation_chi_square"]
        selected = [(name, report["tests"][name]) for name in names]
        metrics: list[EvidenceMetric] = []
        records: list[dict[str, Any]] = []
        limitations = list(report["global_limitations"])
        for name, test in selected:
            metrics.extend(
                [
                    _metric(f"{name}.statistic", float(test["statistic"]), "test_statistic"),
                    _metric(f"{name}.p_value", float(test["p_value"]), "probability"),
                    _metric(
                        f"{name}.effect_size",
                        float(test["effect_size"]),
                        test["effect_size_name"],
                    ),
                ]
            )
            records.append(
                {
                    "test": name,
                    "method": test["test_used"],
                    "result": test["result"],
                    "interpretation": test["business_interpretation"],
                }
            )
            limitations.extend(test["limitations"])
        return AnalysisEvidence(
            tool=ToolName.STATISTICS,
            title="Validated statistical evidence",
            summary=f"Retrieved {len(selected)} precomputed, assumption-documented statistical analyses.",
            metrics=metrics,
            records=records,
            sources=["data/generated/statistics_report.json", "Phase 6 statistical inference"],
            limitations=_deduplicate(limitations),
        )

    def _ml_analysis(self, intent: QuestionIntent) -> AnalysisEvidence:
        report = self.bundle.churn_report
        threshold = float(report["threshold_selection"]["selected_threshold"])
        predictions = filter_churn_predictions(
            self.bundle.churn_predictions,
            threshold=threshold,
            regions=tuple(intent.filters.regions),
            plans=tuple(intent.filters.plans),
        )
        customer_ids = self._scoped_customer_ids(intent)
        if customer_ids is not None:
            predictions = predictions[predictions.customer_id.isin(customer_ids)].reset_index(
                drop=True
            )
        flagged = predictions[predictions.predicted_churn]
        held_out = report["held_out_test_metrics"]
        records = [
            {
                "customer_id": row.customer_id,
                "region": row.region,
                "plan": row.primary_plan,
                "churn_probability": float(row.churn_probability),
            }
            for row in flagged.head(25).itertuples(index=False)
        ]
        return AnalysisEvidence(
            tool=ToolName.ML,
            title="Held-out churn-risk evidence",
            summary=(
                f"Ranked {len(predictions)} visible held-out customers and flagged {len(flagged)} "
                "at the training-derived operating threshold."
            ),
            metrics=[
                _metric("visible_customers", len(predictions), "customers"),
                _metric("flagged_customers", len(flagged), "customers"),
                _metric("operating_threshold", threshold, "probability"),
                _metric("held_out_average_precision", float(held_out["average_precision"]), "score"),
                _metric("held_out_recall", float(held_out["recall"]), "ratio"),
                _metric("held_out_precision", float(held_out["precision"]), "ratio"),
            ],
            records=records,
            sources=["models/churn_model_report.json", "models/churn_test_predictions.csv"],
            limitations=list(report["limitations"]),
        )

    def _forecast_analysis(self, intent: QuestionIntent) -> AnalysisEvidence:
        report = self.bundle.forecast_report
        selected = report["selected_test_metrics"]
        records = [
            {
                "month": row.month,
                "horizon_month": int(row.horizon_month),
                "forecast_net_revenue": float(row.forecast_net_revenue),
            }
            for row in self.bundle.future_forecast.itertuples(index=False)
        ]
        first = records[0]
        return AnalysisEvidence(
            tool=ToolName.FORECAST,
            title="Revenue forecast evidence",
            summary=(
                f"Returned the selected {report['selected_method'].replace('_', ' ')} point forecast "
                f"beginning {first['month']}."
            ),
            metrics=[
                _metric("next_month_net_revenue", first["forecast_net_revenue"], "currency"),
                _metric("test_mae", float(selected["mae"]), "currency"),
                _metric("test_rmse", float(selected["rmse"]), "currency"),
                _metric("test_wape", float(selected["wape"]), "ratio"),
                _metric(
                    "test_wape_improvement_over_naive",
                    float(report["test_wape_improvement_over_naive"]),
                    "ratio",
                ),
            ],
            records=records,
            sources=["models/revenue_forecast_report.json", "models/revenue_forecast.csv"],
            limitations=list(report["limitations"]),
        )

    def _anomaly_analysis(self, intent: QuestionIntent) -> AnalysisEvidence:
        report = self.bundle.anomaly_report
        start, end = self._period(intent)
        scores = self.bundle.anomaly_scores.copy()
        dates = pd.to_datetime(scores.date)
        strong = scores[
            dates.between(start, end)
            & scores.strong_anomaly.astype(str).str.lower().eq("true")
        ]
        records = [
            {
                "date": row.date,
                "observed_value": float(row.observed_value),
                "normal_range_lower": float(row.normal_range_lower),
                "normal_range_upper": float(row.normal_range_upper),
                "anomaly_score": float(row.anomaly_score),
            }
            for row in strong.itertuples(index=False)
        ]
        return AnalysisEvidence(
            tool=ToolName.ANOMALY,
            title="GPU-cost anomaly evidence",
            summary=(
                f"Found {len(strong)} strong consensus anomalies between {start.date()} and {end.date()}."
            ),
            metrics=[
                _metric("strong_anomalies_in_period", len(strong), "days"),
                _metric("all_iqr_candidates", int(report["counts"]["iqr_candidates"]), "days"),
                _metric(
                    "all_isolation_forest_anomalies",
                    int(report["counts"]["isolation_forest_anomalies"]),
                    "days",
                ),
            ],
            records=records,
            sources=["models/cloud_cost_anomaly_report.json", "models/cloud_cost_anomaly_scores.csv"],
            limitations=list(report["limitations"]),
        )

    def _data_quality_analysis(self, intent: QuestionIntent) -> AnalysisEvidence:
        cleaning = self.bundle.cleaning
        validation = cleaning.validation
        if validation is None:
            raise ValueError("Dashboard bundle has no validation report")
        records = [
            {"table": table, "rows": count}
            for table, count in validation.row_counts.items()
        ]
        return AnalysisEvidence(
            tool=ToolName.DATA_QUALITY,
            title="Data-quality evidence",
            summary="Reported cleaning actions and post-cleaning relational validation status.",
            metrics=[
                _metric("validation_errors", len(validation.errors), "checks"),
                _metric("validation_warnings", len(validation.warnings), "checks"),
                _metric("duplicate_events_removed", cleaning.duplicate_events_removed, "rows"),
                _metric("industries_filled", cleaning.industries_filled, "rows"),
                _metric(
                    "missing_satisfaction_retained",
                    cleaning.satisfaction_values_retained_missing,
                    "rows",
                ),
            ],
            records=records,
            sources=[f"{self.bundle.source}: Phase 4 cleaning and validation report"],
            limitations=[
                "Validation checks known contracts; they cannot prove every business value is correct.",
                "Some missing dates and satisfaction values are structural or intentionally retained.",
            ],
        )
