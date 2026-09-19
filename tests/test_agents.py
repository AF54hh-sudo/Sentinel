from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from sentinel.agents import (
    AnalyticsToolkit,
    ToolName,
    WorkflowStatus,
    compile_workflow,
    create_analysis_plan,
    run_workflow,
    select_tool,
)
from sentinel.dashboard import load_dashboard_bundle
from sentinel.llm import IntentResult, QuestionContext, QuestionIntent


@pytest.fixture(scope="module")
def dashboard_bundle():
    return load_dashboard_bundle("csv")


def _intent(
    *,
    question: str,
    intent: str,
    metrics: list[str],
    mode: str,
    start_date: str | None = None,
    end_date: str | None = None,
    regions: list[str] | None = None,
    plans: list[str] | None = None,
    industries: list[str] | None = None,
    customer_ids: list[str] | None = None,
    clarification_needed: bool = False,
) -> QuestionIntent:
    clarification_question = "Which period should Sentinel analyse?" if clarification_needed else None
    return QuestionIntent.model_validate(
        {
            "original_question": question,
            "intent": intent,
            "metrics": metrics,
            "analysis_mode": mode,
            "time_period": {
                "label": "Selected period" if start_date else "Not specified",
                "start_date": start_date,
                "end_date": end_date,
                "granularity": "month" if start_date else "not_specified",
                "comparison_label": None,
                "comparison_start_date": None,
                "comparison_end_date": None,
            },
            "filters": {
                "regions": regions or [],
                "plans": plans or [],
                "industries": industries or [],
                "customer_ids": customer_ids or [],
            },
            "asks_for_cause": "Why" in question or "affect" in question,
            "clarification_needed": clarification_needed,
            "clarification_question": clarification_question,
            "interpretation": "Validated test intent.",
            "confidence": 0.95,
        }
    )


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        (
            _intent(
                question="Show Q2 revenue",
                intent="revenue_analysis",
                metrics=["net_revenue"],
                mode="descriptive",
            ),
            ToolName.BUSINESS,
        ),
        (
            _intent(
                question="Did support quality affect churn?",
                intent="support_analysis",
                metrics=["support_resolution_time", "satisfaction_score", "churn_rate"],
                mode="inferential",
            ),
            ToolName.STATISTICS,
        ),
        (
            _intent(
                question="Which customers have highest churn risk?",
                intent="churn_risk_analysis",
                metrics=["churn_probability"],
                mode="predictive",
            ),
            ToolName.ML,
        ),
        (
            _intent(
                question="Forecast next month revenue",
                intent="forecast_analysis",
                metrics=["revenue_forecast"],
                mode="forecast",
            ),
            ToolName.FORECAST,
        ),
        (
            _intent(
                question="What caused the cloud cost spike?",
                intent="anomaly_analysis",
                metrics=["cost_anomaly", "gpu_cost"],
                mode="anomaly_detection",
            ),
            ToolName.ANOMALY,
        ),
        (
            _intent(
                question="Are there data quality problems?",
                intent="data_quality_analysis",
                metrics=["validation_errors", "duplicate_rows", "missing_values"],
                mode="descriptive",
            ),
            ToolName.DATA_QUALITY,
        ),
    ],
)
def test_tool_selection_is_deterministic(intent, expected) -> None:
    assert select_tool(intent) == expected
    plan = create_analysis_plan(intent)
    assert plan.selected_tool == expected
    assert plan.requires_clarification is False
    assert all(step.tool == expected for step in plan.steps)


@pytest.mark.parametrize(
    ("intent", "expected_tool", "metric_name", "branch"),
    [
        (
            _intent(
                question="Show Q2 revenue",
                intent="revenue_analysis",
                metrics=["net_revenue"],
                mode="descriptive",
                start_date="2024-04-01",
                end_date="2024-06-30",
            ),
            ToolName.BUSINESS,
            "net_revenue",
            "business_analysis",
        ),
        (
            _intent(
                question="Did support quality affect churn?",
                intent="support_analysis",
                metrics=["support_resolution_time", "satisfaction_score", "churn_rate"],
                mode="inferential",
            ),
            ToolName.STATISTICS,
            "support_resolution_welch.p_value",
            "statistical_analysis",
        ),
        (
            _intent(
                question="Which customers have highest churn risk?",
                intent="churn_risk_analysis",
                metrics=["churn_probability"],
                mode="predictive",
                regions=["Europe"],
                plans=["Professional"],
            ),
            ToolName.ML,
            "flagged_customers",
            "ml_analysis",
        ),
        (
            _intent(
                question="Forecast next month revenue",
                intent="forecast_analysis",
                metrics=["revenue_forecast"],
                mode="forecast",
            ),
            ToolName.FORECAST,
            "next_month_net_revenue",
            "forecasting",
        ),
        (
            _intent(
                question="What caused the May cloud cost spike?",
                intent="anomaly_analysis",
                metrics=["cost_anomaly", "gpu_cost"],
                mode="anomaly_detection",
                start_date="2024-05-01",
                end_date="2024-05-31",
            ),
            ToolName.ANOMALY,
            "strong_anomalies_in_period",
            "anomaly_analysis",
        ),
        (
            _intent(
                question="Are there data quality problems?",
                intent="data_quality_analysis",
                metrics=["validation_errors", "duplicate_rows", "missing_values"],
                mode="descriptive",
            ),
            ToolName.DATA_QUALITY,
            "validation_errors",
            "data_quality",
        ),
    ],
)
def test_graph_executes_one_approved_branch(
    dashboard_bundle,
    intent,
    expected_tool,
    metric_name,
    branch,
) -> None:
    result = run_workflow(dashboard_bundle, intent.original_question, intent=intent)

    assert result.status == WorkflowStatus.COMPLETED
    assert result.selected_tool == expected_tool
    assert result.evidence is not None and result.evidence.verified is True
    assert any(metric.name == metric_name for metric in result.evidence.metrics)
    assert result.execution_trace == [
        "understand_question",
        "create_analysis_plan",
        branch,
        "validate_evidence",
    ]


def test_business_graph_metric_reconciles_source_data(dashboard_bundle) -> None:
    intent = _intent(
        question="Show Q2 revenue",
        intent="revenue_analysis",
        metrics=["net_revenue"],
        mode="descriptive",
        start_date="2024-04-01",
        end_date="2024-06-30",
    )
    result = run_workflow(dashboard_bundle, intent.original_question, intent=intent)
    actual = next(metric.value for metric in result.evidence.metrics if metric.name == "net_revenue")
    sales = dashboard_bundle.tables["sales"]
    expected = sales.loc[
        pd.to_datetime(sales.sale_date).between("2024-04-01", "2024-06-30"),
        "net_revenue",
    ].sum()
    assert actual == pytest.approx(expected)


def test_industry_scope_is_applied_to_churn_risk(dashboard_bundle) -> None:
    intent = _intent(
        question="Which Finance customers have highest churn risk?",
        intent="churn_risk_analysis",
        metrics=["churn_probability"],
        mode="predictive",
        industries=["Finance"],
    )
    result = run_workflow(dashboard_bundle, intent.original_question, intent=intent)
    finance_ids = set(
        dashboard_bundle.tables["customers"].loc[
            dashboard_bundle.tables["customers"].industry.eq("Finance"), "customer_id"
        ]
    )
    assert all(record["customer_id"] in finance_ids for record in result.evidence.records)


@pytest.mark.parametrize(
    "intent",
    [
        _intent(
            question="Which period had higher churn?",
            intent="churn_analysis",
            metrics=["churn_rate"],
            mode="comparative",
            clarification_needed=True,
        ),
        _intent(
            question="Forecast Professional plan revenue",
            intent="forecast_analysis",
            metrics=["revenue_forecast"],
            mode="forecast",
            plans=["Professional"],
        ),
        _intent(
            question="Show Professional plan GPU cost",
            intent="cloud_cost_analysis",
            metrics=["gpu_cost"],
            mode="descriptive",
            plans=["Professional"],
        ),
    ],
)
def test_graph_stops_when_scope_is_ambiguous_or_unsupported(dashboard_bundle, intent) -> None:
    result = run_workflow(dashboard_bundle, intent.original_question, intent=intent)

    assert result.status == WorkflowStatus.NEEDS_CLARIFICATION
    assert result.selected_tool == ToolName.NONE
    assert result.evidence is None
    assert result.execution_trace[-1] == "stop_without_analysis"


class StaticIntentClient:
    def __init__(self, intent: QuestionIntent) -> None:
        self.intent = intent
        self.calls = 0

    def understand_question(self, question: str, context: QuestionContext) -> IntentResult:
        self.calls += 1
        assert question == self.intent.original_question
        assert context.data_end_date == date(2024, 12, 31)
        return IntentResult(
            intent=self.intent,
            provider="test-provider",
            model="test-model",
            response_id="test-response",
        )


def test_graph_can_obtain_intent_from_injected_client(dashboard_bundle) -> None:
    intent = _intent(
        question="Forecast next month revenue",
        intent="forecast_analysis",
        metrics=["revenue_forecast"],
        mode="forecast",
    )
    client = StaticIntentClient(intent)
    result = run_workflow(
        dashboard_bundle,
        intent.original_question,
        intent_client=client,
    )

    assert client.calls == 1
    assert result.intent_provider == "test-provider"
    assert result.intent_model == "test-model"
    assert result.status == WorkflowStatus.COMPLETED


def test_compiled_graph_has_only_bounded_phase_13_nodes(dashboard_bundle) -> None:
    graph = compile_workflow(AnalyticsToolkit(dashboard_bundle))
    nodes = set(graph.get_graph().nodes)

    assert nodes == {
        "__start__",
        "understand_question",
        "create_analysis_plan",
        "business_analysis",
        "statistical_analysis",
        "ml_analysis",
        "forecasting",
        "anomaly_analysis",
        "data_quality",
        "validate_evidence",
        "stop_without_analysis",
        "__end__",
    }
    assert not any("rag" in node or "sql_generation" in node for node in nodes)
