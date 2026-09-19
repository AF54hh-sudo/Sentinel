"""Pure planning, routing, and evidence-validation nodes for Phase 13."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from sentinel.agents.state import (
    AnalysisPlan,
    AnalysisStep,
    ToolName,
    WorkflowState,
    WorkflowStatus,
)
from sentinel.agents.tools import AnalyticsToolkit
from sentinel.llm import (
    AnalysisMode,
    BusinessIntent,
    BusinessMetric,
    IntentResult,
    QuestionContext,
    QuestionIntent,
    validate_intent_context,
)


class IntentClient(Protocol):
    def understand_question(
        self,
        question: str,
        context: QuestionContext,
    ) -> IntentResult: ...


ROUTE_BY_TOOL = {
    ToolName.BUSINESS: "business_analysis",
    ToolName.STATISTICS: "statistical_analysis",
    ToolName.ML: "ml_analysis",
    ToolName.FORECAST: "forecasting",
    ToolName.ANOMALY: "anomaly_analysis",
    ToolName.DATA_QUALITY: "data_quality",
}


def select_tool(intent: QuestionIntent) -> ToolName:
    """Map allowlisted intent to one deterministic analytical branch."""
    if intent.clarification_needed or intent.intent == BusinessIntent.UNSUPPORTED:
        return ToolName.NONE
    if intent.intent == BusinessIntent.CHURN_RISK:
        return ToolName.ML
    if intent.intent == BusinessIntent.FORECAST:
        return ToolName.FORECAST
    if intent.intent == BusinessIntent.ANOMALY:
        return ToolName.ANOMALY
    if intent.intent == BusinessIntent.DATA_QUALITY:
        return ToolName.DATA_QUALITY
    if intent.intent == BusinessIntent.SUPPORT or intent.analysis_mode == AnalysisMode.INFERENTIAL:
        return ToolName.STATISTICS
    return ToolName.BUSINESS


def create_analysis_plan(intent: QuestionIntent) -> AnalysisPlan:
    tool = select_tool(intent)
    if tool == ToolName.NONE:
        reason = (
            intent.clarification_question
            if intent.clarification_needed
            else "The question is outside Sentinel's supported analytical domains."
        )
        return AnalysisPlan(
            selected_tool=ToolName.NONE,
            steps=[],
            rationale=reason or "Clarification is required before analysis.",
            requires_clarification=True,
        )
    filters = intent.filters
    any_segment_filter = bool(
        filters.regions or filters.plans or filters.industries or filters.customer_ids
    )
    entity_filter = bool(filters.industries or filters.customer_ids)
    unsupported_scope: str | None = None
    if tool in {
        ToolName.STATISTICS,
        ToolName.FORECAST,
        ToolName.ANOMALY,
        ToolName.DATA_QUALITY,
    } and any_segment_filter:
        unsupported_scope = f"{tool.value} evidence is not available at the requested segment grain."
    cost_metrics = {
        BusinessMetric.INFRASTRUCTURE_COST,
        BusinessMetric.GPU_COST,
        BusinessMetric.COST_TO_REVENUE,
        BusinessMetric.GROSS_CONTRIBUTION,
        BusinessMetric.COST_ANOMALY,
    }
    if tool == ToolName.BUSINESS and set(intent.metrics) & cost_metrics and (
        filters.plans or entity_filter
    ):
        unsupported_scope = (
            "Shared infrastructure cost cannot be allocated to plans, industries, or customers."
        )
    if unsupported_scope:
        return AnalysisPlan(
            selected_tool=ToolName.NONE,
            steps=[],
            rationale=unsupported_scope,
            requires_clarification=True,
        )
    actions = {
        ToolName.BUSINESS: [
            "Apply the validated time and segment filters to cleaned business facts.",
            "Compute requested metrics and segment comparisons with existing Pandas logic.",
        ],
        ToolName.STATISTICS: [
            "Select the relevant precomputed Phase 6 statistical analyses.",
            "Return test statistics, p-values, effect sizes, assumptions, and limitations.",
        ],
        ToolName.ML: [
            "Apply validated segment filters to held-out churn predictions.",
            "Rank customers at the training-derived threshold and attach held-out metrics.",
        ],
        ToolName.FORECAST: [
            "Load the selected Phase 9 revenue forecast and chronological test evidence.",
            "Return future point estimates with benchmark errors and limitations.",
        ],
        ToolName.ANOMALY: [
            "Filter retrospective anomaly scores to the resolved period.",
            "Return detector-agreement events with robust normal-range evidence.",
        ],
        ToolName.DATA_QUALITY: [
            "Read the post-cleaning relational validation report.",
            "Return row counts, cleaning actions, warnings, and validation status.",
        ],
    }[tool]
    return AnalysisPlan(
        selected_tool=tool,
        steps=[
            AnalysisStep(order=index, action=action, tool=tool)
            for index, action in enumerate(actions, start=1)
        ],
        rationale=(
            f"{intent.intent.value} with {intent.analysis_mode.value} mode maps to the "
            f"approved {tool.value} branch."
        ),
        requires_clarification=False,
    )


def make_understand_question_node(
    intent_client: IntentClient | None,
) -> Callable[[WorkflowState], dict]:
    def understand_question(state: WorkflowState) -> dict:
        if "intent" in state:
            intent = validate_intent_context(
                state["intent"],
                state["question_context"],
                state["question"],
            )
            return {
                "intent": intent,
                "intent_provider": "validated_input",
                "status": WorkflowStatus.READY,
                "execution_trace": ["understand_question"],
            }
        if intent_client is None:
            raise ValueError("An intent client is required when no validated intent is supplied")
        result = intent_client.understand_question(
            state["question"],
            state["question_context"],
        )
        return {
            "intent": result.intent,
            "intent_provider": result.provider,
            "intent_model": result.model,
            "status": WorkflowStatus.READY,
            "execution_trace": ["understand_question"],
        }

    return understand_question


def plan_analysis(state: WorkflowState) -> dict:
    plan = create_analysis_plan(state["intent"])
    return {
        "analysis_plan": plan,
        "selected_tool": plan.selected_tool,
        "execution_trace": ["create_analysis_plan"],
    }


def route_after_plan(
    state: WorkflowState,
) -> Literal[
    "business_analysis",
    "statistical_analysis",
    "ml_analysis",
    "forecasting",
    "anomaly_analysis",
    "data_quality",
    "stop_without_analysis",
]:
    tool = state["selected_tool"]
    return ROUTE_BY_TOOL.get(tool, "stop_without_analysis")


def make_tool_node(
    toolkit: AnalyticsToolkit,
    tool: ToolName,
) -> Callable[[WorkflowState], dict]:
    node_name = ROUTE_BY_TOOL[tool]

    def run_tool(state: WorkflowState) -> dict:
        evidence = toolkit.run(tool, state["intent"])
        return {"evidence": evidence, "execution_trace": [node_name]}

    return run_tool


def validate_evidence(state: WorkflowState) -> dict:
    evidence = state.get("evidence")
    if evidence is None:
        raise ValueError("Analytical branch returned no evidence")
    if evidence.tool != state["selected_tool"]:
        raise ValueError("Evidence tool does not match the selected analytical branch")
    if not evidence.metrics or not evidence.sources:
        raise ValueError("Evidence requires at least one metric and one source")
    verified = evidence.model_copy(update={"verified": True})
    return {
        "evidence": verified,
        "status": WorkflowStatus.COMPLETED,
        "limitations": verified.limitations,
        "execution_trace": ["validate_evidence"],
    }


def stop_without_analysis(state: WorkflowState) -> dict:
    intent = state["intent"]
    if intent.intent == BusinessIntent.UNSUPPORTED:
        status = WorkflowStatus.UNSUPPORTED
        limitation = "The question is outside Sentinel's supported analytical domains."
    else:
        status = WorkflowStatus.NEEDS_CLARIFICATION
        limitation = (
            intent.clarification_question
            or state["analysis_plan"].rationale
            or "Clarification is required before analysis."
        )
    return {
        "status": status,
        "limitations": [limitation],
        "execution_trace": ["stop_without_analysis"],
    }
