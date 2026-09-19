"""Compile and run the bounded Phase 13 LangGraph workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from sentinel.agents.nodes import (
    IntentClient,
    make_tool_node,
    make_understand_question_node,
    plan_analysis,
    route_after_plan,
    stop_without_analysis,
    validate_evidence,
)
from sentinel.agents.state import ToolName, WorkflowResult, WorkflowState
from sentinel.agents.tools import AnalyticsToolkit
from sentinel.dashboard import DashboardBundle
from sentinel.llm import QuestionContext, QuestionIntent, create_intent_client


def compile_workflow(
    toolkit: AnalyticsToolkit,
    *,
    intent_client: IntentClient | None = None,
):
    """Compile a fixed graph with no open-ended agent loop or arbitrary tools."""
    builder = StateGraph(WorkflowState)
    builder.add_node("understand_question", make_understand_question_node(intent_client))
    builder.add_node("create_analysis_plan", plan_analysis)
    builder.add_node(
        "business_analysis",
        make_tool_node(toolkit, ToolName.BUSINESS),
    )
    builder.add_node(
        "statistical_analysis",
        make_tool_node(toolkit, ToolName.STATISTICS),
    )
    builder.add_node("ml_analysis", make_tool_node(toolkit, ToolName.ML))
    builder.add_node("forecasting", make_tool_node(toolkit, ToolName.FORECAST))
    builder.add_node(
        "anomaly_analysis",
        make_tool_node(toolkit, ToolName.ANOMALY),
    )
    builder.add_node(
        "data_quality",
        make_tool_node(toolkit, ToolName.DATA_QUALITY),
    )
    builder.add_node("validate_evidence", validate_evidence)
    builder.add_node("stop_without_analysis", stop_without_analysis)

    builder.add_edge(START, "understand_question")
    builder.add_edge("understand_question", "create_analysis_plan")
    destinations = {
        "business_analysis": "business_analysis",
        "statistical_analysis": "statistical_analysis",
        "ml_analysis": "ml_analysis",
        "forecasting": "forecasting",
        "anomaly_analysis": "anomaly_analysis",
        "data_quality": "data_quality",
        "stop_without_analysis": "stop_without_analysis",
    }
    builder.add_conditional_edges("create_analysis_plan", route_after_plan, destinations)
    for node_name in destinations:
        if node_name != "stop_without_analysis":
            builder.add_edge(node_name, "validate_evidence")
    builder.add_edge("validate_evidence", END)
    builder.add_edge("stop_without_analysis", END)
    return builder.compile()


def run_workflow(
    bundle: DashboardBundle,
    question: str,
    *,
    intent: QuestionIntent | None = None,
    intent_client: IntentClient | None = None,
) -> WorkflowResult:
    """Run one bounded question through intent, planning, one tool, and validation."""
    if not question.strip():
        raise ValueError("question must not be blank")
    minimum, maximum = bundle.date_bounds
    context = QuestionContext(
        data_start_date=minimum,
        data_end_date=maximum,
        regions=list(bundle.regions),
        plans=list(bundle.plans),
        industries=sorted(bundle.tables["customers"].industry.dropna().unique()),
    )
    if intent is None and intent_client is None:
        intent_client = create_intent_client()
    toolkit = AnalyticsToolkit(bundle)
    graph = compile_workflow(toolkit, intent_client=intent_client)
    initial: WorkflowState = {
        "question": question.strip(),
        "question_context": context,
        "execution_trace": [],
    }
    if intent is not None:
        initial["intent"] = intent
    final_state = graph.invoke(initial, config={"recursion_limit": 12})
    return WorkflowResult.from_state(final_state)
