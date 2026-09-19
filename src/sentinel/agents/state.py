"""Typed state and evidence contracts for the bounded Phase 13 workflow."""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, NotRequired, Required, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sentinel.llm import QuestionContext, QuestionIntent


class ToolName(str, Enum):
    BUSINESS = "business_analytics"
    STATISTICS = "statistical_analysis"
    ML = "churn_model"
    FORECAST = "revenue_forecast"
    ANOMALY = "anomaly_detection"
    DATA_QUALITY = "data_quality"
    NONE = "none"


class WorkflowStatus(str, Enum):
    READY = "ready"
    COMPLETED = "completed"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class AnalysisStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1, le=10)
    action: str = Field(min_length=3, max_length=240)
    tool: ToolName


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_tool: ToolName
    steps: list[AnalysisStep] = Field(max_length=5)
    rationale: str = Field(min_length=3, max_length=400)
    requires_clarification: bool

    @model_validator(mode="after")
    def _validate_tool_steps(self) -> AnalysisPlan:
        if self.requires_clarification and self.selected_tool != ToolName.NONE:
            raise ValueError("clarification plans must not select an analytical tool")
        if not self.requires_clarification and self.selected_tool == ToolName.NONE:
            raise ValueError("executable plans must select an analytical tool")
        if any(step.tool != self.selected_tool for step in self.steps):
            raise ValueError("every plan step must use the selected tool")
        return self


class EvidenceMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    value: int | float | str | bool
    unit: str = Field(min_length=1, max_length=60)


class AnalysisEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: ToolName
    title: str = Field(min_length=3, max_length=200)
    summary: str = Field(min_length=3, max_length=500)
    metrics: list[EvidenceMetric] = Field(min_length=1, max_length=60)
    records: list[dict[str, Any]] = Field(max_length=100)
    sources: list[str] = Field(min_length=1, max_length=20)
    limitations: list[str] = Field(max_length=20)
    verified: bool = False


class WorkflowState(TypedDict, total=False):
    question: Required[str]
    question_context: Required[QuestionContext]
    intent: NotRequired[QuestionIntent]
    intent_provider: NotRequired[str]
    intent_model: NotRequired[str]
    analysis_plan: NotRequired[AnalysisPlan]
    selected_tool: NotRequired[ToolName]
    evidence: NotRequired[AnalysisEvidence]
    status: NotRequired[WorkflowStatus]
    limitations: NotRequired[list[str]]
    error: NotRequired[str]
    execution_trace: Annotated[list[str], operator.add]


class WorkflowResult(BaseModel):
    """Stable output returned by the graph wrapper and rendered by Streamlit."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    question: str
    intent: QuestionIntent
    analysis_plan: AnalysisPlan
    selected_tool: ToolName
    evidence: AnalysisEvidence | None
    status: WorkflowStatus
    limitations: list[str]
    execution_trace: list[str]
    intent_provider: str | None
    intent_model: str | None

    @classmethod
    def from_state(cls, state: WorkflowState) -> WorkflowResult:
        required = ("intent", "analysis_plan", "selected_tool", "status")
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(f"Workflow state is incomplete: {missing}")
        return cls(
            question=state["question"],
            intent=state["intent"],
            analysis_plan=state["analysis_plan"],
            selected_tool=state["selected_tool"],
            evidence=state.get("evidence"),
            status=state["status"],
            limitations=state.get("limitations", []),
            execution_trace=state.get("execution_trace", []),
            intent_provider=state.get("intent_provider"),
            intent_model=state.get("intent_model"),
        )
