"""Strict contracts for Phase 12 business-question understanding."""

from __future__ import annotations

import re
from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BusinessIntent(str, Enum):
    PROFITABILITY = "profitability_analysis"
    REVENUE = "revenue_analysis"
    CHURN = "churn_analysis"
    CHURN_RISK = "churn_risk_analysis"
    CLOUD_COST = "cloud_cost_analysis"
    SUPPORT = "support_analysis"
    FORECAST = "forecast_analysis"
    ANOMALY = "anomaly_analysis"
    DATA_QUALITY = "data_quality_analysis"
    GENERAL = "general_business_question"
    UNSUPPORTED = "unsupported"


class BusinessMetric(str, Enum):
    GROSS_REVENUE = "gross_revenue"
    NET_REVENUE = "net_revenue"
    REVENUE_GROWTH = "revenue_growth"
    DISCOUNT_RATE = "discount_rate"
    INFRASTRUCTURE_COST = "infrastructure_cost"
    GPU_COST = "gpu_cost"
    COST_TO_REVENUE = "cost_to_revenue_ratio"
    GROSS_CONTRIBUTION = "estimated_gross_contribution"
    CHURN_RATE = "churn_rate"
    RENEWAL_RATE = "renewal_rate"
    CHURN_PROBABILITY = "churn_probability"
    SUPPORT_RESOLUTION = "support_resolution_time"
    SATISFACTION = "satisfaction_score"
    REVENUE_FORECAST = "revenue_forecast"
    COST_ANOMALY = "cost_anomaly"
    CUSTOMER_COUNT = "customer_count"
    SUBSCRIPTION_COUNT = "subscription_count"
    VALIDATION_ERRORS = "validation_errors"
    VALIDATION_WARNINGS = "validation_warnings"
    DUPLICATE_ROWS = "duplicate_rows"
    MISSING_VALUES = "missing_values"


class AnalysisMode(str, Enum):
    DESCRIPTIVE = "descriptive"
    COMPARATIVE = "comparative"
    DIAGNOSTIC = "diagnostic"
    INFERENTIAL = "inferential"
    PREDICTIVE = "predictive"
    FORECAST = "forecast"
    ANOMALY = "anomaly_detection"


class TimeGranularity(str, Enum):
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    NONE = "not_specified"


class TimePeriod(BaseModel):
    """Resolved period when possible, without inventing unavailable dates."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=100)
    start_date: date | None
    end_date: date | None
    granularity: TimeGranularity
    comparison_label: str | None = Field(max_length=100)
    comparison_start_date: date | None
    comparison_end_date: date | None

    @field_validator("label", "comparison_label")
    @classmethod
    def _strip_labels(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def _validate_ranges(self) -> TimePeriod:
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("start_date and end_date must both be set or both be null")
        if self.start_date is not None and self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        comparison = (self.comparison_start_date, self.comparison_end_date)
        if (comparison[0] is None) != (comparison[1] is None):
            raise ValueError("comparison dates must both be set or both be null")
        if comparison[0] is not None and comparison[0] > comparison[1]:
            raise ValueError("comparison_start_date must not be after comparison_end_date")
        return self


class BusinessFilters(BaseModel):
    """Allowlisted business dimensions extracted from the question."""

    model_config = ConfigDict(extra="forbid")

    regions: list[str] = Field(max_length=4)
    plans: list[str] = Field(max_length=3)
    industries: list[str] = Field(max_length=20)
    customer_ids: list[str] = Field(max_length=50)

    @field_validator("regions", "plans", "industries", "customer_ids")
    @classmethod
    def _unique_values(cls, values: list[str]) -> list[str]:
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            raise ValueError("filter values must not be blank")
        if len(stripped) != len(set(stripped)):
            raise ValueError("filter values must be unique")
        return stripped

    @field_validator("customer_ids")
    @classmethod
    def _validate_customer_ids(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if not re.fullmatch(r"CUS\d{6}", value)]
        if invalid:
            raise ValueError(f"invalid customer IDs: {invalid}")
        return values


class QuestionIntent(BaseModel):
    """The only model-authored output accepted by the Phase 12 application."""

    model_config = ConfigDict(extra="forbid")

    original_question: str = Field(min_length=3, max_length=1_000)
    intent: BusinessIntent
    metrics: list[BusinessMetric] = Field(max_length=12)
    analysis_mode: AnalysisMode
    time_period: TimePeriod
    filters: BusinessFilters
    asks_for_cause: bool
    clarification_needed: bool
    clarification_question: str | None = Field(max_length=300)
    interpretation: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)

    @field_validator("original_question", "interpretation", "clarification_question")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @field_validator("metrics")
    @classmethod
    def _unique_metrics(cls, values: list[BusinessMetric]) -> list[BusinessMetric]:
        if len(values) != len(set(values)):
            raise ValueError("metrics must be unique")
        return values

    @model_validator(mode="after")
    def _validate_clarification(self) -> QuestionIntent:
        if self.clarification_needed and not self.clarification_question:
            raise ValueError("clarification_question is required when clarification is needed")
        if not self.clarification_needed and self.clarification_question is not None:
            raise ValueError("clarification_question must be null when clarification is not needed")
        if self.intent != BusinessIntent.UNSUPPORTED and not self.metrics:
            raise ValueError("supported intents must identify at least one metric")
        return self


class QuestionContext(BaseModel):
    """Trusted dataset context supplied by Sentinel, never authored by the LLM."""

    model_config = ConfigDict(extra="forbid")

    company: str = "AMX Tech"
    data_start_date: date
    data_end_date: date
    regions: list[str]
    plans: list[str]
    industries: list[str]

    @field_validator("regions", "plans", "industries")
    @classmethod
    def _validate_dimensions(cls, values: list[str]) -> list[str]:
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            raise ValueError("context dimensions must not be blank")
        if len(stripped) != len(set(stripped)):
            raise ValueError("context dimensions must be unique")
        return stripped

    @model_validator(mode="after")
    def _validate_dates(self) -> QuestionContext:
        if self.data_start_date > self.data_end_date:
            raise ValueError("data_start_date must not be after data_end_date")
        return self


def validate_intent_context(
    intent: QuestionIntent,
    context: QuestionContext,
    question: str,
) -> QuestionIntent:
    """Enforce trusted dynamic allowlists after model parsing."""
    if intent.original_question != question.strip():
        raise ValueError("structured intent did not preserve the original question")
    dimension_checks = (
        ("regions", intent.filters.regions, context.regions),
        ("plans", intent.filters.plans, context.plans),
        ("industries", intent.filters.industries, context.industries),
    )
    for label, selected, available in dimension_checks:
        unknown = set(selected) - set(available)
        if unknown:
            raise ValueError(f"structured intent contains unknown {label}: {sorted(unknown)}")

    period = intent.time_period
    ranges = (
        (period.start_date, period.end_date),
        (period.comparison_start_date, period.comparison_end_date),
    )
    for start, end in ranges:
        if start is not None and (
            start < context.data_start_date or end > context.data_end_date
        ):
            raise ValueError("structured intent contains dates outside the available data range")
    return intent
