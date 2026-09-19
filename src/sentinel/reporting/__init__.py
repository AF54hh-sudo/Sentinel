"""Evidence-grounded business reporting."""

from sentinel.reporting.client import (
    NarrativeResult,
    OpenAIReportNarrator,
    ReportNarrator,
    ReportRequestError,
    ReportResponseError,
    create_report_narrator,
)
from sentinel.reporting.report_builder import (
    ReportNotAvailableError,
    build_deterministic_narrative,
    build_report_from_workflow,
    determine_confidence,
    render_report_markdown,
    run_evidence_report,
)
from sentinel.reporting.schemas import (
    BusinessReport,
    ClaimType,
    ConfidenceLevel,
    ReportClaim,
    ReportMetric,
    ReportNarrative,
    SupportingDocument,
)
from sentinel.reporting.validation import validate_claim, validate_narrative

__all__ = [
    "BusinessReport",
    "ClaimType",
    "ConfidenceLevel",
    "NarrativeResult",
    "OpenAIReportNarrator",
    "ReportClaim",
    "ReportMetric",
    "ReportNarrative",
    "ReportNarrator",
    "ReportNotAvailableError",
    "ReportRequestError",
    "ReportResponseError",
    "SupportingDocument",
    "build_deterministic_narrative",
    "build_report_from_workflow",
    "create_report_narrator",
    "determine_confidence",
    "render_report_markdown",
    "run_evidence_report",
    "validate_claim",
    "validate_narrative",
]
