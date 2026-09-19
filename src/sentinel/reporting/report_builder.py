"""Deterministic evidence assembly with optional hosted narration."""

from __future__ import annotations

from collections.abc import Sequence

from sentinel.agents import ToolName, WorkflowResult, WorkflowStatus, run_workflow
from sentinel.dashboard import DashboardBundle
from sentinel.evidence import EvidenceBundle, EvidenceKind, EvidenceRegistry
from sentinel.llm import QuestionIntent
from sentinel.rag import RetrievedDocumentChunk
from sentinel.reporting.client import ReportNarrator
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


class ReportNotAvailableError(RuntimeError):
    """Raised when verified analytical evidence is unavailable for reporting."""


def _build_registry(
    workflow: WorkflowResult,
    documents: Sequence[RetrievedDocumentChunk],
) -> tuple[EvidenceBundle, str]:
    if (
        workflow.status != WorkflowStatus.COMPLETED
        or workflow.evidence is None
        or not workflow.evidence.verified
    ):
        raise ReportNotAvailableError(
            "A business report requires a completed workflow with verified analytical evidence"
        )
    registry = EvidenceRegistry()
    analytical = registry.register_analysis(workflow.evidence)
    for document in documents[:5]:
        registry.register_document(document)
    return registry.snapshot(), analytical.evidence_id


def _next_claim_id(counter: int) -> str:
    return f"CLM-{counter:03d}"


def _recommendation(tool: ToolName) -> str:
    return {
        ToolName.BUSINESS: (
            "Review the verified period and segment metrics before changing commercial or cost policy."
        ),
        ToolName.STATISTICS: (
            "Treat the observed relationship as an association and evaluate any intervention "
            "prospectively."
        ),
        ToolName.ML: (
            "Use churn probabilities to prioritize human review rather than treating them as certainty."
        ),
        ToolName.FORECAST: (
            "Use the forecast as a planning estimate and refresh it when the next observed month arrives."
        ),
        ToolName.ANOMALY: (
            "Investigate the flagged dates operationally before interpreting the mechanism."
        ),
        ToolName.DATA_QUALITY: (
            "Address validation errors first and keep documented warnings visible in downstream analysis."
        ),
    }.get(tool, "Review the verified evidence before taking action.")


def build_deterministic_narrative(
    workflow: WorkflowResult,
    evidence: EvidenceBundle,
    analytical_id: str,
) -> ReportNarrative:
    """Create a complete offline narrative by copying, never recalculating, evidence."""
    analytical = evidence.get(analytical_id)
    counter = 1
    executive_summary = ReportClaim(
        claim_id=_next_claim_id(counter),
        claim_type=ClaimType.SUMMARY,
        statement=analytical.summary,
        evidence_ids=(analytical_id,),
    )
    counter += 1
    findings: list[ReportClaim] = []
    metrics = analytical.payload.get("metrics", [])
    for metric in metrics[:5] if isinstance(metrics, list) else []:
        statement = (
            f"{str(metric['name']).replace('_', ' ').title()}: "
            f"{metric['value']} {metric['unit']}."
        )
        findings.append(
            ReportClaim(
                claim_id=_next_claim_id(counter),
                claim_type=ClaimType.FINDING,
                statement=statement,
                evidence_ids=(analytical_id,),
            )
        )
        counter += 1
    for item in evidence.items:
        if item.kind != EvidenceKind.DOCUMENT or len(findings) >= 8:
            continue
        findings.append(
            ReportClaim(
                claim_id=_next_claim_id(counter),
                claim_type=ClaimType.CONTEXT,
                statement=f"{item.title}: {item.summary}.",
                evidence_ids=(item.evidence_id,),
            )
        )
        counter += 1
    recommendation = ReportClaim(
        claim_id=_next_claim_id(counter),
        claim_type=ClaimType.RECOMMENDATION,
        statement=_recommendation(workflow.selected_tool),
        evidence_ids=(analytical_id,),
    )
    narrative = ReportNarrative(
        executive_summary=executive_summary,
        key_findings=tuple(findings),
        business_recommendations=(recommendation,),
    )
    validate_narrative(narrative, evidence)
    return narrative


def determine_confidence(workflow: WorkflowResult) -> tuple[ConfidenceLevel, str]:
    """Apply a transparent qualitative policy rather than inventing a percentage."""
    if workflow.status != WorkflowStatus.COMPLETED or workflow.evidence is None:
        return (
            ConfidenceLevel.INSUFFICIENT,
            "Verified analytical evidence is unavailable.",
        )
    if workflow.selected_tool in {ToolName.BUSINESS, ToolName.DATA_QUALITY}:
        if workflow.intent.asks_for_cause:
            return (
                ConfidenceLevel.MEDIUM,
                "The calculations are verified, but the observational workflow cannot establish cause.",
            )
        return (
            ConfidenceLevel.HIGH,
            "The result is a direct deterministic calculation over validated business records.",
        )
    if workflow.selected_tool == ToolName.STATISTICS:
        return (
            ConfidenceLevel.MEDIUM,
            (
                "The tests and effect sizes are verified, but the synthetic observational design "
                "supports association rather than causation."
            ),
        )
    if workflow.selected_tool == ToolName.ML:
        return (
            ConfidenceLevel.MEDIUM,
            "Predictions use held-out evaluation evidence but remain probabilities from synthetic data.",
        )
    if workflow.selected_tool == ToolName.FORECAST:
        return (
            ConfidenceLevel.MEDIUM,
            "The forecast is chronologically evaluated but uses a short history and no calibrated interval.",
        )
    if workflow.selected_tool == ToolName.ANOMALY:
        return (
            ConfidenceLevel.MEDIUM,
            "Two retrospective detectors agree on unusual values, but anomaly evidence does not identify cause.",
        )
    return ConfidenceLevel.LOW, "The available evidence has material unresolved limitations."


def _report_metrics(evidence: EvidenceBundle, analytical_id: str) -> tuple[ReportMetric, ...]:
    item = evidence.get(analytical_id)
    metrics = item.payload.get("metrics", [])
    return tuple(
        ReportMetric(
            name=str(metric["name"]),
            value=metric["value"],
            unit=str(metric["unit"]),
            evidence_id=analytical_id,
        )
        for metric in (metrics[:12] if isinstance(metrics, list) else [])
    )


def _supporting_documents(evidence: EvidenceBundle) -> tuple[SupportingDocument, ...]:
    results: list[SupportingDocument] = []
    for item in evidence.items:
        if item.kind != EvidenceKind.DOCUMENT:
            continue
        results.append(
            SupportingDocument(
                evidence_id=item.evidence_id,
                chunk_id=str(item.payload["chunk_id"]),
                document_title=item.title,
                section=str(item.payload["section"]),
                page=int(item.payload["page"]),
                similarity_score=float(item.payload["similarity_score"]),
                source_reference=item.source_references[0],
            )
        )
    return tuple(results)


def _limitation_claims(
    evidence: EvidenceBundle,
    analytical_id: str,
    *,
    start_counter: int,
) -> tuple[ReportClaim, ...]:
    analytical = evidence.get(analytical_id)
    claims: list[ReportClaim] = []
    counter = start_counter
    for limitation in analytical.limitations[:8]:
        claim = ReportClaim(
            claim_id=_next_claim_id(counter),
            claim_type=ClaimType.LIMITATION,
            statement=limitation,
            evidence_ids=(analytical_id,),
        )
        validate_claim(claim, evidence)
        claims.append(claim)
        counter += 1
    if any(item.kind == EvidenceKind.DOCUMENT for item in evidence.items):
        document = next(item for item in evidence.items if item.kind == EvidenceKind.DOCUMENT)
        claim = ReportClaim(
            claim_id=_next_claim_id(counter),
            claim_type=ClaimType.LIMITATION,
            statement=document.limitations[0],
            evidence_ids=(document.evidence_id,),
        )
        validate_claim(claim, evidence)
        claims.append(claim)
    return tuple(claims)


def _claim_counter_after(narrative: ReportNarrative) -> int:
    claims = (
        narrative.executive_summary,
        *narrative.key_findings,
        *narrative.business_recommendations,
    )
    return max(int(claim.claim_id.split("-")[1]) for claim in claims) + 1


def build_report_from_workflow(
    workflow: WorkflowResult,
    *,
    documents: Sequence[RetrievedDocumentChunk] = (),
    narrator: ReportNarrator | None = None,
) -> BusinessReport:
    """Build and validate one report from verified analysis plus optional document context."""
    evidence, analytical_id = _build_registry(workflow, documents)
    if narrator is None:
        narrative = build_deterministic_narrative(workflow, evidence, analytical_id)
        provider = "deterministic"
        model = None
    else:
        result = narrator.narrate(
            workflow.question,
            workflow.intent.interpretation,
            evidence,
        )
        narrative = result.narrative
        validate_narrative(narrative, evidence)
        provider = result.provider
        model = result.model
    confidence, confidence_rationale = determine_confidence(workflow)
    limitations = _limitation_claims(
        evidence,
        analytical_id,
        start_counter=_claim_counter_after(narrative),
    )
    return BusinessReport(
        question=workflow.question,
        executive_summary=narrative.executive_summary,
        key_findings=narrative.key_findings,
        key_metrics=_report_metrics(evidence, analytical_id),
        supporting_documents=_supporting_documents(evidence),
        business_recommendations=narrative.business_recommendations,
        confidence=confidence,
        confidence_rationale=confidence_rationale,
        limitations=limitations,
        evidence=evidence,
        narration_provider=provider,
        narration_model=model,
        validated=True,
    )


def run_evidence_report(
    bundle: DashboardBundle,
    question: str,
    *,
    intent: QuestionIntent | None = None,
    documents: Sequence[RetrievedDocumentChunk] = (),
    narrator: ReportNarrator | None = None,
) -> BusinessReport:
    workflow = run_workflow(bundle, question, intent=intent)
    return build_report_from_workflow(workflow, documents=documents, narrator=narrator)


def _citation(evidence_ids: tuple[str, ...]) -> str:
    return " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)


def render_report_markdown(report: BusinessReport) -> str:
    """Render the validated contract without asking a model to format citations."""
    lines = [
        "# AMX Tech Sentinel Business Report",
        "",
        "## Executive Summary",
        "",
        f"{report.executive_summary.statement} {_citation(report.executive_summary.evidence_ids)}",
        "",
        "## Question",
        "",
        report.question,
        "",
        "## Key Findings",
        "",
    ]
    lines.extend(
        f"- {claim.statement} {_citation(claim.evidence_ids)}"
        for claim in report.key_findings
    )
    if report.key_metrics:
        lines.extend(["", "## Key Metrics", ""])
        lines.extend(
            f"- {metric.name}: {metric.value} {metric.unit} [{metric.evidence_id}]"
            for metric in report.key_metrics
        )
    if report.supporting_documents:
        lines.extend(["", "## Supporting Documents", ""])
        lines.extend(
            f"- {document.document_title}, {document.section}, page {document.page} "
            f"(similarity {document.similarity_score:.3f}) [{document.evidence_id}]"
            for document in report.supporting_documents
        )
    if report.business_recommendations:
        lines.extend(["", "## Business Recommendation", ""])
        lines.extend(
            f"- {claim.statement} {_citation(claim.evidence_ids)}"
            for claim in report.business_recommendations
        )
    lines.extend(
        [
            "",
            "## Confidence",
            "",
            f"{report.confidence.value.upper()} — {report.confidence_rationale}",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(
        f"- {claim.statement} {_citation(claim.evidence_ids)}"
        for claim in report.limitations
    )
    return "\n".join(lines).strip() + "\n"
