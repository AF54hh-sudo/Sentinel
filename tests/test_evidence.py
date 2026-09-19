from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel.agents import WorkflowStatus, run_workflow
from sentinel.dashboard import load_dashboard_bundle
from sentinel.evidence import EvidenceRegistry
from sentinel.llm import QuestionIntent
from sentinel.rag import RetrievedDocumentChunk
from sentinel.reporting import (
    ClaimType,
    ConfidenceLevel,
    OpenAIReportNarrator,
    ReportClaim,
    ReportNotAvailableError,
    ReportResponseError,
    build_report_from_workflow,
    determine_confidence,
    render_report_markdown,
    validate_claim,
)


def _intent(*, question: str = "Show Q2 revenue", asks_for_cause: bool = False):
    return QuestionIntent.model_validate(
        {
            "original_question": question,
            "intent": "revenue_analysis",
            "metrics": ["net_revenue", "estimated_gross_contribution"],
            "analysis_mode": "descriptive",
            "time_period": {
                "label": "Q2 2024",
                "start_date": "2024-04-01",
                "end_date": "2024-06-30",
                "granularity": "quarter",
                "comparison_label": None,
                "comparison_start_date": None,
                "comparison_end_date": None,
            },
            "filters": {
                "regions": [],
                "plans": [],
                "industries": [],
                "customer_ids": [],
            },
            "asks_for_cause": asks_for_cause,
            "clarification_needed": False,
            "clarification_question": None,
            "interpretation": "Calculate Q2 net revenue and gross contribution.",
            "confidence": 0.97,
        }
    )


@pytest.fixture(scope="module")
def workflow():
    bundle = load_dashboard_bundle("csv")
    intent = _intent()
    return run_workflow(bundle, intent.original_question, intent=intent)


@pytest.fixture
def document():
    return RetrievedDocumentChunk(
        chunk_id="DOC-002-C001",
        document_id="DOC-002",
        document_title="Q2 2024 Business Review",
        document_date=date(2024, 7, 16),
        category="business_review",
        regions=("India", "North America", "Europe", "APAC"),
        topics=("profitability", "revenue"),
        source_path="02_q2_business_review.md",
        section="Management observations and investigation priorities",
        page=1,
        content="Finance requested a bridge using verified revenue, costs, and discounts.",
        similarity_score=0.89,
        business_relevance="Business Review · company-wide · profitability, revenue",
    )


def test_registry_assigns_deterministic_ids_and_deduplicates_documents(workflow, document):
    registry = EvidenceRegistry()
    analytical = registry.register_analysis(workflow.evidence)
    first_document = registry.register_document(document)
    duplicate = registry.register_document(document)
    bundle = registry.snapshot()

    assert analytical.evidence_id == "SQL-001"
    assert first_document.evidence_id == "DOC-001"
    assert duplicate.evidence_id == "DOC-001"
    assert bundle.ids == ("SQL-001", "DOC-001")
    assert bundle.get("DOC-001").payload["chunk_id"] == "DOC-002-C001"


def test_registry_rejects_unverified_analysis(workflow):
    registry = EvidenceRegistry()
    unverified = workflow.evidence.model_copy(update={"verified": False})
    with pytest.raises(ValueError, match="verified"):
        registry.register_analysis(unverified)


def test_deterministic_report_has_only_registered_citations(workflow, document):
    report = build_report_from_workflow(workflow, documents=[document])
    all_claims = (
        report.executive_summary,
        *report.key_findings,
        *report.business_recommendations,
        *report.limitations,
    )

    assert report.validated is True
    assert report.narration_provider == "deterministic"
    assert report.evidence.ids == ("SQL-001", "DOC-001")
    assert report.supporting_documents[0].chunk_id == "DOC-002-C001"
    assert all(
        evidence_id in report.evidence.ids
        for claim in all_claims
        for evidence_id in claim.evidence_ids
    )
    assert all(metric.evidence_id == "SQL-001" for metric in report.key_metrics)


def test_markdown_renderer_keeps_evidence_ids_visible(workflow, document):
    markdown = render_report_markdown(
        build_report_from_workflow(workflow, documents=[document])
    )

    assert "## Executive Summary" in markdown
    assert "## Supporting Documents" in markdown
    assert "[SQL-001]" in markdown
    assert "[DOC-001]" in markdown
    assert "Q2 2024 Business Review" in markdown


def test_confidence_policy_is_qualitative_and_cause_sensitive(workflow):
    level, rationale = determine_confidence(workflow)
    assert level == ConfidenceLevel.HIGH
    assert "%" not in rationale

    bundle = load_dashboard_bundle("csv")
    cause_intent = _intent(
        question="Why did Q2 profitability change?",
        asks_for_cause=True,
    )
    cause_workflow = run_workflow(
        bundle,
        cause_intent.original_question,
        intent=cause_intent,
    )
    cause_level, cause_rationale = determine_confidence(cause_workflow)
    assert cause_level == ConfidenceLevel.MEDIUM
    assert "cannot establish cause" in cause_rationale


def test_claim_validator_rejects_unknown_citation(workflow):
    registry = EvidenceRegistry()
    registry.register_analysis(workflow.evidence)
    claim = ReportClaim(
        claim_id="CLM-001",
        claim_type=ClaimType.FINDING,
        statement="The evidence was verified.",
        evidence_ids=("SQL-999",),
    )
    with pytest.raises(ValueError, match="Unknown evidence ID"):
        validate_claim(claim, registry.snapshot())


def test_claim_validator_rejects_fabricated_number_and_causation(workflow):
    registry = EvidenceRegistry()
    registry.register_analysis(workflow.evidence)
    evidence = registry.snapshot()
    fabricated = ReportClaim(
        claim_id="CLM-001",
        claim_type=ClaimType.FINDING,
        statement="Net revenue increased by 999999 percent.",
        evidence_ids=("SQL-001",),
    )
    causal = ReportClaim(
        claim_id="CLM-002",
        claim_type=ClaimType.FINDING,
        statement="Discounting caused the result.",
        evidence_ids=("SQL-001",),
    )

    with pytest.raises(ValueError, match="numbers absent"):
        validate_claim(fabricated, evidence)
    with pytest.raises(ValueError, match="causal"):
        validate_claim(causal, evidence)


def test_document_only_evidence_cannot_be_a_numerical_finding(document):
    registry = EvidenceRegistry()
    registry.register_document(document)
    claim = ReportClaim(
        claim_id="CLM-001",
        claim_type=ClaimType.FINDING,
        statement="The review was dated 2024.",
        evidence_ids=("DOC-001",),
    )
    with pytest.raises(ValueError, match="context, not a finding"):
        validate_claim(claim, registry.snapshot())


class FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id="resp-report", output_parsed=self.parsed)


class FakeClient:
    def __init__(self, parsed):
        self.responses = FakeResponses(parsed)


def _model_narrative(workflow):
    deterministic = build_report_from_workflow(workflow)
    recommendation = deterministic.business_recommendations[0].model_copy(
        update={"claim_id": "CLM-003"}
    )
    return {
        "executive_summary": deterministic.executive_summary.model_dump(mode="json"),
        "key_findings": [
            deterministic.key_findings[0].model_dump(mode="json"),
        ],
        "business_recommendations": [
            recommendation.model_dump(mode="json"),
        ],
    }


def test_openai_narrator_uses_structured_output_and_verified_registry(workflow):
    registry = EvidenceRegistry()
    registry.register_analysis(workflow.evidence)
    fake = FakeClient(_model_narrative(workflow))
    narrator = OpenAIReportNarrator(
        api_key="test-secret",
        model="test-model",
        client=fake,
    )

    result = narrator.narrate(
        workflow.question,
        workflow.intent.interpretation,
        registry.snapshot(),
    )

    assert result.provider == "openai"
    assert result.response_id == "resp-report"
    request = fake.responses.calls[0]
    assert request["text_format"].__name__ == "ReportNarrative"
    assert "no tools" in request["input"][0]["content"]
    assert "SQL-001" in request["input"][1]["content"]
    assert "test-secret" not in str(request)


def test_openai_narrator_rejects_fabricated_model_claim(workflow):
    payload = _model_narrative(workflow)
    payload["key_findings"][0]["statement"] = "Net revenue was 999999 currency."
    narrator = OpenAIReportNarrator(
        api_key="test-secret",
        model="test-model",
        client=FakeClient(payload),
    )
    registry = EvidenceRegistry()
    registry.register_analysis(workflow.evidence)

    with pytest.raises(ReportResponseError, match="verified evidence registry"):
        narrator.narrate(
            workflow.question,
            workflow.intent.interpretation,
            registry.snapshot(),
        )


def test_report_stops_without_verified_analysis(workflow):
    incomplete = workflow.model_copy(
        update={"status": WorkflowStatus.NEEDS_CLARIFICATION, "evidence": None}
    )
    with pytest.raises(ReportNotAvailableError, match="verified analytical evidence"):
        build_report_from_workflow(incomplete)


def test_reporting_runtime_has_no_ground_truth_or_arbitrary_execution_dependency():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("src/sentinel/reporting").glob("*.py"))
    )
    assert "ground_truth" not in source
    assert "eval(" not in source
    assert "exec(" not in source
    assert "subprocess" not in source


@pytest.mark.parametrize(
    ("question", "intent_name", "metrics", "mode", "dates", "expected_prefix"),
    [
        (
            "Did support quality affect churn?",
            "support_analysis",
            ["support_resolution_time", "satisfaction_score", "churn_rate"],
            "inferential",
            (None, None),
            "STAT-001",
        ),
        (
            "Which customers have highest churn risk?",
            "churn_risk_analysis",
            ["churn_probability"],
            "predictive",
            (None, None),
            "ML-001",
        ),
        (
            "Forecast next month revenue",
            "forecast_analysis",
            ["revenue_forecast"],
            "forecast",
            (None, None),
            "FORECAST-001",
        ),
        (
            "Show the May cloud cost anomaly",
            "anomaly_analysis",
            ["cost_anomaly", "gpu_cost"],
            "anomaly_detection",
            ("2024-05-01", "2024-05-31"),
            "ANOMALY-001",
        ),
        (
            "Are there data quality problems?",
            "data_quality_analysis",
            ["validation_errors", "duplicate_rows", "missing_values"],
            "descriptive",
            (None, None),
            "DATA-001",
        ),
    ],
)
def test_every_approved_analytical_branch_builds_a_valid_report(
    question,
    intent_name,
    metrics,
    mode,
    dates,
    expected_prefix,
):
    start_date, end_date = dates
    intent = QuestionIntent.model_validate(
        {
            "original_question": question,
            "intent": intent_name,
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
                "regions": [],
                "plans": [],
                "industries": [],
                "customer_ids": [],
            },
            "asks_for_cause": "affect" in question.lower(),
            "clarification_needed": False,
            "clarification_question": None,
            "interpretation": "Validated branch report test.",
            "confidence": 0.95,
        }
    )
    bundle = load_dashboard_bundle("csv")
    branch_workflow = run_workflow(bundle, question, intent=intent)
    report = build_report_from_workflow(branch_workflow)

    assert report.validated is True
    assert report.evidence.ids[0] == expected_prefix
    assert expected_prefix in render_report_markdown(report)
