"""Strict business-report contracts for grounded Phase 15 responses."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sentinel.evidence import EvidenceBundle


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class ClaimType(str, Enum):
    SUMMARY = "summary"
    FINDING = "finding"
    CONTEXT = "context"
    RECOMMENDATION = "recommendation"
    LIMITATION = "limitation"


class ReportClaim(StrictModel):
    claim_id: str = Field(pattern=r"^CLM-\d{3}$")
    claim_type: ClaimType
    statement: str = Field(min_length=3, max_length=800)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=10)

    @field_validator("statement")
    @classmethod
    def clean_statement(cls, value: str) -> str:
        return value.strip()


class ReportMetric(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    value: int | float | str | bool
    unit: str = Field(min_length=1, max_length=60)
    evidence_id: str


class SupportingDocument(StrictModel):
    evidence_id: str
    chunk_id: str
    document_title: str
    section: str
    page: int = Field(ge=1)
    similarity_score: float = Field(ge=-1.0, le=1.0)
    source_reference: str


class ReportNarrative(StrictModel):
    """The only report content a hosted model is allowed to author."""

    executive_summary: ReportClaim
    key_findings: tuple[ReportClaim, ...] = Field(min_length=1, max_length=8)
    business_recommendations: tuple[ReportClaim, ...] = Field(max_length=5)

    @model_validator(mode="after")
    def validate_claim_roles(self) -> ReportNarrative:
        if self.executive_summary.claim_type != ClaimType.SUMMARY:
            raise ValueError("executive_summary must use the summary claim type")
        if any(
            claim.claim_type not in {ClaimType.FINDING, ClaimType.CONTEXT}
            for claim in self.key_findings
        ):
            raise ValueError("key findings may contain only finding or context claims")
        if any(
            claim.claim_type != ClaimType.RECOMMENDATION
            for claim in self.business_recommendations
        ):
            raise ValueError("business recommendations must use the recommendation claim type")
        claim_ids = [
            self.executive_summary.claim_id,
            *(claim.claim_id for claim in self.key_findings),
            *(claim.claim_id for claim in self.business_recommendations),
        ]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("narrative claim IDs must be unique")
        expected_ids = [f"CLM-{index:03d}" for index in range(1, len(claim_ids) + 1)]
        if claim_ids != expected_ids:
            raise ValueError("narrative claim IDs must be sequential beginning with CLM-001")
        return self


class BusinessReport(StrictModel):
    question: str = Field(min_length=3, max_length=1_000)
    executive_summary: ReportClaim
    key_findings: tuple[ReportClaim, ...]
    key_metrics: tuple[ReportMetric, ...]
    supporting_documents: tuple[SupportingDocument, ...]
    business_recommendations: tuple[ReportClaim, ...]
    confidence: ConfidenceLevel
    confidence_rationale: str = Field(min_length=3, max_length=600)
    limitations: tuple[ReportClaim, ...]
    evidence: EvidenceBundle
    narration_provider: str
    narration_model: str | None = None
    validated: bool

    @model_validator(mode="after")
    def validate_structure(self) -> BusinessReport:
        claims = (
            self.executive_summary,
            *self.key_findings,
            *self.business_recommendations,
            *self.limitations,
        )
        claim_ids = [claim.claim_id for claim in claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("report claim IDs must be unique")
        expected_ids = [f"CLM-{index:03d}" for index in range(1, len(claim_ids) + 1)]
        if claim_ids != expected_ids:
            raise ValueError("report claim IDs must be sequential beginning with CLM-001")
        if not self.validated:
            raise ValueError("only claim-validated reports may use the BusinessReport contract")
        for claim in claims:
            self.evidence.validate_references(claim.evidence_ids)
        for metric in self.key_metrics:
            self.evidence.validate_references((metric.evidence_id,))
        for document in self.supporting_documents:
            self.evidence.validate_references((document.evidence_id,))
        return self
