"""Citation, numerical-grounding, and causal-language checks for reports."""

from __future__ import annotations

import json
import re

from sentinel.evidence import EvidenceBundle, EvidenceItem, EvidenceKind
from sentinel.reporting.schemas import ClaimType, ReportClaim, ReportNarrative

NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?")
CAUSAL_PATTERN = re.compile(
    r"\b(caused?|causing|drove|driven|led to|resulted in|responsible for)\b",
    re.IGNORECASE,
)
NEGATED_CAUSAL_PATTERN = re.compile(
    r"\b(no|not|cannot|does not|do not|without)\b[^.]{0,60}"
    r"\b(caused?|causing|drove|driven|led to|resulted in|responsible for)\b",
    re.IGNORECASE,
)


def _normalize_number(value: str) -> str:
    normalized = value.replace(",", "").lstrip("+")
    if normalized.endswith("%"):
        return normalized
    try:
        number = float(normalized)
    except ValueError:
        return normalized
    return f"{number:.12g}"


def _allowed_numbers(bundle: EvidenceBundle, evidence_ids: tuple[str, ...]) -> set[str]:
    allowed: set[str] = set()
    for item in bundle.validate_references(evidence_ids):
        serialized = json.dumps(item.model_dump(mode="json"), ensure_ascii=True, sort_keys=True)
        allowed.update(_normalize_number(token) for token in NUMBER_PATTERN.findall(serialized))
        metrics = item.payload.get("metrics", [])
        for metric in metrics if isinstance(metrics, list) else []:
            value = metric.get("value") if isinstance(metric, dict) else None
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            allowed.add(_normalize_number(str(value)))
            allowed.add(_normalize_number(f"{value:,}"))
            if isinstance(value, float):
                allowed.add(_normalize_number(f"{value:.1%}"))
                allowed.add(_normalize_number(f"{value:.2%}"))
                allowed.add(_normalize_number(f"{value:,.2f}"))
    return allowed


def _document_metadata_numbers(items: tuple[EvidenceItem, ...]) -> set[str]:
    allowed: set[str] = set()
    for item in items:
        metadata_text = " ".join(
            (
                item.title,
                str(item.payload.get("document_date", "")),
                str(item.payload.get("section", "")),
                str(item.payload.get("page", "")),
            )
        )
        allowed.update(
            _normalize_number(token) for token in NUMBER_PATTERN.findall(metadata_text)
        )
    return allowed


def validate_claim(claim: ReportClaim, bundle: EvidenceBundle) -> None:
    items = bundle.validate_references(claim.evidence_ids)
    numbers = {_normalize_number(token) for token in NUMBER_PATTERN.findall(claim.statement)}
    unsupported_numbers = numbers - _allowed_numbers(bundle, claim.evidence_ids)
    if unsupported_numbers:
        raise ValueError(
            f"{claim.claim_id} contains numbers absent from its cited evidence: "
            f"{sorted(unsupported_numbers)}"
        )
    document_only = all(item.kind == EvidenceKind.DOCUMENT for item in items)
    if document_only and claim.claim_type == ClaimType.FINDING:
        raise ValueError("document-only evidence must be presented as context, not a finding")
    if document_only and numbers - _document_metadata_numbers(items):
        raise ValueError(
            "documents cannot be the sole authority for numerical report claims; only citation "
            "metadata such as dates and pages may contain numbers"
        )
    causal_language = CAUSAL_PATTERN.search(claim.statement)
    negated_causal_language = NEGATED_CAUSAL_PATTERN.search(claim.statement)
    if (
        causal_language
        and not negated_causal_language
        and not any(item.causal_support for item in items)
    ):
        raise ValueError(f"{claim.claim_id} makes an unsupported causal claim")


def validate_narrative(narrative: ReportNarrative, bundle: EvidenceBundle) -> None:
    claims = (
        narrative.executive_summary,
        *narrative.key_findings,
        *narrative.business_recommendations,
    )
    for claim in claims:
        validate_claim(claim, bundle)
    summary_items = bundle.validate_references(narrative.executive_summary.evidence_ids)
    if all(item.kind == EvidenceKind.DOCUMENT for item in summary_items):
        raise ValueError("the executive summary must cite verified analytical evidence")
    if not any(
        any(
            item.kind != EvidenceKind.DOCUMENT
            for item in bundle.validate_references(claim.evidence_ids)
        )
        for claim in narrative.key_findings
    ):
        raise ValueError("at least one key finding must cite analytical evidence")
