"""Typed, deterministic evidence registry for Phase 15 reporting."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sentinel.agents import AnalysisEvidence, ToolName
from sentinel.rag import RetrievedDocumentChunk


class EvidenceKind(str, Enum):
    SQL = "sql"
    STATISTICS = "statistics"
    ML = "machine_learning"
    FORECAST = "forecast"
    ANOMALY = "anomaly"
    DATA_QUALITY = "data_quality"
    DOCUMENT = "document"


PREFIX_BY_KIND = {
    EvidenceKind.SQL: "SQL",
    EvidenceKind.STATISTICS: "STAT",
    EvidenceKind.ML: "ML",
    EvidenceKind.FORECAST: "FORECAST",
    EvidenceKind.ANOMALY: "ANOMALY",
    EvidenceKind.DATA_QUALITY: "DATA",
    EvidenceKind.DOCUMENT: "DOC",
}

KIND_BY_TOOL = {
    ToolName.BUSINESS: EvidenceKind.SQL,
    ToolName.STATISTICS: EvidenceKind.STATISTICS,
    ToolName.ML: EvidenceKind.ML,
    ToolName.FORECAST: EvidenceKind.FORECAST,
    ToolName.ANOMALY: EvidenceKind.ANOMALY,
    ToolName.DATA_QUALITY: EvidenceKind.DATA_QUALITY,
}


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(
        pattern=r"^(SQL|STAT|ML|FORECAST|ANOMALY|DATA|DOC)-\d{3}$"
    )
    kind: EvidenceKind
    title: str = Field(min_length=3, max_length=200)
    summary: str = Field(min_length=3, max_length=1_000)
    payload: dict[str, Any]
    source_references: tuple[str, ...] = Field(min_length=1, max_length=20)
    limitations: tuple[str, ...] = Field(max_length=20)
    verified: bool
    causal_support: bool = False


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[EvidenceItem, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.items)

    def get(self, evidence_id: str) -> EvidenceItem:
        for item in self.items:
            if item.evidence_id == evidence_id:
                return item
        raise KeyError(f"Unknown evidence ID: {evidence_id}")

    def validate_references(
        self,
        evidence_ids: tuple[str, ...] | list[str],
        *,
        require_verified: bool = True,
    ) -> tuple[EvidenceItem, ...]:
        if not evidence_ids:
            raise ValueError("at least one evidence citation is required")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence citations must be unique")
        try:
            items = tuple(self.get(evidence_id) for evidence_id in evidence_ids)
        except KeyError as error:
            raise ValueError(str(error)) from error
        if require_verified and any(not item.verified for item in items):
            raise ValueError("report claims may cite only verified evidence")
        return items


class EvidenceRegistry:
    """Assign stable per-report IDs and retain immutable evidence snapshots."""

    def __init__(self) -> None:
        self._items: list[EvidenceItem] = []
        self._counts = {kind: 0 for kind in EvidenceKind}
        self._document_chunk_ids: set[str] = set()

    def _next_id(self, kind: EvidenceKind) -> str:
        self._counts[kind] += 1
        return f"{PREFIX_BY_KIND[kind]}-{self._counts[kind]:03d}"

    def register_analysis(self, evidence: AnalysisEvidence) -> EvidenceItem:
        if not evidence.verified:
            raise ValueError("analytical evidence must be verified before registration")
        if evidence.tool not in KIND_BY_TOOL:
            raise ValueError(f"unsupported analytical evidence tool: {evidence.tool}")
        kind = KIND_BY_TOOL[evidence.tool]
        item = EvidenceItem(
            evidence_id=self._next_id(kind),
            kind=kind,
            title=evidence.title,
            summary=evidence.summary,
            payload={
                "tool": evidence.tool.value,
                "metrics": [metric.model_dump(mode="json") for metric in evidence.metrics],
                "records": evidence.records[:10],
                "record_count": len(evidence.records),
            },
            source_references=tuple(dict.fromkeys(evidence.sources)),
            limitations=tuple(evidence.limitations),
            verified=True,
            causal_support=False,
        )
        self._items.append(item)
        return item

    def register_document(self, document: RetrievedDocumentChunk) -> EvidenceItem:
        if document.chunk_id in self._document_chunk_ids:
            return next(
                item
                for item in self._items
                if item.kind == EvidenceKind.DOCUMENT
                and item.payload.get("chunk_id") == document.chunk_id
            )
        self._document_chunk_ids.add(document.chunk_id)
        item = EvidenceItem(
            evidence_id=self._next_id(EvidenceKind.DOCUMENT),
            kind=EvidenceKind.DOCUMENT,
            title=document.document_title,
            summary=document.business_relevance,
            payload={
                "chunk_id": document.chunk_id,
                "document_id": document.document_id,
                "document_date": document.document_date.isoformat(),
                "section": document.section,
                "page": document.page,
                "content": document.content,
                "similarity_score": document.similarity_score,
                "category": document.category,
                "regions": list(document.regions),
                "topics": list(document.topics),
            },
            source_references=(f"data/documents/{document.source_path}",),
            limitations=(
                "Internal documents provide narrative context, not authoritative numerical evidence.",
            ),
            verified=True,
            causal_support=False,
        )
        self._items.append(item)
        return item

    def snapshot(self) -> EvidenceBundle:
        return EvidenceBundle(items=tuple(self._items))
