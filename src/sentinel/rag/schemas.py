"""Typed contracts for AMX Tech document ingestion and retrieval."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentMetadata(StrictModel):
    document_id: str = Field(pattern=r"^DOC-\d{3}$")
    title: str = Field(min_length=3, max_length=160)
    document_date: date
    category: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    regions: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    source_path: str = Field(min_length=1, max_length=260)

    @field_validator("regions", "topics")
    @classmethod
    def unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values if value.strip())
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("metadata lists must not contain duplicate values")
        return cleaned


class SourceDocument(StrictModel):
    metadata: DocumentMetadata
    content: str = Field(min_length=1)


class DocumentChunk(StrictModel):
    chunk_id: str = Field(pattern=r"^DOC-\d{3}-C\d{3}$")
    document_id: str = Field(pattern=r"^DOC-\d{3}$")
    document_title: str = Field(min_length=3, max_length=160)
    document_date: date
    category: str
    regions: tuple[str, ...]
    topics: tuple[str, ...]
    source_path: str
    section: str = Field(min_length=1, max_length=160)
    page: int = Field(ge=1)
    chunk_index: int = Field(ge=1)
    content: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    estimated_tokens: int = Field(ge=1)


class EmbeddedDocumentChunk(DocumentChunk):
    embedding: tuple[float, ...]
    embedding_model: str = Field(min_length=1, max_length=100)
    embedding_dimensions: int = Field(ge=1)


class RetrievedDocumentChunk(StrictModel):
    chunk_id: str
    document_id: str
    document_title: str
    document_date: date
    category: str
    regions: tuple[str, ...]
    topics: tuple[str, ...]
    source_path: str
    section: str
    page: int
    content: str
    similarity_score: float = Field(ge=-1.0, le=1.0)
    business_relevance: str = Field(min_length=1)


class IngestionSummary(StrictModel):
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    embedded_chunks: int = Field(ge=0)
    stored_chunks: int = Field(ge=0)
    embedding_model: str | None = None
    embedding_dimensions: int | None = Field(default=None, ge=1)
