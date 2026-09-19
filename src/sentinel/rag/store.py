"""PostgreSQL/pgvector storage for citation-preserving document chunks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    Date,
    Engine,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.sql import Select

from sentinel.rag.schemas import EmbeddedDocumentChunk, RetrievedDocumentChunk


def build_document_chunks_table(dimensions: int) -> tuple[MetaData, Table]:
    """Build a fixed-dimension vector table without mutating the business schema metadata."""
    if dimensions < 1:
        raise ValueError("embedding dimensions must be positive")
    metadata = MetaData()
    table = Table(
        "document_chunks",
        metadata,
        Column("chunk_id", String(20), primary_key=True),
        Column("document_id", String(12), nullable=False),
        Column("document_title", String(160), nullable=False),
        Column("document_date", Date, nullable=False),
        Column("category", String(64), nullable=False),
        Column("regions", JSON, nullable=False),
        Column("topics", JSON, nullable=False),
        Column("source_path", String(260), nullable=False),
        Column("section", String(160), nullable=False),
        Column("page", Integer, nullable=False),
        Column("chunk_index", Integer, nullable=False),
        Column("content", Text, nullable=False),
        Column("content_hash", String(64), nullable=False),
        Column("estimated_tokens", Integer, nullable=False),
        Column("embedding_model", String(100), nullable=False),
        Column("embedding_dimensions", Integer, nullable=False),
        Column("embedding", VECTOR(dimensions), nullable=False),
        CheckConstraint("page >= 1", name="document_chunks_page_positive"),
        CheckConstraint("chunk_index >= 1", name="document_chunks_index_positive"),
        CheckConstraint("estimated_tokens >= 1", name="document_chunks_tokens_positive"),
        CheckConstraint(
            f"embedding_dimensions = {dimensions}",
            name="document_chunks_embedding_dimensions",
        ),
    )
    Index("ix_document_chunks_document", table.c.document_id, table.c.chunk_index)
    Index("ix_document_chunks_category", table.c.category)
    Index("ix_document_chunks_embedding_model", table.c.embedding_model)
    return metadata, table


class DocumentVectorStore:
    """Upsert and retrieve the small company corpus using exact cosine distance."""

    def __init__(self, engine: Engine, *, dimensions: int) -> None:
        if engine.url.get_backend_name() != "postgresql":
            raise ValueError("DocumentVectorStore requires PostgreSQL")
        self.engine = engine
        self.dimensions = dimensions
        self.metadata, self.table = build_document_chunks_table(dimensions)

    def ensure_schema(self) -> None:
        """Enable pgvector and create the document table when missing."""
        with self.engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.metadata.create_all(connection, checkfirst=True)

    def upsert_chunks(
        self,
        chunks: Sequence[EmbeddedDocumentChunk],
        *,
        replace: bool = False,
    ) -> int:
        """Atomically insert/update chunks; full replacement requires explicit opt-in."""
        if not chunks:
            raise ValueError("at least one embedded chunk is required")
        if any(chunk.embedding_dimensions != self.dimensions for chunk in chunks):
            raise ValueError("chunk embedding dimensions do not match the vector store")
        records = []
        for chunk in chunks:
            record = chunk.model_dump()
            record["regions"] = list(chunk.regions)
            record["topics"] = list(chunk.topics)
            record["embedding"] = list(chunk.embedding)
            records.append(record)

        with self.engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.metadata.create_all(connection, checkfirst=True)
            if replace:
                connection.execute(delete(self.table))
            statement = postgresql_insert(self.table).values(records)
            update_fields = {
                column.name: getattr(statement.excluded, column.name)
                for column in self.table.columns
                if column.name != "chunk_id"
            }
            connection.execute(
                statement.on_conflict_do_update(
                    index_elements=[self.table.c.chunk_id],
                    set_=update_fields,
                )
            )
        return len(records)

    def build_search_statement(
        self,
        query_embedding: Sequence[float],
        *,
        embedding_model: str,
        top_k: int,
        min_similarity: float,
    ) -> Select[Any]:
        """Build the bound exact-cosine query used by production and SQL-contract tests."""
        if len(query_embedding) != self.dimensions:
            raise ValueError("query embedding dimensions do not match the vector store")
        if not all(math.isfinite(float(value)) for value in query_embedding):
            raise ValueError("query embedding contains non-finite values")
        if not embedding_model.strip():
            raise ValueError("embedding_model is required")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1")

        cosine_distance = self.table.c.embedding.cosine_distance(list(query_embedding))
        similarity = (1.0 - cosine_distance).label("similarity_score")
        return (
            select(
                self.table.c.chunk_id,
                self.table.c.document_id,
                self.table.c.document_title,
                self.table.c.document_date,
                self.table.c.category,
                self.table.c.regions,
                self.table.c.topics,
                self.table.c.source_path,
                self.table.c.section,
                self.table.c.page,
                self.table.c.content,
                similarity,
            )
            .where(
                self.table.c.embedding_model == embedding_model,
                similarity >= min_similarity,
            )
            .order_by(cosine_distance, self.table.c.chunk_id)
            .limit(top_k)
        )

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        embedding_model: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[RetrievedDocumentChunk]:
        statement = self.build_search_statement(
            query_embedding,
            embedding_model=embedding_model,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        results: list[RetrievedDocumentChunk] = []
        for row in rows:
            regions = tuple(row["regions"] or ())
            topics = tuple(row["topics"] or ())
            scope = ", ".join(regions) if regions else "company-wide"
            topic_text = ", ".join(topic.replace("_", " ") for topic in topics)
            relevance = f"{row['category'].replace('_', ' ').title()} · {scope}"
            if topic_text:
                relevance += f" · {topic_text}"
            results.append(
                RetrievedDocumentChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    document_title=row["document_title"],
                    document_date=row["document_date"],
                    category=row["category"],
                    regions=regions,
                    topics=topics,
                    source_path=row["source_path"],
                    section=row["section"],
                    page=row["page"],
                    content=row["content"],
                    similarity_score=float(row["similarity_score"]),
                    business_relevance=relevance,
                )
            )
        return results

    def count_chunks(self) -> int:
        with self.engine.connect() as connection:
            return int(connection.scalar(select(func.count()).select_from(self.table)) or 0)
