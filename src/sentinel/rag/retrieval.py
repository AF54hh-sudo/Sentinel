"""Question embedding and citation-preserving document retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from sentinel.config.settings import Settings, get_settings
from sentinel.database.connection import create_database_engine
from sentinel.rag.embeddings import EmbeddingProvider, create_embedding_client
from sentinel.rag.schemas import RetrievedDocumentChunk
from sentinel.rag.store import DocumentVectorStore


class VectorSearchStore(Protocol):
    def search(
        self,
        query_embedding: Sequence[float],
        *,
        embedding_model: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[RetrievedDocumentChunk]: ...


class DocumentRetriever:
    """Retrieve supporting context without generating an analytical conclusion."""

    def __init__(self, provider: EmbeddingProvider, store: VectorSearchStore) -> None:
        self.provider = provider
        self.store = store

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[RetrievedDocumentChunk]:
        normalized = question.strip()
        if len(normalized) < 3:
            raise ValueError("document search requires at least three characters")
        query_embedding = self.provider.embed_texts([normalized])[0]
        return self.store.search(
            query_embedding,
            embedding_model=self.provider.model,
            top_k=top_k,
            min_similarity=min_similarity,
        )


def retrieve_documents(
    question: str,
    *,
    settings: Settings | None = None,
) -> list[RetrievedDocumentChunk]:
    """Run configured OpenAI embedding and pgvector search with clean resource disposal."""
    configured = settings or get_settings()
    provider = create_embedding_client(configured)
    engine = create_database_engine(configured.database_url)
    try:
        store = DocumentVectorStore(engine, dimensions=provider.dimensions)
        return DocumentRetriever(provider, store).retrieve(
            question,
            top_k=configured.rag_top_k,
            min_similarity=configured.rag_min_similarity,
        )
    finally:
        engine.dispose()
