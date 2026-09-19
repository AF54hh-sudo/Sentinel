"""AMX Tech internal-document ingestion, embeddings, and pgvector retrieval."""

from sentinel.rag.embeddings import (
    EmbeddingProvider,
    EmbeddingRequestError,
    OpenAIEmbeddingClient,
    create_embedding_client,
    embed_chunks,
)
from sentinel.rag.ingestion import (
    DocumentIngestionError,
    chunk_corpus,
    chunk_document,
    load_corpus,
    load_document,
)
from sentinel.rag.retrieval import DocumentRetriever, retrieve_documents
from sentinel.rag.schemas import (
    DocumentChunk,
    DocumentMetadata,
    EmbeddedDocumentChunk,
    IngestionSummary,
    RetrievedDocumentChunk,
    SourceDocument,
)
from sentinel.rag.store import DocumentVectorStore, build_document_chunks_table

__all__ = [
    "DocumentChunk",
    "DocumentIngestionError",
    "DocumentMetadata",
    "DocumentRetriever",
    "DocumentVectorStore",
    "EmbeddedDocumentChunk",
    "EmbeddingProvider",
    "EmbeddingRequestError",
    "IngestionSummary",
    "OpenAIEmbeddingClient",
    "RetrievedDocumentChunk",
    "SourceDocument",
    "build_document_chunks_table",
    "chunk_corpus",
    "chunk_document",
    "create_embedding_client",
    "embed_chunks",
    "load_corpus",
    "load_document",
    "retrieve_documents",
]
