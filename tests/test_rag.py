from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql

from sentinel.rag import (
    DocumentRetriever,
    DocumentVectorStore,
    EmbeddingRequestError,
    OpenAIEmbeddingClient,
    RetrievedDocumentChunk,
    chunk_corpus,
    embed_chunks,
    load_corpus,
)
from sentinel.rag import retrieval as retrieval_module


class FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [
            SimpleNamespace(index=index, embedding=[float(index + 1), 0.5, -0.5])
            for index, _ in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=list(reversed(data)))


class FakeHostedClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsAPI()


class FakeProvider:
    model = "test-embedding"
    dimensions = 3

    def embed_texts(self, texts):
        return [(1.0, 0.0, 0.0) for _ in texts]


class FakeStore:
    def __init__(self) -> None:
        self.call = None

    def search(self, query_embedding, **kwargs):
        self.call = (tuple(query_embedding), kwargs)
        return [
            RetrievedDocumentChunk(
                chunk_id="DOC-003-C001",
                document_id="DOC-003",
                document_title="Cloud Infrastructure Incident Review",
                document_date=date(2024, 5, 24),
                category="incident_review",
                regions=("India", "North America", "Europe", "APAC"),
                topics=("gpu_costs", "anomaly"),
                source_path="03_cloud_infrastructure_incident.md",
                section="GPU capacity event, response, and limitations",
                page=1,
                content="Reserved GPU workers stayed active after workloads shifted.",
                similarity_score=0.91,
                business_relevance="Incident Review · company-wide · gpu costs, anomaly",
            )
        ]


def test_phase14_corpus_has_ten_valid_documents_and_target_sized_chunks() -> None:
    documents = load_corpus(Path("data/documents"))
    chunks = chunk_corpus(documents)

    assert len(documents) == 10
    assert len(chunks) == 10
    assert len({document.metadata.document_id for document in documents}) == 10
    assert len({chunk.chunk_id for chunk in chunks}) == 10
    assert all(400 <= chunk.estimated_tokens <= 700 for chunk in chunks)
    assert all(chunk.page == 1 and chunk.source_path.endswith(".md") for chunk in chunks)
    assert all("NovaTech" not in document.content for document in documents)


def test_rag_runtime_has_no_evaluation_ground_truth_dependency() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("src/sentinel/rag").glob("*.py"))
    )
    assert "ground_truth" not in source
    assert "eval(" not in source
    assert "exec(" not in source


def test_openai_embedding_client_uses_batched_fixed_dimension_contract() -> None:
    hosted = FakeHostedClient()
    client = OpenAIEmbeddingClient(
        api_key="test-key",
        model="text-embedding-3-small",
        dimensions=3,
        batch_size=2,
        client=hosted,
    )

    vectors = client.embed_texts(["first", "second", "third"])

    assert vectors == [(1.0, 0.5, -0.5), (2.0, 0.5, -0.5), (1.0, 0.5, -0.5)]
    assert len(hosted.embeddings.calls) == 2
    assert hosted.embeddings.calls[0] == {
        "input": ["first", "second"],
        "model": "text-embedding-3-small",
        "dimensions": 3,
        "encoding_format": "float",
    }


def test_openai_embedding_client_rejects_wrong_dimension() -> None:
    hosted = FakeHostedClient()
    client = OpenAIEmbeddingClient(
        api_key="test-key",
        model="text-embedding-3-small",
        dimensions=4,
        client=hosted,
    )
    with pytest.raises(EmbeddingRequestError, match="dimensions"):
        client.embed_texts(["question"])


def test_chunk_embeddings_preserve_citation_metadata() -> None:
    chunk = chunk_corpus(load_corpus(Path("data/documents")))[0]
    embedded = embed_chunks([chunk], FakeProvider())[0]

    assert embedded.chunk_id == chunk.chunk_id
    assert embedded.content_hash == chunk.content_hash
    assert embedded.embedding == (1.0, 0.0, 0.0)
    assert embedded.embedding_model == "test-embedding"
    assert embedded.embedding_dimensions == 3


def test_retriever_embeds_only_question_and_returns_typed_citations() -> None:
    store = FakeStore()
    results = DocumentRetriever(FakeProvider(), store).retrieve(
        "What explains the GPU cost spike?", top_k=3, min_similarity=0.2
    )

    assert results[0].chunk_id == "DOC-003-C001"
    assert store.call == (
        (1.0, 0.0, 0.0),
        {"embedding_model": "test-embedding", "top_k": 3, "min_similarity": 0.2},
    )


def test_retriever_rejects_empty_questions_before_embedding() -> None:
    with pytest.raises(ValueError, match="at least three characters"):
        DocumentRetriever(FakeProvider(), FakeStore()).retrieve("  ")


def test_configured_retrieval_disposes_database_resources(monkeypatch) -> None:
    engine = SimpleNamespace(disposed=False)
    engine.dispose = lambda: setattr(engine, "disposed", True)
    store = FakeStore()
    settings = SimpleNamespace(
        database_url="postgresql+psycopg://test:test@localhost/test",
        rag_top_k=3,
        rag_min_similarity=0.2,
    )
    monkeypatch.setattr(
        retrieval_module,
        "create_embedding_client",
        lambda configured: FakeProvider(),
    )
    monkeypatch.setattr(
        retrieval_module,
        "create_database_engine",
        lambda database_url: engine,
    )
    monkeypatch.setattr(
        retrieval_module,
        "DocumentVectorStore",
        lambda configured_engine, dimensions: store,
    )

    results = retrieval_module.retrieve_documents(
        "What explains the GPU cost spike?",
        settings=settings,
    )

    assert results[0].chunk_id == "DOC-003-C001"
    assert store.call[1] == {
        "embedding_model": "test-embedding",
        "top_k": 3,
        "min_similarity": 0.2,
    }
    assert engine.disposed is True


def test_pgvector_search_statement_uses_bound_exact_cosine_distance() -> None:
    engine = create_engine("postgresql+psycopg://sentinel:test@localhost/sentinel_test")
    try:
        store = DocumentVectorStore(engine, dimensions=3)
        statement = store.build_search_statement(
            [1.0, 0.0, 0.0],
            embedding_model="test-embedding",
            top_k=5,
            min_similarity=0.1,
        )
        sql = str(statement.compile(dialect=postgresql.dialect()))
    finally:
        engine.dispose()

    assert "document_chunks.embedding <=>" in sql
    assert "document_chunks.embedding_model =" in sql
    assert "LIMIT" in sql
    assert "DROP" not in sql.upper()
