"""Injected OpenAI embedding boundary with strict response validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

from sentinel.config.settings import Settings, get_settings
from sentinel.llm.client import LLMConfigurationError
from sentinel.rag.schemas import DocumentChunk, EmbeddedDocumentChunk


class EmbeddingRequestError(RuntimeError):
    """Raised when embedding generation fails or returns an invalid vector."""


class EmbeddingsAPI(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class HostedEmbeddingClient(Protocol):
    embeddings: EmbeddingsAPI


class EmbeddingProvider(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


class OpenAIEmbeddingClient:
    """Create fixed-dimension float embeddings without exposing provider details downstream."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int = 1536,
        batch_size: int = 64,
        timeout_seconds: float = 30,
        max_retries: int = 2,
        client: HostedEmbeddingClient | None = None,
        request_error_types: tuple[type[BaseException], ...] | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("SENTINEL_LLM_API_KEY is not configured")
        if not model.strip():
            raise LLMConfigurationError("SENTINEL_EMBEDDING_MODEL is not configured")
        if dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        if batch_size < 1 or batch_size > 2048:
            raise ValueError("embedding batch size must be between 1 and 2048")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")

        if client is None:
            try:
                from openai import OpenAI, OpenAIError
            except ImportError as error:
                raise LLMConfigurationError(
                    "Install the Phase 14 dependencies with: pip install -e .[rag]"
                ) from error
            client = OpenAI(
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )
            request_error_types = (OpenAIError,)

        self._client = client
        self._model = model.strip()
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._request_error_types = request_error_types or (
            OSError,
            RuntimeError,
            TimeoutError,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _validate_vector(self, vector: Any) -> tuple[float, ...]:
        try:
            values = tuple(float(value) for value in vector)
        except (TypeError, ValueError) as error:
            raise EmbeddingRequestError("Embedding response contains a non-numeric vector") from error
        if len(values) != self._dimensions:
            raise EmbeddingRequestError(
                f"Embedding response has {len(values)} dimensions; expected {self._dimensions}"
            )
        if not all(math.isfinite(value) for value in values):
            raise EmbeddingRequestError("Embedding response contains non-finite values")
        return values

    def embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        normalized = [text.strip() for text in texts]
        if not normalized or any(not text for text in normalized):
            raise ValueError("embedding input must contain non-empty text")

        vectors: list[tuple[float, ...]] = []
        for offset in range(0, len(normalized), self._batch_size):
            batch = normalized[offset : offset + self._batch_size]
            try:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model,
                    dimensions=self._dimensions,
                    encoding_format="float",
                )
            except self._request_error_types as error:
                raise EmbeddingRequestError("OpenAI embedding request failed") from error
            data = list(getattr(response, "data", ()))
            if len(data) != len(batch):
                raise EmbeddingRequestError("Embedding response count does not match input count")
            try:
                ordered = sorted(data, key=lambda item: int(item.index))
            except (AttributeError, TypeError, ValueError) as error:
                raise EmbeddingRequestError("Embedding response indexes are invalid") from error
            if [int(item.index) for item in ordered] != list(range(len(batch))):
                raise EmbeddingRequestError("Embedding response indexes are incomplete")
            vectors.extend(self._validate_vector(item.embedding) for item in ordered)
        return vectors


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    provider: EmbeddingProvider,
) -> list[EmbeddedDocumentChunk]:
    """Attach validated embeddings while preserving the complete citation contract."""
    if not chunks:
        return []
    vectors = provider.embed_texts([chunk.content for chunk in chunks])
    if len(vectors) != len(chunks):
        raise EmbeddingRequestError("Embedding provider returned an unexpected vector count")
    embedded: list[EmbeddedDocumentChunk] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        if len(vector) != provider.dimensions:
            raise EmbeddingRequestError("Embedding provider returned an unexpected dimension")
        embedded.append(
            EmbeddedDocumentChunk(
                **chunk.model_dump(),
                embedding=vector,
                embedding_model=provider.model,
                embedding_dimensions=provider.dimensions,
            )
        )
    return embedded


def create_embedding_client(settings: Settings | None = None) -> OpenAIEmbeddingClient:
    """Create the explicitly configured embedding provider."""
    configured = settings or get_settings()
    provider = configured.embedding_provider.strip().lower()
    if provider != "openai":
        raise LLMConfigurationError(f"Unsupported embedding provider: {provider or '(empty)'}")
    return OpenAIEmbeddingClient(
        api_key=configured.llm_api_key,
        model=configured.embedding_model,
        dimensions=configured.embedding_dimensions,
        batch_size=configured.embedding_batch_size,
        timeout_seconds=configured.llm_timeout_seconds,
        max_retries=configured.llm_max_retries,
    )
