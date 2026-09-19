"""Optional live PostgreSQL check enabled only with an explicit test URL."""

import os

import pytest
from sqlalchemy import text

from sentinel.database.connection import create_database_engine
from sentinel.rag import DocumentVectorStore


@pytest.mark.skipif(
    not os.getenv("SENTINEL_TEST_DATABASE_URL"),
    reason="Set SENTINEL_TEST_DATABASE_URL to run the live PostgreSQL connectivity check.",
)
def test_live_postgresql_connection():
    engine = create_database_engine(os.environ["SENTINEL_TEST_DATABASE_URL"])
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT 1")) == 1
    finally:
        engine.dispose()


@pytest.mark.skipif(
    not os.getenv("SENTINEL_TEST_DATABASE_URL"),
    reason="Set SENTINEL_TEST_DATABASE_URL to run the live pgvector schema check.",
)
def test_live_pgvector_schema_creation():
    engine = create_database_engine(os.environ["SENTINEL_TEST_DATABASE_URL"])
    try:
        store = DocumentVectorStore(engine, dimensions=3)
        store.ensure_schema()
        with engine.connect() as connection:
            extension = connection.scalar(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            table = connection.scalar(text("SELECT to_regclass('document_chunks')"))
        assert extension == "vector"
        assert table == "document_chunks"
    finally:
        engine.dispose()
