"""Extract, chunk, embed, and store the AMX Tech internal-document corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentinel.config import get_settings
from sentinel.database.connection import create_database_engine
from sentinel.rag import (
    DocumentVectorStore,
    IngestionSummary,
    chunk_corpus,
    create_embedding_client,
    embed_chunks,
    load_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-dir", type=Path, help="Override the configured corpus path")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing document chunks before the atomic upsert",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and chunk documents without an API or database call",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    documents_dir = args.documents_dir or settings.documents_dir
    documents = load_corpus(documents_dir)
    chunks = chunk_corpus(documents)

    if args.dry_run:
        summary = IngestionSummary(
            documents=len(documents),
            chunks=len(chunks),
            embedded_chunks=0,
            stored_chunks=0,
        )
        print(summary.model_dump_json(indent=2))
        for chunk in chunks:
            print(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document": chunk.document_title,
                        "section": chunk.section,
                        "estimated_tokens": chunk.estimated_tokens,
                    }
                )
            )
        return 0

    provider = create_embedding_client(settings)
    embedded = embed_chunks(chunks, provider)
    engine = create_database_engine(settings.database_url)
    try:
        store = DocumentVectorStore(engine, dimensions=provider.dimensions)
        stored = store.upsert_chunks(embedded, replace=args.replace)
    finally:
        engine.dispose()
    summary = IngestionSummary(
        documents=len(documents),
        chunks=len(chunks),
        embedded_chunks=len(embedded),
        stored_chunks=stored,
        embedding_model=provider.model,
        embedding_dimensions=provider.dimensions,
    )
    print(summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
