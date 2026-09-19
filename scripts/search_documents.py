"""Search AMX Tech internal documents using configured OpenAI embeddings and pgvector."""

from __future__ import annotations

import argparse

from sentinel.rag import retrieve_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Business question used to retrieve document context")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = retrieve_documents(args.question)
    if not results:
        print("No document chunks met the configured similarity threshold.")
        return 0
    for result in results:
        print(f"[{result.chunk_id}] {result.document_title}")
        print(
            f"{result.section} · page {result.page} · similarity "
            f"{result.similarity_score:.3f}"
        )
        print(result.business_relevance)
        print(result.content)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
