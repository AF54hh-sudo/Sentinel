# Phase 14 Internal-Document Retrieval

Phase 14 adds a deliberately small retrieval layer for supporting AMX Tech business context. The
corpus contains ten fictional Markdown records in `data/documents`. Each source has YAML metadata
for a stable document ID, title, date, category, regions, and topics.

## Boundary

Documents can explain management intent, operational chronology, policy, and known limitations.
They are not authoritative sources for revenue, churn, forecast, model, statistical, or anomaly
values. Phase 13 calculations remain unchanged. Phase 15 may register retrieved chunks beside the
analytical result, but document-only statements remain contextual and cannot supply business metrics.

The runtime never reads `data/ground_truth`. The corpus itself repeatedly labels observational and
causal limitations so retrieved context cannot silently become a numerical conclusion.

## Ingestion

```text
UTF-8 Markdown
    → validated YAML metadata
    → whitespace cleaning
    → heading/paragraph sections
    → approximately 400–700-token chunks
    → stable chunk IDs and SHA-256 content hashes
    → OpenAI float embeddings
    → PostgreSQL document_chunks table with pgvector
```

The ten current documents each produce one citation-preserving chunk. Page is `1` for Markdown;
the contract retains the field so a future, explicitly implemented PDF extractor can preserve
source pages. Unsupported file formats fail rather than being guessed.

OpenAI requests are batched, use the configured `text-embedding-3-small` model by default, request
the configured fixed dimension, and validate response count, indexes, numeric values, and vector
length. Tests inject a fake client and make no network request.

## Storage and retrieval

`DocumentVectorStore` enables the `vector` extension and creates a separate `document_chunks`
table. Upserts are atomic; deleting the existing corpus requires explicit `--replace`. Exact cosine
distance is appropriate for this tiny corpus, so Phase 14 adds no approximate index, reranker,
hybrid BM25 search, external vector database, or autonomous retrieval agent.

A search embeds only the user's question, filters rows to the same embedding model, orders by exact
cosine distance, applies the configured similarity threshold, and returns at most 20 chunks. Every
result preserves document title, date, section/page, source path, chunk ID, text, similarity, and a
deterministic metadata-based business-relevance label.

## Commands

Validate extraction and chunk sizes without credentials or PostgreSQL:

```powershell
python scripts/ingest_documents.py --dry-run
```

Embed and atomically replace the current corpus:

```powershell
python scripts/ingest_documents.py --replace
```

Search from the CLI:

```powershell
python scripts/search_documents.py "What operational context explains the GPU-cost spike?"
```

Production ingestion/search requires `SENTINEL_LLM_API_KEY`, the embedding settings from
`.env.example`, and PostgreSQL credentials allowed to create the pgvector extension and table. A
change to embedding dimensions requires an explicit database schema migration; `--replace` changes
rows, not column dimensions.

## Streamlit

The **Documents** tab runs the same configured query embedding and pgvector retrieval. Results are
shown as supporting records with their citations and similarity score. The tab fails independently
when the API, extension, table, or database is unavailable, so the manual dashboard and verified
analytical workflow continue to function. Phase 15 reports can include up to five currently retrieved
chunks while retaining their source IDs and contextual boundary.
