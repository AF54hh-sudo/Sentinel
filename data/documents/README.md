# AMX Tech Internal Document Corpus

This directory contains ten fictional Markdown records for the Phase 14 retrieval layer. Each
record has validated YAML metadata and narrative business context. Documents are supporting
context, not authoritative numerical evidence; Sentinel's calculations continue to come from
the validated analytical layers.

Run a corpus-only validation without credentials or PostgreSQL:

```powershell
python scripts/ingest_documents.py --dry-run
```

Production ingestion requires an OpenAI API key and a PostgreSQL database where the configured
user may enable the `vector` extension and create `document_chunks`.
