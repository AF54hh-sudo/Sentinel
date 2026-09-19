# PostgreSQL Setup

Phase 3 targets a local PostgreSQL database named `sentinel_db`. PostgreSQL is an external prerequisite; Sentinel does not install or start the server.

For this development workspace, the live Phase 3 verification used the installed Docker Desktop runtime because a native PostgreSQL service was not available. The database is localhost-only and runs in the `sentinel-postgres` container. A native installation remains equally supported.

## Create a local role and database

Run these commands from an administrator PostgreSQL session, choosing a private password instead of the example value:

```sql
CREATE ROLE sentinel WITH LOGIN PASSWORD 'choose-a-private-password';
CREATE DATABASE sentinel_db OWNER sentinel;
```

Copy `.env.example` to `.env` and set the matching connection URL:

```dotenv
SENTINEL_DATABASE_URL=postgresql+psycopg://sentinel:choose-a-private-password@localhost:5432/sentinel_db
```

The password must remain only in `.env`, which is excluded from Git.

## Generate and seed

```powershell
python scripts/generate_data.py
python scripts/seed_database.py
```

The seed is transactional: any load failure rolls back all inserted rows. A non-empty Sentinel database is protected by default. To deliberately replace its business-table contents:

```powershell
python scripts/seed_database.py --replace
```

## Verification

The seed command compares loaded and database row counts and checks all four foreign-key relationships for orphans. To enable the optional connectivity test, set `SENTINEL_TEST_DATABASE_URL` to a safe PostgreSQL database URL before running `pytest`.

Phase 3 creates only the five validated business tables. Phase 14 adds the separate
`document_chunks` table when `scripts/ingest_documents.py` runs. That command first executes
`CREATE EXTENSION IF NOT EXISTS vector`, so the PostgreSQL installation must provide pgvector and
the configured role must be allowed to enable it. The table uses the configured fixed embedding
dimension; changing that dimension later requires an explicit schema migration.

## Development container lifecycle

When using the verified development container:

```powershell
docker start sentinel-postgres
docker stop sentinel-postgres
```

The container uses the disposable local-only password recorded in the git-ignored `.env`. Change it before exposing PostgreSQL beyond localhost. Removing the container also removes its database because no persistent volume is configured; regenerate and reseed the synthetic data if that is done.
