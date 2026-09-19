# Phase 17 Testing and Coverage

Sentinel's test strategy verifies analytical correctness, evidence boundaries, model discipline,
and the complete Streamlit workflow without requiring hosted services for the default suite.

## Verification result

The Phase 17 checkpoint collects **179 tests**:

- **168 passed** in the offline/default environment.
- **11 skipped** because live PostgreSQL or pgvector was unavailable.
- **82.03% total branch-aware package coverage**, above the enforced 80% floor.
- Ruff and `pip check` pass.

The coverage measurement targets `src/sentinel`. Streamlit behavior is tested separately with the
framework's `AppTest` renderer because `app.py` is a rerun-oriented entry point rather than an
importable package module.

## Test layers

- **Data and persistence:** deterministic generation, relational integrity, business identities,
  intentional scenario discoverability, validation failures, and CSV round trips.
- **Analytics and SQL:** metric definitions, query safety, compiled statements, and optional live
  PostgreSQL/Pandas reconciliation.
- **Statistics:** confidence intervals, assumptions, effect sizes, null-hypothesis decisions, and
  interpretation boundaries.
- **Machine learning:** leakage-safe snapshots, required dummy baseline, candidate pipelines,
  cross-validation, moderate tuning, out-of-fold threshold selection, held-out evaluation, error
  analysis, ranking, interpretation, and deployment refit.
- **Forecasting and anomalies:** chronological evaluation, baseline comparison, forecast horizon,
  robust anomaly ranges, detector agreement, and evaluation-only ground truth separation.
- **LLM and orchestration:** strict intent schemas, normalized failures, allowlisted routing, bounded
  graph execution, and no arbitrary SQL or Python execution.
- **RAG and reporting:** deterministic citations, embedding contracts, vector-search statements,
  database resource disposal, evidence IDs, number copying, causal-language rejection, confidence,
  report validation, and Markdown rendering.
- **Streamlit:** startup, guided-stage transitions, verified workflow execution, report creation,
  citation-preserving document display, quick-start state behavior, and session reset.

## Regression found during Phase 17

The new end-to-end churn-training test exposed a threshold-selection edge case. Some classifiers can
return exact `0.0` or `1.0` probabilities; those values were being considered as thresholds even
though the metric contract correctly requires a threshold strictly inside `(0, 1)`. The selector now
filters boundary values before evaluating candidates, and a dedicated regression assertion protects
the fix.

## Commands

Run the default suite with explicit pass/skip totals:

```powershell
python -m pytest -o addopts="--strict-markers"
```

Run the enforced branch-aware coverage gate:

```powershell
python -m pytest --cov=sentinel --cov-report=term-missing:skip-covered
```

Run static and environment checks:

```powershell
python -m ruff check .
python -m pip check
```

## Optional integration tests

Two tests in `tests/test_database_integration.py` require
`SENTINEL_TEST_DATABASE_URL`. Nine SQL reconciliation tests use the configured local PostgreSQL
database and skip as a group when it is unavailable. To exercise all 179 tests, start and seed
PostgreSQL, enable pgvector, configure the test URL, and rerun the suite.

No default test sends requests to OpenAI. Hosted clients are replaced with deterministic fakes, and
RAG resource-lifecycle tests inject an in-memory search seam.
