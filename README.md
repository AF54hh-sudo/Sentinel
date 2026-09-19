# SENTINEL

Business research and decision intelligence, with data science at the center.

## Project Overview

Sentinel combines SQL analytics, statistical inference, machine learning, anomaly detection, forecasting, internal-document retrieval, and evidence-based reporting over AMX Tech, a realistic fictional SaaS business. The analytical foundation remains authoritative while a lightweight hosted LLM handles question understanding and optional structured narration. Phases 1–15 build and validate the evidence pipeline; Phase 16 presents it as one polished, auditable investigation workspace.

## Business Problem

The application answers evidence-backed questions about revenue, profitability, churn, support quality, and cloud costs without delegating calculations to an LLM.

## Architecture

The current flow is configuration → seeded generator → five relational tables → PostgreSQL → validated analytical layers → bounded question workflow → evidence registry → citation-validated report, with a separate document → embedding → pgvector context path. See the [architecture](docs/architecture.md).

## Dataset

The generator creates about 5,000 customers, 8,000 subscriptions, 30,000-50,000 monthly sales records, 2,924 daily regional cloud-cost records, and 10,003 support-ticket records. See the [data dictionary](docs/data_dictionary.md).

Six controlled patterns support later investigations: Q2 profitability pressure, European churn, a May GPU anomaly, Enterprise discounting, Professional-plan churn risk, and forecastable revenue structure. Their causes are stored separately for evaluation and are forbidden at inference time.

## Data Science Workflow

The project follows business framing, extraction, validation, EDA, cleaning, feature engineering, statistics, baselines, modeling, evaluation, interpretation, visualization, and recommendation. The current checkpoint connects structured questions to deterministic analysis without granting the LLM executable tools.

## SQL

Phase 3 defines five constrained PostgreSQL business tables, indexes, transactional loading, conservative read-only query execution, row-count checks, and foreign-key smoke tests. Phase 5 adds parameterized revenue, contribution, churn, renewal, support, monthly-trend, region, and plan analytics with live SQL/Pandas reconciliation. See [PostgreSQL Setup](docs/database_setup.md) and [SQL Analytics](docs/sql_analytics.md).

## EDA

Phase 4 provides reusable cleaning, missingness, numerical/categorical profiling, IQR outlier summaries, correlations, business metrics, and monthly revenue, churn, cloud-cost, and support trends. The workflow runs from CSV or PostgreSQL with `python scripts/run_eda.py --source csv|database`. See [EDA findings](docs/eda_findings.md).

## Statistics

Phase 6 implements descriptive statistics, t and Wilson confidence intervals, Welch's t-test, Mann-Whitney U, Pearson and Spearman correlations, chi-square, Hedges' g, rank-biserial correlation, and Cramér's V. Reports include hypotheses, test rationale, sample size, assumptions, effect size, interpretation, and limitations. See [Statistical Findings](docs/statistical_findings.md).

## Machine Learning

Phase 7 builds a leakage-safe customer snapshot, compares Dummy, Logistic Regression, Random Forest, and HistGradientBoosting pipelines with stratified cross-validation, performs moderate randomized tuning, selects a recall-aware threshold from out-of-fold training predictions, evaluates once on a held-out test set, and persists a deployment pipeline. Average precision is the primary selection metric because churn prevalence is about 9.7%. See [Model Evaluation](docs/model_evaluation.md).

## Forecasting

Phase 9 compares previous-month naïve, three-month moving-average, and damped-trend exponential-smoothing forecasts with expanding-window validation. Exponential smoothing is selected before testing, reaches 0.583% test WAPE, and improves test WAPE 15.16% over naïve. The January 2025 point forecast is approximately 26.23 million. See [Monthly Revenue Forecast Findings](docs/forecast_findings.md).

## Anomaly Detection

Phase 8 aggregates reconciled regional costs to a daily company series, builds a robust rolling-median/IQR baseline, fits Isolation Forest, and requires conservative detector agreement for strong events. It recovers all seven injected GPU anomaly dates without false positives on the seed-42 dataset. See [Cloud-Cost Anomaly Findings](docs/anomaly_findings.md).

## Visualization

Phase 10 provides eight reusable Plotly views for revenue, churn, cloud-cost composition, region and plan performance, anomaly evidence, churn-model evaluation, and revenue forecasting. Figure functions validate explicit analytical contracts and can be reused directly by Streamlit. See the [Visualization Layer](docs/visualization.md).

## AI Layer and RAG

Phase 12 uses OpenAI Structured Outputs to convert free-form business questions into an allowlisted Pydantic intent contract. Phase 13 routes that contract through a fixed LangGraph to one of six verified analytical branches. Phase 14 retrieves exact-cosine company context from pgvector. Phase 15 registers analytical and document evidence, assigns visible citations, validates numbers and causal language, applies qualitative confidence, and optionally uses a second strict structured output for narration. See [Question Understanding](docs/question_understanding.md), [Bounded Workflow](docs/workflow.md), [Internal-Document Retrieval](docs/retrieval.md), and [Evidence-Based Reporting](docs/reporting.md).

## Streamlit Demo

The Streamlit interface guides a user through business question, structured intent, verified analysis, optional document context, and downloadable evidence report stages. It adds synchronized progress, branch-appropriate evidence visuals, capability readiness, session reset, actionable failure states, and a complete manual evidence explorer. Filters are applied only at supported analytical grains, and company-wide artifacts remain clearly labelled. See the [Dashboard Guide](docs/dashboard.md).

## Results and Evaluation

Phase 17 collects 178 tests: 167 pass offline and 11 optional PostgreSQL/pgvector checks skip when local services are unavailable. Branch-aware package coverage is 82.03% against an enforced 80% floor. Validation covers deterministic generation, relational integrity, accounting identities, SQL/Pandas reconciliation, statistics, end-to-end leakage-safe ML, anomaly detection, chronological forecasting, Plotly contracts, bounded orchestration, retrieval citations and resource disposal, evidence IDs, numerical grounding, causal-language rejection, qualitative confidence, structured narration, report rendering, Streamlit state transitions, and offline audit-tool behavior. See [Testing and Coverage](docs/testing.md).

## Limitations

The data is synthetic and cannot establish real-world causality. Currency is generic, billing is monthly, and the dataset covers two years. Churn probabilities are not separately calibrated, the anomaly baseline is retrospective, and the revenue forecast has only 24 observations with no calibrated prediction interval.

## Project Structure

Production code lives in `src/sentinel`, generated CSVs in `data/generated`, the fictional retrieval corpus in `data/documents`, scripts in `scripts`, checks in `tests`, notebooks in `notebooks`, and technical notes in `docs`. Trained artifacts and the visualization gallery are regenerated into the git-ignored `models` directory.

## Installation

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,phase3,science,dashboard,llm,orchestration,rag]"
```

## Usage

```powershell
python scripts/generate_data.py
python scripts/seed_database.py
python scripts/run_eda.py --source database
python scripts/run_sql_analytics.py
python scripts/run_statistics.py
python scripts/train_churn_model.py
python scripts/run_anomaly_detection.py --source database
python scripts/run_revenue_forecast.py --source database
python scripts/build_visualization_gallery.py --source database
streamlit run app.py
python scripts/understand_question.py "Why did Q2 profit fall?"
python scripts/run_workflow.py "Forecast next month's revenue"
python scripts/ingest_documents.py --dry-run
python scripts/ingest_documents.py --replace
python scripts/search_documents.py "What operational context explains the GPU-cost spike?"
python scripts/build_report.py "Why did Q2 profit fall?" --with-documents
pytest
```

Generated CSVs and trained model artifacts are intentionally git-ignored and reproducible with seed 42.

## Skills Demonstrated

Current: Python packaging, Pandas/NumPy data engineering, relational modeling, validation, EDA, PostgreSQL, parameterized business SQL, statistical inference, leakage-safe ML, anomaly detection, rolling-origin forecasting, Plotly, Streamlit interaction design, Pydantic structured outputs, prompt boundaries, bounded LangGraph state/routing, deterministic document chunking, OpenAI embeddings, pgvector cosine retrieval, typed evidence registries, claim/citation validation, qualitative confidence, grounded report narration, dependency injection, testing, configuration, and logging.
