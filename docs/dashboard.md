# Phase 16 Final Streamlit Experience

The AMX Tech Sentinel interface presents the project as one decision-intelligence workspace. A
reviewer can move from a free-form business question to structured intent, verified analytical
evidence, optional internal-document context, and a citation-validated report without losing the
manual Data Science explorer that was established in Phase 11.

The application uses a fixed light presentation theme so metric values, labels, form fields, tabs,
and evidence cards keep consistent high contrast regardless of the browser or operating-system
color preference. The navy sidebar and dark hero remain intentionally isolated brand surfaces.

## Run locally

Install the application and analytical dependencies:

```powershell
python -m pip install -e ".[dev,phase3,science,dashboard,llm,orchestration,rag]"
```

Regenerate the Phase 7–9 artifacts if `models/` is absent, then start the interface:

```powershell
streamlit run app.py
```

Generated CSV is the default source and works without PostgreSQL. PostgreSQL document retrieval
requires pgvector, an ingested document index, and configured OpenAI embeddings. Question
understanding requires the hosted LLM configuration. Missing optional services produce actionable
messages and do not disable the manual analytical explorer.

## Guided investigation

The top workspace shows four explicit stages:

1. **Business question:** choose a quick-start prompt or write the decision question directly.
2. **Structured intent:** inspect the allowlisted metrics, time period, filters, analysis mode, and
   confidence returned by the question-understanding boundary.
3. **Verified analysis:** review the deterministic plan, calculated metrics, records, sources,
   limitations, execution trace, and a branch-appropriate evidence visual.
4. **Evidence report:** optionally add cited internal-document context, choose deterministic or
   hosted structured narration, validate every claim, and download the Markdown report.

**Start new investigation** clears question, intent, workflow, document, and report state. The
sidebar keeps data-source and analytical-scope controls visible and separates validated local
capabilities from optional services that still need configuration.

## Manual evidence explorer

Sidebar filters are intentionally grain-aware. Revenue accepts date, region, and plan filters.
Churn accepts region and plan filters, then displays only months in the selected period.
Infrastructure costs accept date and region because shared costs cannot be allocated credibly to
plans. Forecast and anomaly artifacts are company-wide and are labelled as unaffected by those
filters.

The evidence tabs provide:

- **Overview:** filter-aware commercial and cost metrics with the three principal time series.
- **Revenue:** monthly gross/net revenue, discounts, and region/plan comparisons.
- **Churn & ML:** monthly churn, held-out model evidence, an adjustable operating threshold, and a
  ranked held-out risk table.
- **Costs:** monthly cost composition, contribution, and cost-to-revenue context.
- **Forecast:** selected-method test evidence and the three-month point forecast.
- **Anomalies:** robust normal ranges, detector agreement, and strong anomaly rows.
- **Documents:** citation-preserving semantic search over fictional AMX Tech internal records.
- **Methodology:** data-quality actions and analytical guardrails.

## Evidence boundaries

`src/sentinel/dashboard/data.py` remains the dashboard's sole loading and filtering boundary. It
cleans and validates the five business tables, verifies required Phase 7–9 artifacts, and returns
chart-ready frames without mutating source data. Plotly functions remain presentation-only.

Question understanding receives trusted metadata rather than business rows. The LangGraph workflow
routes an allowlisted intent to one deterministic analytical branch and cannot execute arbitrary SQL
or Python. Document retrieval is a separate context path. The report registry validates evidence
IDs, copied numbers, causal language, and qualitative confidence before rendering. Phase 16 changes
the composition and interaction flow only; it does not move calculations into Streamlit or the LLM.

Predictions are held-out evaluation signals rather than live scores for every customer. Forecasts
are company-wide point estimates, and anomaly detection is retrospective. None of these outputs
establishes causation.
