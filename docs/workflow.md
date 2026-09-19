# Phase 13 Bounded LangGraph Workflow

Phase 13 connects validated question intent to Sentinel's existing analytical capabilities through
one small, deterministic LangGraph workflow. It is orchestration, not an autonomous agent swarm.

```text
START
  → understand_question
  → create_analysis_plan
  → one approved analytical branch
  → validate_evidence
  → END
```

Questions that need clarification, request unsupported segment grains, or fall outside Sentinel's
domains go from planning to `stop_without_analysis` and end without running a tool.

## Analytical branches

| Branch | Evidence source |
|---|---|
| Business analytics | Cleaned relational tables and Phase 4 Pandas metrics |
| Statistical analysis | Phase 6 test report with p-values, effects, assumptions, and limitations |
| Churn model | Phase 7 held-out probabilities, operating threshold, and held-out evaluation |
| Revenue forecast | Phase 9 selected method, future point forecasts, and chronological test errors |
| Anomaly detection | Phase 8 robust ranges and detector-agreement scores |
| Data quality | Post-cleaning relational validation and cleaning report |

Routing is an explicit mapping from the allowlisted `QuestionIntent`; the LLM cannot name functions,
generate SQL, execute Python, or add graph nodes. Each run selects at most one analytical branch.
RAG remains deliberately absent from this analytical graph. Phase 14 exposes document retrieval as
a separate capability, and Phase 15 joins already verified analysis with optional document context
in the downstream evidence registry rather than adding open-ended graph nodes.

## Evidence contract

Every tool returns an `AnalysisEvidence` object containing a tool identifier, summary, typed metrics,
optional records, explicit sources, limitations, and a verification flag. The final validation node
checks that evidence matches the selected branch and has at least one metric and source before setting
`verified=true` and completing the run.

Shared-grain restrictions are enforced during planning. For example, plan/customer/industry cloud
cost allocation stops for clarification because infrastructure facts do not support those dimensions.
Company-wide forecasts, anomaly evidence, statistical reports, and data-quality reports likewise do
not silently inherit unsupported segment filters.

## Run

Configure Phase 12's hosted LLM settings, then use Streamlit or the CLI:

```powershell
python -m pip install -e ".[dev,phase3,science,dashboard,llm,orchestration]"
streamlit run app.py
python scripts/run_workflow.py "Forecast next month's revenue"
```

The Streamlit interface displays the structured intent first. A separate **Run verified analysis**
action executes the approved branch and displays the deterministic plan, metrics, records, sources,
limitations, and execution trace. No LLM-generated executive interpretation is added in Phase 13.
