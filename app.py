"""AMX Tech Sentinel Phase 16 decision-intelligence experience."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from sentinel.agents import ToolName, WorkflowResult, WorkflowStatus, run_workflow
from sentinel.config import get_settings
from sentinel.dashboard import (
    DashboardBundle,
    filter_churn_predictions,
    load_dashboard_bundle,
    prepare_dashboard_view,
)
from sentinel.llm import (
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
    QuestionContext,
    QuestionIntent,
    create_intent_client,
)
from sentinel.rag import EmbeddingRequestError, RetrievedDocumentChunk, retrieve_documents
from sentinel.reporting import (
    BusinessReport,
    ReportNotAvailableError,
    ReportRequestError,
    ReportResponseError,
    build_report_from_workflow,
    create_report_narrator,
    render_report_markdown,
)
from sentinel.visualization import (
    anomaly_evidence_chart,
    churn_model_evaluation_chart,
    churn_trend_chart,
    cloud_cost_chart,
    forecast_evidence_chart,
    revenue_trend_chart,
    segment_performance_chart,
)
from sentinel.visualization.charts import EXPORT_CONFIG

st.set_page_config(
    page_title="AMX Tech | Sentinel",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --amx-blue: #2563eb;
        --amx-blue-dark: #1d4ed8;
        --amx-ink: #172033;
        --amx-muted: #64748b;
        --amx-line: #dbe3ef;
        --amx-soft: #eff6ff;
    }
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] { background: #f5f7fb; color: var(--amx-ink); }
    .block-container { max-width: 1480px; padding-top: 2.1rem; padding-bottom: 4rem; }
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #f8fafc; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #172033; }
    [data-testid="stSidebar"] input { color: #172033; }
    [data-testid="stSidebar"] hr { border-color: rgba(248, 250, 252, .14); }
    [data-testid="stMetric"] {
        background: white; border: 1px solid var(--amx-line); border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 4px 18px rgba(15, 23, 42, .035);
    }
    [data-testid="stMain"] [data-testid="stMetricLabel"] p {
        color: #526176 !important; font-weight: 650;
    }
    [data-testid="stMain"] [data-testid="stMetricValue"] {
        color: #111827 !important; font-weight: 750;
    }
    [data-testid="stMain"] [data-testid="stMetricDelta"] {
        font-weight: 650;
    }
    [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff; border-color: var(--amx-line);
    }
    [data-testid="stMain"] [data-testid="stTextInput"] input,
    [data-testid="stMain"] [data-testid="stTextArea"] textarea,
    [data-testid="stMain"] [data-baseweb="select"] > div {
        background: #ffffff !important; color: #172033 !important;
    }
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        color: #526176 !important;
    }
    [data-testid="stMain"] [data-baseweb="tab-list"] [role="tab"] {
        color: #526176;
    }
    [data-testid="stMain"] [data-baseweb="tab-list"] [aria-selected="true"] {
        color: #1d4ed8; font-weight: 700;
    }
    .amx-hero {
        background: linear-gradient(122deg, #0f172a 0%, #172554 56%, #1d4ed8 140%);
        border-radius: 20px;
        color: white;
        padding: 2rem 2.2rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 14px 34px rgba(15, 23, 42, .14);
    }
    .amx-kicker { color: #93c5fd; font-weight: 750; letter-spacing: .14em;
        font-size: .75rem; text-transform: uppercase; margin-bottom: .45rem; }
    .amx-hero h1 { color: white; font-size: clamp(2.2rem, 5vw, 3.5rem); line-height: 1;
        letter-spacing: -.04em; margin: 0 0 .7rem; }
    .amx-subtitle { color: #dbeafe; max-width: 860px; font-size: 1.02rem;
        line-height: 1.6; margin-bottom: 1.1rem; }
    .amx-chips { display: flex; flex-wrap: wrap; gap: .5rem; }
    .amx-chip { background: rgba(255, 255, 255, .1); border: 1px solid rgba(255, 255, 255, .2);
        border-radius: 999px; color: #f8fafc; font-size: .78rem; padding: .35rem .65rem; }
    .amx-section-label { color: var(--amx-blue); font-size: .74rem; font-weight: 750;
        letter-spacing: .1em; text-transform: uppercase; margin: .25rem 0 .35rem; }
    .amx-stage { background: white; border: 1px solid var(--amx-line); border-radius: 12px;
        min-height: 78px; padding: .75rem .85rem; }
    .amx-stage.active { border-color: var(--amx-blue); box-shadow: 0 0 0 2px rgba(37, 99, 235, .08); }
    .amx-stage.complete { background: var(--amx-soft); border-color: #bfdbfe; }
    .amx-stage-number { align-items: center; background: #e2e8f0; border-radius: 999px;
        color: #475569; display: inline-flex; font-size: .72rem; font-weight: 800;
        height: 24px; justify-content: center; margin-right: .35rem; width: 24px; }
    .amx-stage.active .amx-stage-number, .amx-stage.complete .amx-stage-number {
        background: var(--amx-blue); color: white;
    }
    .amx-stage-title { color: var(--amx-ink); font-size: .86rem; font-weight: 700; }
    .amx-stage-status { color: var(--amx-muted); font-size: .74rem; margin-top: .35rem; }
    .amx-note { background: #eff6ff; border-left: 4px solid var(--amx-blue);
        border-radius: 8px; padding: .8rem 1rem; color: #1e3a5f; }
    .amx-status { align-items: center; display: flex; font-size: .82rem;
        justify-content: space-between; margin: .5rem 0; }
    .amx-status-pill { border: 1px solid rgba(255, 255, 255, .22); border-radius: 999px;
        font-size: .68rem; font-weight: 700; letter-spacing: .04em; padding: .18rem .45rem;
        text-transform: uppercase; }
    .amx-status-pill.ready { background: rgba(16, 185, 129, .18); color: #a7f3d0; }
    .amx-status-pill.setup { background: rgba(245, 158, 11, .16); color: #fde68a; }
    div[data-testid="stPlotlyChart"] { background: white; border: 1px solid #e2e8f0;
        border-radius: 14px; padding: 4px; }
    div.stButton > button[kind="primary"] { box-shadow: 0 6px 16px rgba(37, 99, 235, .18); }
    @media (max-width: 720px) {
        .block-container { padding-top: 1rem; }
        .amx-hero { border-radius: 14px; padding: 1.4rem; }
        .amx-stage { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


_INVESTIGATION_KEYS = (
    "sentinel_intent_result",
    "sentinel_workflow_result",
    "sentinel_business_report",
    "sentinel_document_results",
    "sentinel_business_question",
    "sentinel_example_question",
    "sentinel_report_document_query",
    "sentinel_document_query",
)


def _reset_investigation() -> None:
    for key in _INVESTIGATION_KEYS:
        st.session_state.pop(key, None)


def _copy_example_question() -> None:
    selected = st.session_state.get("sentinel_example_question", "")
    if selected:
        for key in (
            "sentinel_intent_result",
            "sentinel_workflow_result",
            "sentinel_business_report",
            "sentinel_document_results",
            "sentinel_report_document_query",
            "sentinel_document_query",
        ):
            st.session_state.pop(key, None)
        st.session_state["sentinel_business_question"] = selected


def _investigation_stage() -> int:
    if "sentinel_business_report" in st.session_state:
        return 4
    if "sentinel_workflow_result" in st.session_state:
        return 3
    if "sentinel_intent_result" in st.session_state:
        return 2
    return 1


def _render_investigation_progress() -> None:
    current = _investigation_stage()
    stages = (
        ("Business question", "Define the decision"),
        ("Structured intent", "Confirm the scope"),
        ("Verified analysis", "Review the evidence"),
        ("Evidence report", "Share the decision support"),
    )
    columns = st.columns(4)
    for index, (title, pending_text) in enumerate(stages, start=1):
        state = "complete" if index < current else "active" if index == current else ""
        status = "Complete" if index < current else "In progress" if index == current else pending_text
        marker = "✓" if index < current else str(index)
        columns[index - 1].markdown(
            f'<div class="amx-stage {state}"><span class="amx-stage-number">{marker}</span>'
            f'<span class="amx-stage-title">{title}</span>'
            f'<div class="amx-stage-status">{status}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption(f"Investigation progress: step {current} of 4 · {stages[current - 1][0]}")


def _sidebar_status(label: str, *, ready: bool, ready_label: str = "Ready") -> None:
    state = "ready" if ready else "setup"
    value = ready_label if ready else "Setup"
    st.markdown(
        f'<div class="amx-status"><span>{label}</span>'
        f'<span class="amx-status-pill {state}">{value}</span></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=300)
def _load_bundle(source: str) -> DashboardBundle:
    return load_dashboard_bundle(source)


def _money(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:,.1f}K"
    return f"{value:,.0f}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _metric_row(view: Any) -> None:
    columns = st.columns(5)
    columns[0].metric("Net revenue", _money(view.net_revenue), _percent(view.revenue_growth))
    columns[1].metric("Gross contribution", _money(view.contribution))
    columns[2].metric("Latest monthly churn", _percent(view.latest_churn_rate))
    columns[3].metric("Infrastructure cost", _money(view.infrastructure_cost))
    columns[4].metric("Revenue customers", f"{view.revenue_customers:,}")


def _plot(figure: go.Figure, *, key: str) -> None:
    st.plotly_chart(figure, width="stretch", config=EXPORT_CONFIG, key=key)


def _limitations(items: list[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")


def _render_intent_result(payload: dict[str, Any]) -> None:
    intent = QuestionIntent.model_validate(payload["intent"])
    st.markdown("#### Structured intent")
    intent_columns = st.columns(4)
    intent_columns[0].metric("Intent", intent.intent.value.replace("_", " ").title())
    intent_columns[1].metric("Analysis mode", intent.analysis_mode.value.replace("_", " ").title())
    intent_columns[2].metric("Confidence", f"{intent.confidence:.0%}")
    intent_columns[3].metric("Cause requested", "Yes" if intent.asks_for_cause else "No")
    st.write(intent.interpretation)

    detail_columns = st.columns(3)
    detail_columns[0].markdown(
        "**Metrics**  \n" + ", ".join(metric.value.replace("_", " ") for metric in intent.metrics)
    )
    period = intent.time_period
    resolved_dates = (
        f"{period.start_date:%d %b %Y} – {period.end_date:%d %b %Y}"
        if period.start_date and period.end_date
        else "Not resolved"
    )
    detail_columns[1].markdown(f"**Time period**  \n{period.label} · {resolved_dates}")
    filters = [*intent.filters.regions, *intent.filters.plans, *intent.filters.industries]
    detail_columns[2].markdown(
        "**Filters**  \n" + (", ".join(filters) if filters else "Company-wide")
    )

    if intent.clarification_needed:
        st.warning(intent.clarification_question)
    else:
        st.info("The validated intent is ready for the separate Phase 13 analytical workflow.")
    with st.expander("Validated intent payload"):
        st.json(intent.model_dump(mode="json"))
        st.caption(f"Provider: {payload['provider']} · Model: {payload['model']}")


def _render_workflow_visualization(result: WorkflowResult, bundle: DashboardBundle) -> None:
    if result.status != WorkflowStatus.COMPLETED or result.evidence is None:
        return
    st.markdown("#### Decision visual")
    if result.selected_tool == ToolName.FORECAST:
        _plot(
            forecast_evidence_chart(
                bundle.revenue_history,
                bundle.forecast_backtest,
                bundle.future_forecast,
                selected_method=bundle.forecast_report["selected_method"],
            ),
            key="workflow-forecast",
        )
    elif result.selected_tool == ToolName.ANOMALY:
        _plot(anomaly_evidence_chart(bundle.anomaly_scores), key="workflow-anomaly")
    elif result.selected_tool == ToolName.ML:
        _plot(churn_model_evaluation_chart(bundle.churn_report), key="workflow-ml")
    elif result.selected_tool == ToolName.BUSINESS and result.evidence.records:
        records = pd.DataFrame(result.evidence.records)
        for dimension in ("region", "plan_type"):
            segment_rows = records[records["dimension"].eq(dimension)]
            if segment_rows.empty:
                continue
            segment_rows = segment_rows.rename(columns={"segment": dimension})
            _plot(
                segment_performance_chart(segment_rows, dimension),
                key=f"workflow-business-{dimension}",
            )
            break
    else:
        st.caption(
            "This branch is best represented by its validated metric and evidence tables below."
        )


def _render_workflow_result(payload: dict[str, Any], bundle: DashboardBundle) -> None:
    result = WorkflowResult.model_validate(payload)
    st.markdown("#### Analysis plan")
    st.caption(result.analysis_plan.rationale)
    for step in result.analysis_plan.steps:
        st.markdown(f"{step.order}. {step.action}")

    if result.status != WorkflowStatus.COMPLETED or result.evidence is None:
        for limitation in result.limitations:
            st.warning(limitation)
        return

    evidence = result.evidence
    st.markdown(f"#### {evidence.title}")
    st.write(evidence.summary)
    _render_workflow_visualization(result, bundle)
    metric_rows = [metric.model_dump(mode="json") for metric in evidence.metrics]
    st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
    if evidence.records:
        with st.expander("Evidence records", expanded=True):
            st.dataframe(pd.DataFrame(evidence.records), width="stretch", hide_index=True)
    with st.expander("Evidence sources and limitations"):
        st.markdown("**Sources**")
        _limitations(evidence.sources)
        st.markdown("**Limitations**")
        _limitations(evidence.limitations)
        st.caption("Execution: " + " → ".join(result.execution_trace))


def _render_document_result(result: RetrievedDocumentChunk) -> None:
    st.markdown(f"#### {result.document_title}")
    st.caption(
        f"{result.chunk_id} · {result.document_date:%d %b %Y} · "
        f"{result.section} · page {result.page} · similarity {result.similarity_score:.3f}"
    )
    st.markdown(f"**Business relevance:** {result.business_relevance}")
    st.write(result.content)
    st.caption(f"Source: data/documents/{result.source_path}")


def _document_search_form(
    default_query: str,
    *,
    settings: Any,
    form_key: str,
    input_key: str,
    button_label: str,
) -> list[RetrievedDocumentChunk]:
    with st.form(form_key):
        query = st.text_input(
            "Search internal documents",
            value=default_query,
            placeholder="What operational context explains the GPU-cost spike?",
            key=input_key,
        )
        search_documents = st.form_submit_button(button_label)
    if search_documents:
        st.session_state.pop("sentinel_document_results", None)
        try:
            with st.spinner("Embedding the question and searching the evidence index…"):
                matches = retrieve_documents(query, settings=settings)
            st.session_state["sentinel_document_results"] = [
                match.model_dump(mode="json") for match in matches
            ]
        except (
            EmbeddingRequestError,
            LLMConfigurationError,
            OSError,
            RuntimeError,
            SQLAlchemyError,
            ValueError,
        ) as error:
            st.error(f"Document search could not complete: {error}")
            st.info(
                "Configure the OpenAI key, start PostgreSQL with pgvector, and run "
                "python scripts/ingest_documents.py --replace. The verified analytical "
                "workflow remains available without document context."
            )
    return [
        RetrievedDocumentChunk.model_validate(payload)
        for payload in st.session_state.get("sentinel_document_results", [])
    ]


def _report_citations(evidence_ids: tuple[str, ...]) -> str:
    return " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)


def _render_business_report(payload: dict[str, Any]) -> None:
    report = BusinessReport.model_validate(payload)
    st.markdown("### Evidence-based business report")
    st.markdown("#### Executive summary")
    st.write(report.executive_summary.statement)
    st.caption(_report_citations(report.executive_summary.evidence_ids))

    st.markdown("#### Key findings")
    for claim in report.key_findings:
        st.markdown(f"- {claim.statement} {_report_citations(claim.evidence_ids)}")
    if report.key_metrics:
        st.markdown("#### Key metrics")
        st.dataframe(
            pd.DataFrame(metric.model_dump(mode="json") for metric in report.key_metrics),
            width="stretch",
            hide_index=True,
        )
    if report.supporting_documents:
        st.markdown("#### Supporting documents")
        for document in report.supporting_documents:
            st.markdown(
                f"- **{document.document_title}** · {document.section} · page {document.page} · "
                f"similarity {document.similarity_score:.3f} [{document.evidence_id}]"
            )
    if report.business_recommendations:
        st.markdown("#### Business recommendation")
        for claim in report.business_recommendations:
            st.markdown(f"- {claim.statement} {_report_citations(claim.evidence_ids)}")
    st.markdown("#### Confidence")
    st.info(f"{report.confidence.value.upper()} — {report.confidence_rationale}")
    st.markdown("#### Limitations")
    for claim in report.limitations:
        st.markdown(f"- {claim.statement} {_report_citations(claim.evidence_ids)}")
    st.caption(
        f"Narration: {report.narration_provider}"
        + (f" · {report.narration_model}" if report.narration_model else "")
        + " · claim validation passed"
    )
    with st.expander("Evidence registry"):
        st.json(report.evidence.model_dump(mode="json"))
    st.download_button(
        "Download report as Markdown",
        data=render_report_markdown(report),
        file_name="amx_tech_sentinel_report.md",
        mime="text/markdown",
    )


with st.sidebar:
    st.markdown("## ◆ SENTINEL")
    st.caption("AMX Tech · Decision intelligence")
    st.markdown("---")
    st.caption("PHASE 16 · FINAL EXPERIENCE")
    source_label = st.selectbox("Data source", ("Generated CSV", "PostgreSQL"))
    source = "csv" if source_label == "Generated CSV" else "database"
    st.button(
        "Start new investigation",
        on_click=_reset_investigation,
        help="Clear the current question, evidence, documents, and report.",
    )

try:
    bundle = _load_bundle(source)
except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
    st.error(f"Sentinel could not load the selected source: {error}")
    st.info("Choose Generated CSV, or start and seed the local PostgreSQL service.")
    st.stop()

with st.sidebar:
    st.markdown("### Analysis scope")
    minimum_date, maximum_date = bundle.date_bounds
    selected_dates = st.date_input(
        "Analysis period",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )
    selected_regions = tuple(
        st.multiselect("Regions", bundle.regions, default=list(bundle.regions))
    )
    selected_plans = tuple(st.multiselect("Plans", bundle.plans, default=list(bundle.plans)))
    llm_settings = get_settings()
    llm_ready = bool(
        llm_settings.llm_provider and llm_settings.llm_model and llm_settings.llm_api_key
    )
    rag_ready = bool(
        llm_settings.embedding_provider
        and llm_settings.embedding_model
        and llm_settings.llm_api_key
    )
    st.markdown("### System readiness")
    _sidebar_status("Analytical data", ready=True, ready_label="Validated")
    _sidebar_status("Models & forecasts", ready=True, ready_label="Verified")
    _sidebar_status("Question understanding", ready=llm_ready, ready_label="Configured")
    _sidebar_status("Document retrieval", ready=rag_ready, ready_label="Configured")
    with st.expander("Data and model details"):
        for table_name, table in bundle.tables.items():
            st.caption(f"{table_name.replace('_', ' ').title()}: {len(table):,} rows")
        st.markdown("**Selected evidence models**")
        st.caption(
            f"Churn · {bundle.churn_report['selected_model'].replace('_', ' ').title()}"
        )
        st.caption(
            f"Forecast · {bundle.forecast_report['selected_method'].replace('_', ' ').title()}"
        )
        st.caption("Anomaly · Robust IQR + Isolation Forest")

if len(selected_dates) != 2:
    st.info("Select both a start and end date to load the analysis.")
    st.stop()

try:
    view = prepare_dashboard_view(
        bundle,
        start_date=selected_dates[0],
        end_date=selected_dates[1],
        regions=selected_regions,
        plans=selected_plans,
    )
except ValueError as error:
    st.warning(str(error))
    st.stop()

st.markdown(
    '<section class="amx-hero" aria-labelledby="sentinel-title">'
    '<div class="amx-kicker">AMX Tech · Decision intelligence workspace</div>'
    '<h1 id="sentinel-title">SENTINEL</h1>'
    '<div class="amx-subtitle">Turn a business question into verified analytical evidence, '
    "an optional internal-document context layer, and a citation-validated decision report.</div>"
    '<div class="amx-chips">'
    f'<span class="amx-chip">{source_label}</span>'
    f'<span class="amx-chip">{minimum_date:%b %Y} – {maximum_date:%b %Y}</span>'
    f'<span class="amx-chip">{len(bundle.tables):,} validated tables</span>'
    '<span class="amx-chip">Calculations stay in Python &amp; SQL</span>'
    "</div></section>",
    unsafe_allow_html=True,
)

st.markdown('<div class="amx-section-label">Guided investigation</div>', unsafe_allow_html=True)
_render_investigation_progress()

with st.container(border=True):
    question_heading, question_help = st.columns((3, 2))
    with question_heading:
        st.subheader("Ask a business question")
        st.caption("Start with the decision you need to make—not a query or model name.")
    with question_help:
        st.info(
            "Sentinel uses the hosted model only to structure your question. Verified analytical "
            "code calculates every business value."
        )
    st.selectbox(
        "Quick-start question",
        (
            "",
            "Why did Q2 profit fall?",
            "Which region has the highest churn?",
            "What caused the cloud-cost spike?",
            "Which customers are at highest churn risk?",
            "Forecast next month's revenue.",
            "Did support quality affect churn?",
        ),
        format_func=lambda value: value or "Choose an example or write your own",
        key="sentinel_example_question",
        on_change=_copy_example_question,
    )
    with st.form("question-understanding-form"):
        business_question = st.text_area(
            "Business question",
            placeholder="Why did European churn increase?",
            max_chars=1_000,
            key="sentinel_business_question",
        )
        understand = st.form_submit_button("Understand question", type="primary")

if understand:
    for key in (
        "sentinel_intent_result",
        "sentinel_workflow_result",
        "sentinel_business_report",
        "sentinel_document_results",
        "sentinel_report_document_query",
        "sentinel_document_query",
    ):
        st.session_state.pop(key, None)
    try:
        if len(business_question.strip()) < 3:
            raise ValueError("Enter a business question with at least three characters")
        context = QuestionContext(
            data_start_date=minimum_date,
            data_end_date=maximum_date,
            regions=list(bundle.regions),
            plans=list(bundle.plans),
            industries=sorted(bundle.tables["customers"].industry.dropna().unique()),
        )
        with st.spinner("Structuring the business question…"):
            result = create_intent_client().understand_question(business_question, context)
        st.session_state["sentinel_intent_result"] = {
            "intent": result.intent.model_dump(mode="json"),
            "provider": result.provider,
            "model": result.model,
            "response_id": result.response_id,
        }
        st.rerun()
    except (LLMConfigurationError, LLMRequestError, LLMResponseError, ValueError) as error:
        st.error(f"Question understanding could not complete: {error}")
        st.info(
            "Configure SENTINEL_LLM_PROVIDER, SENTINEL_LLM_MODEL, and "
            "SENTINEL_LLM_API_KEY in the local .env file. Manual evidence exploration below "
            "remains fully available."
        )

if "sentinel_intent_result" in st.session_state:
    intent_payload = st.session_state["sentinel_intent_result"]
    with st.container(border=True):
        _render_intent_result(intent_payload)
        parsed_intent = QuestionIntent.model_validate(intent_payload["intent"])
        if not parsed_intent.clarification_needed:
            execute_analysis = st.button("Run verified analysis", type="primary")
            if execute_analysis:
                st.session_state.pop("sentinel_business_report", None)
                try:
                    with st.spinner("Running the approved analytical branch…"):
                        workflow_result = run_workflow(
                            bundle,
                            parsed_intent.original_question,
                            intent=parsed_intent,
                        )
                    st.session_state["sentinel_workflow_result"] = workflow_result.model_dump(
                        mode="json"
                    )
                    st.rerun()
                except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
                    st.error(f"The verified workflow could not complete: {error}")

if "sentinel_workflow_result" in st.session_state:
    workflow_payload = st.session_state["sentinel_workflow_result"]
    completed_workflow = WorkflowResult.model_validate(workflow_payload)
    with st.container(border=True):
        _render_workflow_result(workflow_payload, bundle)
        if completed_workflow.status == WorkflowStatus.COMPLETED:
            st.markdown("#### Complete the evidence package")
            with st.expander("Optional · Add internal-document context"):
                st.caption(
                    "Document chunks add cited operational context. They never replace computed "
                    "metrics or prove causation."
                )
                available_documents = _document_search_form(
                    completed_workflow.question,
                    settings=llm_settings,
                    form_key="report-document-search-form",
                    input_key="sentinel_report_document_query",
                    button_label="Find supporting context",
                )
                if available_documents:
                    st.success(
                        f"{len(available_documents)} cited document chunk(s) are ready for the "
                        "report. Review their full text in the Documents tab."
                    )
            available_documents = [
                RetrievedDocumentChunk.model_validate(payload)
                for payload in st.session_state.get("sentinel_document_results", [])
            ]
            report_options, report_action = st.columns((3, 2))
            with report_options:
                use_hosted_narration = st.checkbox(
                    "Use hosted evidence narration",
                    value=llm_ready,
                    disabled=not llm_ready,
                    help=(
                        "The model receives only the verified evidence registry and must return "
                        "strict citations. Leave off for the deterministic offline narrator."
                    ),
                )
                st.caption(
                    f"Report context: {min(len(available_documents), 5)} document chunk(s) · "
                    "claim validation always on"
                )
            with report_action:
                build_report = st.button("Build evidence report", type="primary")
            if build_report:
                try:
                    narrator = (
                        create_report_narrator(llm_settings) if use_hosted_narration else None
                    )
                    with st.spinner("Registering evidence and validating every report claim…"):
                        report = build_report_from_workflow(
                            completed_workflow,
                            documents=available_documents,
                            narrator=narrator,
                        )
                    st.session_state["sentinel_business_report"] = report.model_dump(mode="json")
                    st.rerun()
                except (
                    LLMConfigurationError,
                    ReportNotAvailableError,
                    ReportRequestError,
                    ReportResponseError,
                    TypeError,
                    ValueError,
                ) as error:
                    st.error(f"The evidence report could not be built: {error}")

if "sentinel_business_report" in st.session_state:
    with st.container(border=True):
        _render_business_report(st.session_state["sentinel_business_report"])

st.divider()
st.markdown('<div class="amx-section-label">Analytical evidence library</div>', unsafe_allow_html=True)
st.header("Explore the validated business evidence")
st.caption(
    "Use the tabs for manual analysis or to audit an investigation. Sidebar filters apply only "
    "where the data supports them; company-wide artifacts are explicitly labelled."
)

tabs = st.tabs(
    (
        "Overview",
        "Revenue",
        "Churn & ML",
        "Costs",
        "Forecast",
        "Anomalies",
        "Documents",
        "Methodology",
    )
)

with tabs[0]:
    st.subheader("Executive overview")
    _metric_row(view)
    st.caption(
        "Revenue growth compares the first and last visible month. Cost attribution supports "
        "date and region filters; plan filters do not allocate shared infrastructure cost."
    )
    left, right = st.columns(2)
    with left:
        _plot(revenue_trend_chart(view.revenue), key="overview-revenue")
    with right:
        _plot(churn_trend_chart(view.churn), key="overview-churn")
    _plot(cloud_cost_chart(view.costs), key="overview-costs")

with tabs[1]:
    st.subheader("Revenue performance")
    first, second, third = st.columns(3)
    first.metric("Visible net revenue", _money(view.net_revenue))
    second.metric("Period growth", _percent(view.revenue_growth))
    third.metric("Average discount", f"{view.revenue.discount_rate.mean():.1%}")
    _plot(revenue_trend_chart(view.revenue), key="revenue-trend")
    segment = st.radio("Compare revenue by", ("Region", "Plan"), horizontal=True)
    if segment == "Region":
        _plot(
            segment_performance_chart(view.region_metrics, "region"),
            key="revenue-regions",
        )
        st.dataframe(view.region_metrics, width="stretch", hide_index=True)
    else:
        _plot(
            segment_performance_chart(view.plan_metrics, "plan_type"),
            key="revenue-plans",
        )
        st.dataframe(view.plan_metrics, width="stretch", hide_index=True)

with tabs[2]:
    st.subheader("Subscription churn and held-out model predictions")
    _plot(churn_trend_chart(view.churn), key="churn-trend")
    report = bundle.churn_report
    held_out = report["held_out_test_metrics"]
    model_columns = st.columns(5)
    model_columns[0].metric("Average precision", f"{held_out['average_precision']:.3f}")
    model_columns[1].metric("Recall", f"{held_out['recall']:.1%}")
    model_columns[2].metric("Precision", f"{held_out['precision']:.1%}")
    model_columns[3].metric("F1", f"{held_out['f1']:.3f}")
    model_columns[4].metric("ROC-AUC", f"{held_out['roc_auc']:.3f}")
    _plot(churn_model_evaluation_chart(report), key="churn-model-evaluation")

    default_threshold = float(report["threshold_selection"]["selected_threshold"])
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=default_threshold,
        step=0.01,
        help="The validated operating threshold was selected from out-of-fold training predictions.",
    )
    predictions = filter_churn_predictions(
        bundle.churn_predictions,
        threshold=threshold,
        regions=selected_regions,
        plans=selected_plans,
    )
    flagged = predictions[predictions.predicted_churn]
    st.markdown(
        f'<div class="amx-note"><strong>{len(flagged):,}</strong> of '
        f"<strong>{len(predictions):,}</strong> visible held-out customers meet the selected "
        "risk threshold. These are prioritization signals, not certainties.</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        flagged.head(100),
        width="stretch",
        hide_index=True,
        column_order=("customer_id", "region", "primary_plan", "churn_probability"),
    )
    with st.expander("Model limitations"):
        _limitations(report["limitations"])

with tabs[3]:
    st.subheader("Infrastructure costs")
    cost_columns = st.columns(3)
    cost_columns[0].metric("Visible infrastructure cost", _money(view.infrastructure_cost))
    cost_columns[1].metric("Cost / net revenue", _percent(view.cost_to_revenue_ratio))
    cost_columns[2].metric("Gross contribution", _money(view.contribution))
    _plot(cloud_cost_chart(view.costs), key="cost-composition")
    cost_table = view.costs.copy()
    cost_table["month"] = pd.to_datetime(cost_table.month).dt.strftime("%b %Y")
    st.dataframe(cost_table, width="stretch", hide_index=True)

with tabs[4]:
    st.subheader("Monthly net-revenue forecast")
    forecast_report = bundle.forecast_report
    test_metrics = forecast_report["selected_test_metrics"]
    forecast_columns = st.columns(4)
    forecast_columns[0].metric("Selected method", forecast_report["selected_method"].replace("_", " ").title())
    forecast_columns[1].metric("Test WAPE", f"{test_metrics['wape']:.3%}")
    forecast_columns[2].metric("Test MAE", _money(test_metrics["mae"]))
    forecast_columns[3].metric(
        "Improvement vs naïve", f"{forecast_report['test_wape_improvement_over_naive']:.1%}"
    )
    _plot(
        forecast_evidence_chart(
            bundle.revenue_history,
            bundle.forecast_backtest,
            bundle.future_forecast,
            selected_method=forecast_report["selected_method"],
        ),
        key="revenue-forecast",
    )
    st.caption("Forecasts are company-wide point estimates and do not inherit dashboard filters.")
    st.dataframe(bundle.future_forecast, width="stretch", hide_index=True)
    with st.expander("Forecast limitations"):
        _limitations(forecast_report["limitations"])

with tabs[5]:
    st.subheader("GPU-cost anomaly evidence")
    anomaly_report = bundle.anomaly_report
    counts = anomaly_report["counts"]
    anomaly_columns = st.columns(4)
    anomaly_columns[0].metric("Daily observations", f"{anomaly_report['data']['daily_observations']:,}")
    anomaly_columns[1].metric("IQR candidates", f"{counts['iqr_candidates']:,}")
    anomaly_columns[2].metric("Isolation Forest", f"{counts['isolation_forest_anomalies']:,}")
    anomaly_columns[3].metric("Strong agreement", f"{counts['final_strong_anomalies']:,}")
    _plot(anomaly_evidence_chart(bundle.anomaly_scores), key="anomaly-evidence")
    st.caption(
        "Anomaly evidence is company-wide and retrospective. It does not inherit regional, plan, "
        "or date filters."
    )
    strong_scores = bundle.anomaly_scores[
        bundle.anomaly_scores.strong_anomaly.astype(str).str.lower().eq("true")
    ]
    st.dataframe(
        strong_scores[
            ["date", "observed_value", "normal_range_lower", "normal_range_upper", "anomaly_score"]
        ],
        width="stretch",
        hide_index=True,
    )
    with st.expander("Anomaly limitations"):
        _limitations(anomaly_report["limitations"])

with tabs[6]:
    st.subheader("Supporting AMX Tech documents")
    st.caption(
        "Semantic retrieval returns company context with source metadata. It does not calculate "
        "metrics. Retrieved chunks can be included in an evidence report, where document-only "
        "claims remain contextual."
    )
    document_results = _document_search_form(
        business_question.strip(),
        settings=llm_settings,
        form_key="document-search-form",
        input_key="sentinel_document_query",
        button_label="Search documents",
    )
    if "sentinel_document_results" in st.session_state:
        if not document_results:
            st.info("No document chunks met the configured similarity threshold.")
        for document_result in document_results:
            _render_document_result(document_result)
            st.divider()

with tabs[7]:
    st.subheader("Methodology and evidence boundaries")
    st.markdown(
        "Sentinel computes numerical values from validated Pandas/SQL-compatible analytics and "
        "persisted model evidence. The reporting layer registers those results with optional "
        "document context, then validates citations, copied numbers, causal language, and "
        "qualitative confidence."
    )
    st.markdown("#### Data quality")
    quality_columns = st.columns(3)
    quality_columns[0].metric(
        "Duplicate ticket events removed", f"{bundle.cleaning.duplicate_events_removed:,}"
    )
    quality_columns[1].metric("Missing industries filled", f"{bundle.cleaning.industries_filled:,}")
    quality_columns[2].metric(
        "Missing satisfaction retained",
        f"{bundle.cleaning.satisfaction_values_retained_missing:,}",
    )
    st.markdown("#### Analytical guardrails")
    st.markdown(
        "- Ground-truth scenario metadata is never loaded by the dashboard.\n"
        "- Churn predictions shown here come only from the held-out evaluation cohort.\n"
        "- Feature importance and anomaly detection do not establish causation.\n"
        "- Forecasts are point estimates from 24 synthetic monthly observations.\n"
        "- Shared infrastructure costs cannot be attributed to subscription plans.\n"
        "- Retrieved documents are supporting context, not authoritative metric sources.\n"
        "- Hosted narration cannot add evidence IDs or calculate new values."
    )
