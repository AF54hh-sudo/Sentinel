from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import pytest

from sentinel.dashboard import (
    filter_churn_predictions,
    load_dashboard_bundle,
    prepare_dashboard_view,
)


@pytest.fixture(scope="module")
def dashboard_bundle():
    return load_dashboard_bundle("csv")


def test_dashboard_bundle_loads_validated_sources(dashboard_bundle) -> None:
    assert dashboard_bundle.source == "csv"
    assert set(dashboard_bundle.tables) == {
        "customers",
        "subscriptions",
        "sales",
        "cloud_costs",
        "support_tickets",
    }
    assert dashboard_bundle.regions == ("APAC", "Europe", "India", "North America")
    assert dashboard_bundle.plans == ("Basic", "Enterprise", "Professional")
    assert dashboard_bundle.churn_report["company"] == "AMX Tech"
    assert dashboard_bundle.forecast_report["company"] == "AMX Tech"
    assert dashboard_bundle.anomaly_report["company"] == "AMX Tech"
    assert len(dashboard_bundle.revenue_history) == 24


def test_full_dashboard_view_reconciles_source_totals(dashboard_bundle) -> None:
    start, end = dashboard_bundle.date_bounds
    view = prepare_dashboard_view(
        dashboard_bundle,
        start_date=start,
        end_date=end,
        regions=dashboard_bundle.regions,
        plans=dashboard_bundle.plans,
    )

    assert len(view.revenue) == 24
    assert view.net_revenue == pytest.approx(
        dashboard_bundle.tables["sales"].net_revenue.sum()
    )
    assert view.infrastructure_cost == pytest.approx(
        dashboard_bundle.tables["cloud_costs"].total_cost.sum()
    )
    assert view.contribution == pytest.approx(view.net_revenue - view.infrastructure_cost)
    assert view.revenue_customers == dashboard_bundle.tables["sales"].customer_id.nunique()


def test_dashboard_view_applies_date_region_and_plan_filters(dashboard_bundle) -> None:
    view = prepare_dashboard_view(
        dashboard_bundle,
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-06-30"),
        regions=("Europe",),
        plans=("Professional",),
    )

    assert view.revenue.month.min() == pd.Timestamp("2024-01-01")
    assert view.revenue.month.max() == pd.Timestamp("2024-06-01")
    assert set(view.region_metrics.region) == {"Europe"}
    assert set(view.plan_metrics.plan_type) == {"Professional"}
    assert 0 < view.infrastructure_cost < dashboard_bundle.tables["cloud_costs"].total_cost.sum()


def test_dashboard_view_does_not_mutate_source_tables(dashboard_bundle) -> None:
    original_sales = dashboard_bundle.tables["sales"].copy(deep=True)
    prepare_dashboard_view(
        dashboard_bundle,
        start_date=pd.Timestamp("2023-01-01"),
        end_date=pd.Timestamp("2023-12-31"),
        regions=("India",),
        plans=("Basic",),
    )
    pd.testing.assert_frame_equal(dashboard_bundle.tables["sales"], original_sales)


@pytest.mark.parametrize(
    ("start", "end", "regions", "plans", "message"),
    [
        ("2024-02-01", "2024-01-01", (), (), "start_date"),
        ("2024-01-01", "2024-02-01", ("Atlantis",), (), "Unknown regions"),
        ("2024-01-01", "2024-02-01", (), ("Unlimited",), "Unknown plans"),
    ],
)
def test_dashboard_view_rejects_invalid_filters(
    dashboard_bundle, start, end, regions, plans, message
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_dashboard_view(
            dashboard_bundle,
            start_date=pd.Timestamp(start),
            end_date=pd.Timestamp(end),
            regions=regions,
            plans=plans,
        )


def test_churn_predictions_are_filtered_ranked_and_thresholded(dashboard_bundle) -> None:
    result = filter_churn_predictions(
        dashboard_bundle.churn_predictions,
        threshold=0.75,
        regions=("Europe",),
        plans=("Professional",),
    )

    assert set(result.region) == {"Europe"}
    assert set(result.primary_plan) == {"Professional"}
    assert result.churn_probability.is_monotonic_decreasing
    assert result.predicted_churn.equals(result.churn_probability.ge(0.75))


def test_churn_prediction_contract_rejects_bad_threshold(dashboard_bundle) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        filter_churn_predictions(dashboard_bundle.churn_predictions, threshold=1.1)


def test_dashboard_entry_point_keeps_rag_separate_from_ground_truth() -> None:
    source = Path("app.py").read_text(encoding="utf-8")
    assert "ground_truth" not in source
    assert "sentinel.rag" in source
    assert "create_intent_client" in source
    assert "run_workflow" in source
    assert "AMX Tech" in source


def test_streamlit_theme_locks_readable_light_contrast() -> None:
    config_path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    with config_path.open("rb") as stream:
        theme = tomllib.load(stream)["theme"]

    assert theme["base"] == "light"
    assert theme["backgroundColor"] == "#F5F7FB"
    assert theme["secondaryBackgroundColor"] == "#FFFFFF"
    assert theme["textColor"] == "#172033"

    source = Path("app.py").read_text(encoding="utf-8")
    assert '[data-testid="stMetricValue"]' in source
    assert 'color: #111827 !important' in source


def test_streamlit_dashboard_smoke() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = testing.AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert any(
        'id="sentinel-title">SENTINEL</h1>' in block.value for block in app.markdown
    )
    assert any("step 1 of 4 · Business question" in item.value for item in app.caption)
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Revenue",
        "Churn & ML",
        "Costs",
        "Forecast",
        "Anomalies",
        "Documents",
        "Methodology",
    ]


def test_streamlit_runs_verified_workflow_from_validated_intent() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = testing.AppTest.from_file(app_path, default_timeout=30)
    app.session_state["sentinel_intent_result"] = {
        "intent": {
            "original_question": "Forecast next month's revenue",
            "intent": "forecast_analysis",
            "metrics": ["revenue_forecast"],
            "analysis_mode": "forecast",
            "time_period": {
                "label": "Next month after observed data",
                "start_date": None,
                "end_date": None,
                "granularity": "month",
                "comparison_label": None,
                "comparison_start_date": None,
                "comparison_end_date": None,
            },
            "filters": {
                "regions": [],
                "plans": [],
                "industries": [],
                "customer_ids": [],
            },
            "asks_for_cause": False,
            "clarification_needed": False,
            "clarification_question": None,
            "interpretation": "Forecast company-wide net revenue for the next month.",
            "confidence": 0.98,
        },
        "provider": "test-provider",
        "model": "test-model",
        "response_id": "test-response",
    }
    app.run()
    workflow_button = next(button for button in app.button if button.label == "Run verified analysis")
    workflow_button.click().run()

    assert not app.exception
    workflow = app.session_state["sentinel_workflow_result"]
    assert workflow["status"] == "completed"
    assert workflow["selected_tool"] == "revenue_forecast"
    assert workflow["evidence"]["verified"] is True
    assert any("step 3 of 4 · Verified analysis" in item.value for item in app.caption)

    report_button = next(button for button in app.button if button.label == "Build evidence report")
    report_button.click().run()
    report = app.session_state["sentinel_business_report"]
    assert report["validated"] is True
    assert report["narration_provider"] == "deterministic"
    assert report["evidence"]["items"][0]["evidence_id"] == "FORECAST-001"
    assert any("step 4 of 4 · Evidence report" in item.value for item in app.caption)


def test_streamlit_renders_citation_preserving_document_result() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = testing.AppTest.from_file(app_path, default_timeout=30)
    app.session_state["sentinel_document_results"] = [
        {
            "chunk_id": "DOC-003-C001",
            "document_id": "DOC-003",
            "document_title": "Cloud Infrastructure Incident Review",
            "document_date": "2024-05-24",
            "category": "incident_review",
            "regions": ["India", "North America", "Europe", "APAC"],
            "topics": ["gpu_costs", "anomaly"],
            "source_path": "03_cloud_infrastructure_incident.md",
            "section": "GPU capacity event, response, and limitations",
            "page": 1,
            "content": "Reserved GPU workers stayed active after workloads shifted.",
            "similarity_score": 0.91,
            "business_relevance": "Incident Review · company-wide · gpu costs, anomaly",
        }
    ]
    app.run()

    assert not app.exception
    assert any(
        "Cloud Infrastructure Incident Review" in heading.value for heading in app.markdown
    )


def test_streamlit_can_clear_an_investigation_session() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = testing.AppTest.from_file(app_path, default_timeout=30)
    app.session_state["sentinel_document_results"] = []
    app.run()

    reset_button = next(
        button for button in app.button if button.label == "Start new investigation"
    )
    reset_button.click().run()

    assert not app.exception
    assert "sentinel_document_results" not in app.session_state
    assert any("step 1 of 4 · Business question" in item.value for item in app.caption)


def test_quick_start_question_populates_input_and_clears_stale_context() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = testing.AppTest.from_file(app_path, default_timeout=30)
    app.session_state["sentinel_document_results"] = []
    app.run()

    question = "Why did Q2 profit fall?"
    quick_start = next(
        item for item in app.selectbox if item.label == "Quick-start question"
    )
    quick_start.select(question).run()

    business_question = next(
        item for item in app.text_area if item.label == "Business question"
    )
    assert not app.exception
    assert business_question.value == question
    assert "sentinel_document_results" not in app.session_state
