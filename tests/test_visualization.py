import json

import pandas as pd
import plotly.graph_objects as go
import pytest

from sentinel.visualization import (
    anomaly_evidence_chart,
    churn_model_evaluation_chart,
    churn_trend_chart,
    cloud_cost_chart,
    export_chart_gallery,
    forecast_evidence_chart,
    revenue_trend_chart,
    segment_performance_chart,
)


@pytest.fixture
def months():
    return pd.date_range("2024-01-01", periods=6, freq="MS")


def _assert_serializable_figure(figure, title):
    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text == title
    payload = json.loads(figure.to_json())
    assert payload["data"]
    assert payload["layout"]["template"]


def test_revenue_chart_contract_and_no_input_mutation(months):
    frame = pd.DataFrame(
        {
            "month": months,
            "gross_revenue": [120, 125, 130, 132, 136, 140],
            "discount_amount": [12, 13, 14, 15, 16, 17],
            "net_revenue": [108, 112, 116, 117, 120, 123],
        }
    )
    original = frame.copy(deep=True)
    figure = revenue_trend_chart(frame)
    _assert_serializable_figure(
        figure, "AMX Tech monthly revenue and discount trend"
    )
    assert [trace.name for trace in figure.data] == [
        "Gross revenue",
        "Net revenue",
        "Discount rate",
    ]
    pd.testing.assert_frame_equal(frame, original)


def test_churn_chart_contract(months):
    frame = pd.DataFrame(
        {
            "month": months,
            "subscriptions_at_risk": [100] * 6,
            "cancellations": [2, 3, 4, 3, 6, 4],
            "churn_rate": [0.02, 0.03, 0.04, 0.03, 0.06, 0.04],
        }
    )
    figure = churn_trend_chart(frame)
    _assert_serializable_figure(figure, "AMX Tech monthly subscription churn")
    assert [trace.type for trace in figure.data] == ["scatter", "bar"]


def test_cloud_cost_chart_uses_all_components(months):
    frame = pd.DataFrame(
        {
            "month": months,
            "compute_cost": [40] * 6,
            "storage_cost": [10] * 6,
            "network_cost": [8] * 6,
            "gpu_cost": [20, 21, 22, 45, 23, 22],
            "total_cost": [78, 79, 80, 103, 81, 80],
        }
    )
    figure = cloud_cost_chart(frame)
    _assert_serializable_figure(
        figure, "AMX Tech monthly cloud-cost composition"
    )
    assert {trace.name for trace in figure.data} == {
        "Compute",
        "Storage",
        "Network",
        "GPU",
    }
    assert all(trace.stackgroup == "cost" for trace in figure.data)


def test_anomaly_chart_marks_only_strong_events():
    dates = pd.date_range("2024-05-10", periods=7, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "normal_range_lower": [60] * 7,
            "normal_range_upper": [85] * 7,
            "observed_value": [72, 75, 73, 150, 155, 74, 71],
            "anomaly_score": [-0.2, -0.1, -0.2, 0.1, 0.12, -0.1, -0.2],
            "strong_anomaly": [False, False, False, True, True, False, False],
        }
    )
    figure = anomaly_evidence_chart(frame)
    _assert_serializable_figure(figure, "AMX Tech GPU-cost anomaly evidence")
    strong_trace = next(trace for trace in figure.data if trace.name == "Strong anomaly")
    assert len(strong_trace.x) == 2
    assert strong_trace.marker.symbol == "diamond"


def test_churn_model_evaluation_combines_baseline_and_confusion_evidence():
    report = {
        "selected_model": "gradient_boosting",
        "cross_validation": {
            "dummy": {"average_precision": {"mean": 0.1, "std": 0.0}},
            "gradient_boosting": {
                "average_precision": {"mean": 0.7, "std": 0.04}
            },
        },
        "held_out_test_metrics": {
            "confusion_matrix": {
                "true_negative": 90,
                "false_positive": 5,
                "false_negative": 3,
                "true_positive": 12,
            }
        },
    }
    figure = churn_model_evaluation_chart(report)
    _assert_serializable_figure(
        figure,
        "AMX Tech churn-model evaluation — selected: gradient boosting",
    )
    assert [trace.type for trace in figure.data] == ["bar", "heatmap"]
    assert figure.data[1].z[1][1] == 12


def test_forecast_chart_separates_observed_test_and_future(months):
    history = pd.Series([100, 103, 106, 108, 110, 112], index=months)
    backtest = pd.DataFrame(
        {
            "phase": ["test", "test"],
            "month": months[-2:],
            "method": ["exponential_smoothing"] * 2,
            "actual_net_revenue": [110, 112],
            "forecast_net_revenue": [109, 113],
        }
    )
    future = pd.DataFrame(
        {
            "month": pd.date_range("2024-07-01", periods=3, freq="MS"),
            "method": ["exponential_smoothing"] * 3,
            "horizon_month": [1, 2, 3],
            "forecast_net_revenue": [114, 116, 118],
        }
    )
    figure = forecast_evidence_chart(
        history,
        backtest,
        future,
        selected_method="exponential_smoothing",
    )
    _assert_serializable_figure(
        figure,
        "AMX Tech monthly revenue forecast — exponential smoothing",
    )
    assert [trace.name for trace in figure.data] == [
        "Observed net revenue",
        "One-step test forecast",
        "Future point forecast",
    ]


@pytest.mark.parametrize("dimension", ["region", "plan_type"])
def test_segment_chart_uses_aligned_scales(dimension):
    frame = pd.DataFrame(
        {
            dimension: ["A", "B", "C"],
            "net_revenue": [100, 150, 80],
            "churn_rate": [0.04, 0.08, 0.03],
        }
    )
    figure = segment_performance_chart(frame, dimension)
    _assert_serializable_figure(
        figure, f"AMX Tech performance by {dimension.replace('_', ' ')}"
    )
    assert [trace.type for trace in figure.data] == ["bar", "scatter"]


def test_gallery_export_writes_standalone_charts_and_index(tmp_path, months):
    revenue = pd.DataFrame(
        {
            "month": months,
            "gross_revenue": [120, 125, 130, 132, 136, 140],
            "discount_amount": [12, 13, 14, 15, 16, 17],
            "net_revenue": [108, 112, 116, 117, 120, 123],
        }
    )
    figures = {"monthly-revenue": revenue_trend_chart(revenue)}
    outputs = export_chart_gallery(figures, tmp_path)
    assert outputs["monthly-revenue"].exists()
    assert outputs["gallery"].exists()
    assert "Plotly.newPlot" in outputs["monthly-revenue"].read_text(encoding="utf-8")
    gallery = outputs["gallery"].read_text(encoding="utf-8")
    assert "AMX Tech Sentinel visualization gallery" in gallery
    assert "monthly revenue and discount trend" in gallery


def test_chart_inputs_and_gallery_names_are_validated(months, tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        revenue_trend_chart(pd.DataFrame({"month": months, "net_revenue": 1}))
    with pytest.raises(ValueError, match="dimension must be"):
        segment_performance_chart(pd.DataFrame({"x": [1]}), "industry")
    figure = go.Figure(go.Scatter(x=[1], y=[1]))
    with pytest.raises(ValueError, match="lowercase hyphenated ASCII"):
        export_chart_gallery({"Bad Name": figure}, tmp_path)
