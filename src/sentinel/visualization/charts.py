"""Reusable Plotly charts for AMX Tech analytical evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

COLORS = {
    "blue": "#2563EB",
    "teal": "#0F9D8A",
    "amber": "#E99A2D",
    "red": "#D1495B",
    "purple": "#7C5CE5",
    "slate": "#64748B",
    "grid": "#E2E8F0",
    "text": "#172033",
    "muted": "#5B6474",
    "background": "#FFFFFF",
}

EXPORT_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _prepared_frame(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{label} must be a non-empty DataFrame")
    prepared = frame.copy()
    prepared.columns = ["".join(column) for column in prepared.columns]
    missing = required - set(prepared.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    return prepared


def _apply_layout(
    figure: go.Figure,
    *,
    title: str,
    height: int = 480,
    hovermode: str | None = None,
) -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        height=height,
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "color": COLORS["text"]},
        hovermode=hovermode,
        hoverlabel={"bgcolor": COLORS["background"], "font_color": COLORS["text"]},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "title": None,
        },
        margin={"l": 72, "r": 36, "t": 88, "b": 64},
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor=COLORS["grid"],
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["text"]},
    )
    figure.update_yaxes(
        gridcolor=COLORS["grid"],
        zeroline=False,
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["text"]},
    )
    return figure


def revenue_trend_chart(monthly: pd.DataFrame) -> go.Figure:
    """Show gross/net revenue together with the monthly discount rate."""
    frame = _prepared_frame(
        monthly,
        {"month", "gross_revenue", "discount_amount", "net_revenue"},
        "monthly revenue",
    )
    frame["month"] = pd.to_datetime(frame.month, errors="raise")
    frame = frame.sort_values("month")
    frame["discount_rate"] = frame.discount_amount / frame.gross_revenue
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
    )
    for column, name, color in (
        ("gross_revenue", "Gross revenue", COLORS["blue"]),
        ("net_revenue", "Net revenue", COLORS["teal"]),
    ):
        figure.add_trace(
            go.Scatter(
                x=frame.month,
                y=frame[column],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2.5},
                marker={"size": 6},
                hovertemplate=f"%{{x|%b %Y}}<br>{name}: %{{y:,.0f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Bar(
            x=frame.month,
            y=frame.discount_rate,
            name="Discount rate",
            marker_color=COLORS["amber"],
            hovertemplate="%{x|%b %Y}<br>Discount rate: %{y:.1%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="Revenue", tickformat="~s", row=1, col=1)
    figure.update_yaxes(title_text="Discount", tickformat=".0%", row=2, col=1)
    figure.update_xaxes(title_text="Month", row=2, col=1)
    return _apply_layout(
        figure,
        title="AMX Tech monthly revenue and discount trend",
        height=560,
        hovermode="x unified",
    )


def churn_trend_chart(monthly: pd.DataFrame) -> go.Figure:
    """Show exposed-subscription churn rates and cancellation counts over time."""
    frame = _prepared_frame(
        monthly,
        {"month", "subscriptions_at_risk", "cancellations", "churn_rate"},
        "monthly churn",
    )
    frame["month"] = pd.to_datetime(frame.month, errors="raise")
    frame = frame.sort_values("month")
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.08,
    )
    figure.add_trace(
        go.Scatter(
            x=frame.month,
            y=frame.churn_rate,
            mode="lines+markers",
            name="Churn rate",
            line={"color": COLORS["red"], "width": 2.5},
            customdata=frame[["subscriptions_at_risk"]],
            hovertemplate=(
                "%{x|%b %Y}<br>Churn: %{y:.2%}<br>At risk: %{customdata[0]:,.0f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=frame.month,
            y=frame.cancellations,
            name="Cancellations",
            marker_color=COLORS["slate"],
            hovertemplate="%{x|%b %Y}<br>Cancellations: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="Churn rate", tickformat=".1%", row=1, col=1)
    figure.update_yaxes(title_text="Cancellations", row=2, col=1)
    figure.update_xaxes(title_text="Month", row=2, col=1)
    return _apply_layout(
        figure,
        title="AMX Tech monthly subscription churn",
        height=550,
        hovermode="x unified",
    )


def cloud_cost_chart(monthly: pd.DataFrame) -> go.Figure:
    """Show monthly infrastructure-cost composition as a stacked time series."""
    components = {
        "compute_cost": ("Compute", COLORS["blue"]),
        "storage_cost": ("Storage", COLORS["teal"]),
        "network_cost": ("Network", COLORS["amber"]),
        "gpu_cost": ("GPU", COLORS["purple"]),
    }
    frame = _prepared_frame(
        monthly,
        {"month", "total_cost", *components},
        "monthly cloud costs",
    )
    frame["month"] = pd.to_datetime(frame.month, errors="raise")
    frame = frame.sort_values("month")
    figure = go.Figure()
    for column, (name, color) in components.items():
        figure.add_trace(
            go.Scatter(
                x=frame.month,
                y=frame[column],
                mode="lines",
                name=name,
                stackgroup="cost",
                line={"color": color, "width": 1.5},
                hovertemplate=f"%{{x|%b %Y}}<br>{name}: %{{y:,.0f}}<extra></extra>",
            )
        )
    figure.update_xaxes(title_text="Month")
    figure.update_yaxes(title_text="Infrastructure cost", tickformat="~s")
    return _apply_layout(
        figure,
        title="AMX Tech monthly cloud-cost composition",
        height=500,
        hovermode="x unified",
    )


def anomaly_evidence_chart(scores: pd.DataFrame) -> go.Figure:
    """Show daily observed GPU cost, its robust range, and final strong anomalies."""
    frame = _prepared_frame(
        scores,
        {
            "date",
            "normal_range_lower",
            "normal_range_upper",
            "observed_value",
            "anomaly_score",
            "strong_anomaly",
        },
        "anomaly scores",
    )
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    frame = frame.sort_values("date")
    if frame.strong_anomaly.dtype != bool:
        frame["strong_anomaly"] = frame.strong_anomaly.astype(str).str.lower().eq("true")
    strong = frame[frame.strong_anomaly]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame.date,
            y=frame.normal_range_lower,
            mode="lines",
            line={"width": 0},
            name="Normal range",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame.date,
            y=frame.normal_range_upper,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.13)",
            name="Robust normal range",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame.date,
            y=frame.observed_value,
            mode="lines",
            name="Observed GPU cost",
            line={"color": COLORS["blue"], "width": 2},
            customdata=np.column_stack(
                [frame.normal_range_lower, frame.normal_range_upper]
            ),
            hovertemplate=(
                "%{x|%d %b %Y}<br>Observed: %{y:,.0f}<br>Normal: "
                "%{customdata[0]:,.0f}–%{customdata[1]:,.0f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=strong.date,
            y=strong.observed_value,
            mode="markers",
            name="Strong anomaly",
            marker={"color": COLORS["red"], "size": 9, "symbol": "diamond"},
            customdata=strong[["anomaly_score"]],
            hovertemplate=(
                "%{x|%d %b %Y}<br>Observed: %{y:,.0f}<br>Anomaly score: "
                "%{customdata[0]:.3f}<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title_text="Date")
    figure.update_yaxes(title_text="Daily GPU cost", tickformat="~s")
    return _apply_layout(
        figure,
        title="AMX Tech GPU-cost anomaly evidence",
        height=500,
        hovermode="x unified",
    )


def churn_model_evaluation_chart(report: dict[str, Any]) -> go.Figure:
    """Compare churn candidates and show the selected model's confusion matrix."""
    required = {"cross_validation", "held_out_test_metrics", "selected_model"}
    missing = required - set(report)
    if missing:
        raise ValueError(f"Churn report is missing keys: {sorted(missing)}")
    candidates = list(report["cross_validation"])
    means = [
        report["cross_validation"][name]["average_precision"]["mean"]
        for name in candidates
    ]
    deviations = [
        report["cross_validation"][name]["average_precision"]["std"]
        for name in candidates
    ]
    matrix = report["held_out_test_metrics"]["confusion_matrix"]
    values = [
        [matrix["true_negative"], matrix["false_positive"]],
        [matrix["false_negative"], matrix["true_positive"]],
    ]
    labels = [[f"TN<br>{values[0][0]}", f"FP<br>{values[0][1]}"], [f"FN<br>{values[1][0]}", f"TP<br>{values[1][1]}"]]
    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.16,
        subplot_titles=("Cross-validation average precision", "Held-out confusion matrix"),
    )
    colors = [
        COLORS["teal"] if name == report["selected_model"] else COLORS["blue"]
        for name in candidates
    ]
    figure.add_trace(
        go.Bar(
            x=[name.replace("_", " ").title() for name in candidates],
            y=means,
            marker_color=colors,
            error_y={"type": "data", "array": deviations, "visible": True},
            name="CV average precision",
            hovertemplate="%{x}<br>AP: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            z=values,
            x=["Predicted retained", "Predicted churned"],
            y=["Actual retained", "Actual churned"],
            text=labels,
            texttemplate="%{text}",
            colorscale=[[0, "#EFF6FF"], [1, COLORS["blue"]]],
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>Customers: %{z}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(title_text="Average precision", range=[0, 1], row=1, col=1)
    figure.update_xaxes(tickangle=-18, row=1, col=1)
    return _apply_layout(
        figure,
        title=(
            "AMX Tech churn-model evaluation — selected: "
            f"{report['selected_model'].replace('_', ' ')}"
        ),
        height=500,
    )


def forecast_evidence_chart(
    history: pd.Series,
    backtest: pd.DataFrame,
    future: pd.DataFrame,
    *,
    selected_method: str,
) -> go.Figure:
    """Show observed revenue, selected test predictions, and future point forecasts."""
    if not isinstance(history, pd.Series) or history.empty:
        raise ValueError("history must be a non-empty monthly Series")
    observed = history.copy()
    observed.index = pd.to_datetime(observed.index, errors="raise")
    tested = _prepared_frame(
        backtest,
        {"phase", "month", "method", "actual_net_revenue", "forecast_net_revenue"},
        "forecast backtest",
    )
    projected = _prepared_frame(
        future,
        {"month", "method", "horizon_month", "forecast_net_revenue"},
        "future forecast",
    )
    tested["month"] = pd.to_datetime(tested.month, errors="raise")
    projected["month"] = pd.to_datetime(projected.month, errors="raise")
    selected_test = tested[
        tested.phase.eq("test") & tested.method.eq(selected_method)
    ].sort_values("month")
    selected_future = projected[projected.method.eq(selected_method)].sort_values("month")
    if selected_test.empty or selected_future.empty:
        raise ValueError("Selected method is absent from test or future forecasts")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=observed.index,
            y=observed.to_numpy(),
            mode="lines+markers",
            name="Observed net revenue",
            line={"color": COLORS["blue"], "width": 2.5},
            hovertemplate="%{x|%b %Y}<br>Observed: %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=selected_test.month,
            y=selected_test.forecast_net_revenue,
            mode="lines+markers",
            name="One-step test forecast",
            line={"color": COLORS["amber"], "width": 2, "dash": "dash"},
            hovertemplate="%{x|%b %Y}<br>Test forecast: %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=selected_future.month,
            y=selected_future.forecast_net_revenue,
            mode="lines+markers",
            name="Future point forecast",
            line={"color": COLORS["teal"], "width": 2.5, "dash": "dot"},
            marker={"size": 8, "symbol": "diamond"},
            hovertemplate="%{x|%b %Y}<br>Forecast: %{y:,.0f}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="Month")
    figure.update_yaxes(title_text="Monthly net revenue", tickformat="~s")
    return _apply_layout(
        figure,
        title=(
            "AMX Tech monthly revenue forecast — "
            f"{selected_method.replace('_', ' ')}"
        ),
        height=500,
        hovermode="x unified",
    )


def segment_performance_chart(metrics: pd.DataFrame, dimension: str) -> go.Figure:
    """Compare segment revenue and churn on separate aligned scales."""
    if dimension not in {"region", "plan_type"}:
        raise ValueError("dimension must be 'region' or 'plan_type'")
    frame = _prepared_frame(
        metrics,
        {dimension, "net_revenue", "churn_rate"},
        "segment metrics",
    ).sort_values("net_revenue", ascending=True)
    categories = frame[dimension].astype(str)
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=("Net revenue", "Subscription churn rate"),
    )
    figure.add_trace(
        go.Bar(
            x=frame.net_revenue,
            y=categories,
            orientation="h",
            name="Net revenue",
            marker_color=COLORS["blue"],
            hovertemplate="%{y}<br>Net revenue: %{x:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=frame.churn_rate,
            y=categories,
            mode="markers",
            name="Churn rate",
            marker={"color": COLORS["red"], "size": 11, "symbol": "diamond"},
            hovertemplate="%{y}<br>Churn rate: %{x:.2%}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Net revenue", tickformat="~s", row=1, col=1)
    figure.update_xaxes(title_text="Churn rate", tickformat=".1%", row=1, col=2)
    return _apply_layout(
        figure,
        title=f"AMX Tech performance by {dimension.replace('_', ' ')}",
        height=max(420, 80 * len(frame) + 180),
    )


def export_chart_gallery(
    figures: dict[str, go.Figure],
    output_dir: Path,
) -> dict[str, Path]:
    """Write standalone chart HTML files plus one compact review gallery."""
    if not figures:
        raise ValueError("At least one figure is required")
    invalid = [name for name in figures if not re.fullmatch(r"[a-z0-9-]+", name)]
    if invalid:
        raise ValueError(f"Figure names must be lowercase hyphenated ASCII: {invalid}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    fragments: list[str] = []
    for index, (name, figure) in enumerate(figures.items()):
        if not isinstance(figure, go.Figure):
            raise TypeError(f"{name!r} is not a Plotly Figure")
        chart_path = output_dir / f"{name}.html"
        figure.write_html(
            chart_path,
            include_plotlyjs="cdn",
            full_html=True,
            config=EXPORT_CONFIG,
        )
        outputs[name] = chart_path
        fragments.append(
            pio.to_html(
                figure,
                include_plotlyjs="cdn" if index == 0 else False,
                full_html=False,
                config=EXPORT_CONFIG,
            )
        )
    sections = "\n".join(
        f'<section aria-label="{name.replace("-", " ")}">{fragment}</section>'
        for (name, _), fragment in zip(figures.items(), fragments)
    )
    gallery = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AMX Tech Sentinel visualization gallery</title>
  <style>
    body {{ margin: 0; padding: 24px; background: #f6f8fb; color: #172033;
      font-family: Inter, Segoe UI, Arial, sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ font-size: 24px; font-weight: 600; margin: 0 0 20px; }}
    section {{ background: white; margin: 0 0 20px; padding: 8px; border: 1px solid #e2e8f0;
      border-radius: 12px; }}
  </style>
</head>
<body><main><h1>AMX Tech Sentinel visualization gallery</h1>{sections}</main></body>
</html>"""
    index_path = output_dir / "index.html"
    index_path.write_text(gallery, encoding="utf-8")
    outputs["gallery"] = index_path
    return outputs
