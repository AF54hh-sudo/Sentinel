import numpy as np
import pandas as pd
import pytest

from sentinel.forecasting import (
    FORECAST_METHODS,
    RevenueForecastConfig,
    build_monthly_net_revenue,
    forecast_metrics,
    rolling_origin_predictions,
    run_revenue_forecast,
)


@pytest.fixture(scope="module")
def synthetic_sales():
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    index = np.arange(len(months))
    monthly = 1_000_000 * (1 + 0.012 * index) * (
        1 + 0.018 * np.sin(2 * np.pi * index / 12)
    )
    rows = []
    for month, value in zip(months, monthly):
        rows.extend(
            [
                {"sale_date": month, "net_revenue": value * 0.55},
                {"sale_date": month, "net_revenue": value * 0.45},
            ]
        )
    return pd.DataFrame(rows)


def test_monthly_series_is_complete_aggregated_and_positive(synthetic_sales):
    monthly = build_monthly_net_revenue(synthetic_sales)
    assert len(monthly) == 24
    assert monthly.index.freqstr == "MS"
    assert monthly.index[0] == pd.Timestamp("2023-01-01")
    assert monthly.index[-1] == pd.Timestamp("2024-12-01")
    assert monthly.gt(0).all()


def test_forecast_metric_contract():
    metrics = forecast_metrics(np.array([100.0, 200.0]), np.array([90.0, 220.0]))
    assert metrics["mae"] == 15.0
    assert metrics["rmse"] == pytest.approx(np.sqrt(250))
    assert metrics["wape"] == 0.1
    with pytest.raises(ValueError, match="same non-zero shape"):
        forecast_metrics(np.array([1.0]), np.array([1.0, 2.0]))


def test_naive_rolling_origin_uses_only_previous_observation(synthetic_sales):
    series = build_monthly_net_revenue(synthetic_sales)
    predictions = rolling_origin_predictions(series, 12, 18, "naive")
    expected = series.iloc[11:17].to_numpy()
    np.testing.assert_allclose(predictions.to_numpy(), expected)


def test_target_month_cannot_change_its_own_prediction(synthetic_sales):
    series = build_monthly_net_revenue(synthetic_sales)
    original = rolling_origin_predictions(series, 12, 14, "naive")
    modified = series.copy()
    modified.iloc[13] *= 100
    changed = rolling_origin_predictions(modified, 12, 14, "naive")
    assert changed.iloc[1] == original.iloc[1]


def test_full_workflow_compares_baselines_and_preserves_test_boundary(synthetic_sales):
    future, backtest, fitted_model, report = run_revenue_forecast(synthetic_sales)
    assert set(report["validation_metrics"]) == set(FORECAST_METHODS)
    assert set(report["test_metrics"]) == set(FORECAST_METHODS)
    assert report["selected_method"] == min(
        FORECAST_METHODS,
        key=lambda method: report["validation_metrics"][method]["wape"],
    )
    assert report["protocol"]["initial_training_period"] == (
        "2023-01-01/2023-12-01"
    )
    assert report["protocol"]["validation_period"] == "2024-01-01/2024-06-01"
    assert report["protocol"]["test_period"] == "2024-07-01/2024-12-01"
    assert len(backtest) == 36
    assert set(backtest.phase) == {"validation", "test"}
    assert len(future) == 3
    assert future.month.tolist() == list(
        pd.date_range("2025-01-01", periods=3, freq="MS")
    )
    assert future.forecast_net_revenue.gt(0).all()
    assert fitted_model is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_months": 0},
        {"test_months": 0},
        {"moving_average_window": 0},
        {"future_horizon_months": 0},
        {"minimum_initial_training_months": 2, "moving_average_window": 3},
    ],
)
def test_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RevenueForecastConfig(**kwargs)


def test_missing_month_is_rejected(synthetic_sales):
    incomplete = synthetic_sales[
        pd.to_datetime(synthetic_sales.sale_date).ne(pd.Timestamp("2023-08-01"))
    ]
    with pytest.raises(ValueError, match="complete chronological series"):
        build_monthly_net_revenue(incomplete)


def test_insufficient_history_and_unknown_method_are_rejected(synthetic_sales):
    series = build_monthly_net_revenue(synthetic_sales)
    with pytest.raises(ValueError, match="method must be"):
        rolling_origin_predictions(series, 12, 18, "oracle")
    short_sales = synthetic_sales[
        pd.to_datetime(synthetic_sales.sale_date).lt(pd.Timestamp("2024-12-01"))
    ]
    with pytest.raises(ValueError, match="At least 24 monthly observations"):
        run_revenue_forecast(short_sales)
