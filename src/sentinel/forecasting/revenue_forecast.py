"""Chronological monthly net-revenue forecasting for AMX Tech."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

FORECAST_METHODS = ("naive", "moving_average", "exponential_smoothing")


@dataclass(frozen=True)
class RevenueForecastConfig:
    """Configuration for walk-forward selection, testing, and future forecasting."""

    validation_months: int = 6
    test_months: int = 6
    moving_average_window: int = 3
    future_horizon_months: int = 3
    minimum_initial_training_months: int = 12
    damped_trend: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "validation_months",
            "test_months",
            "moving_average_window",
            "future_horizon_months",
            "minimum_initial_training_months",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be positive")
        if self.minimum_initial_training_months < self.moving_average_window:
            raise ValueError(
                "minimum_initial_training_months must cover the moving-average window"
            )


def build_monthly_net_revenue(sales: pd.DataFrame) -> pd.Series:
    """Aggregate sales into one positive, complete month-start revenue series."""
    frame = sales.copy()
    frame.columns = ["".join(column) for column in frame.columns]
    missing = {"sale_date", "net_revenue"} - set(frame.columns)
    if missing:
        raise ValueError(f"Sales data is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Sales data cannot be empty")
    frame["sale_date"] = pd.to_datetime(frame.sale_date, errors="raise")
    frame["net_revenue"] = pd.to_numeric(frame.net_revenue, errors="raise")
    frame["month"] = frame.sale_date.dt.to_period("M").dt.to_timestamp()
    monthly = frame.groupby("month").net_revenue.sum().sort_index()
    expected = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    if not monthly.index.equals(expected):
        raise ValueError("Monthly revenue must form one complete chronological series")
    if monthly.le(0).any():
        raise ValueError("Monthly net revenue must be positive")
    monthly = monthly.asfreq("MS")
    monthly.name = "net_revenue"
    return monthly.astype(float)


def forecast_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute scale-aware errors without percentage division by individual months."""
    truth = np.asarray(actual, dtype=float)
    forecast = np.asarray(predicted, dtype=float)
    if truth.shape != forecast.shape or truth.size == 0:
        raise ValueError("actual and predicted must have the same non-zero shape")
    if not np.isfinite(truth).all() or not np.isfinite(forecast).all():
        raise ValueError("Forecast metrics require finite values")
    denominator = float(np.abs(truth).sum())
    if denominator == 0:
        raise ValueError("WAPE is undefined when total absolute actual value is zero")
    errors = truth - forecast
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "wape": float(np.abs(errors).sum() / denominator),
    }


def _fit_exponential_smoothing(history: pd.Series, *, damped_trend: bool) -> Any:
    model = ExponentialSmoothing(
        history,
        trend="add",
        damped_trend=damped_trend,
        seasonal=None,
        initialization_method="estimated",
    )
    return model.fit(optimized=True, remove_bias=True)


def _one_step_forecast(
    history: pd.Series,
    method: str,
    config: RevenueForecastConfig,
) -> float:
    if method == "naive":
        return float(history.iloc[-1])
    if method == "moving_average":
        return float(history.iloc[-config.moving_average_window :].mean())
    if method == "exponential_smoothing":
        fitted = _fit_exponential_smoothing(
            history, damped_trend=config.damped_trend
        )
        return float(fitted.forecast(1).iloc[0])
    raise ValueError(f"Unknown forecast method: {method}")


def rolling_origin_predictions(
    series: pd.Series,
    start_index: int,
    end_index: int,
    method: str,
    config: RevenueForecastConfig | None = None,
) -> pd.Series:
    """Produce one-step forecasts using only observations before each target month."""
    config = config or RevenueForecastConfig()
    if method not in FORECAST_METHODS:
        raise ValueError(f"method must be one of {FORECAST_METHODS}")
    if not config.minimum_initial_training_months <= start_index < end_index <= len(series):
        raise ValueError("Invalid rolling-origin boundaries")
    predictions = [
        _one_step_forecast(series.iloc[:index], method, config)
        for index in range(start_index, end_index)
    ]
    return pd.Series(
        predictions,
        index=series.index[start_index:end_index],
        name="forecast_net_revenue",
        dtype=float,
    )


def _backtest_phase(
    series: pd.Series,
    start_index: int,
    end_index: int,
    phase: str,
    config: RevenueForecastConfig,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    actual = series.iloc[start_index:end_index]
    metrics: dict[str, dict[str, float]] = {}
    rows: list[pd.DataFrame] = []
    for method in FORECAST_METHODS:
        predictions = rolling_origin_predictions(
            series, start_index, end_index, method, config
        )
        metrics[method] = forecast_metrics(actual, predictions)
        rows.append(
            pd.DataFrame(
                {
                    "phase": phase,
                    "month": actual.index,
                    "method": method,
                    "actual_net_revenue": actual.to_numpy(),
                    "forecast_net_revenue": predictions.to_numpy(),
                    "absolute_error": np.abs(
                        actual.to_numpy() - predictions.to_numpy()
                    ),
                }
            )
        )
    return metrics, pd.concat(rows, ignore_index=True)


def _future_forecast(
    series: pd.Series,
    selected_method: str,
    config: RevenueForecastConfig,
) -> tuple[pd.DataFrame, Any]:
    future_index = pd.date_range(
        series.index[-1] + pd.offsets.MonthBegin(1),
        periods=config.future_horizon_months,
        freq="MS",
    )
    if selected_method == "naive":
        predictions = np.repeat(float(series.iloc[-1]), config.future_horizon_months)
        fitted_model: Any = {"last_observation": float(series.iloc[-1])}
    elif selected_method == "moving_average":
        history = series.to_list()
        future_values: list[float] = []
        for _ in range(config.future_horizon_months):
            prediction = float(np.mean(history[-config.moving_average_window :]))
            future_values.append(prediction)
            history.append(prediction)
        predictions = np.asarray(future_values)
        fitted_model = {
            "window": config.moving_average_window,
            "latest_history": series.iloc[-config.moving_average_window :].to_list(),
        }
    else:
        fitted_model = _fit_exponential_smoothing(
            series, damped_trend=config.damped_trend
        )
        predictions = fitted_model.forecast(config.future_horizon_months).to_numpy()
    return (
        pd.DataFrame(
            {
                "month": future_index,
                "method": selected_method,
                "horizon_month": np.arange(1, config.future_horizon_months + 1),
                "forecast_net_revenue": predictions,
            }
        ),
        fitted_model,
    )


def run_revenue_forecast(
    sales: pd.DataFrame,
    config: RevenueForecastConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, dict[str, Any]]:
    """Select a method chronologically, evaluate it, and refit for future forecasts."""
    config = config or RevenueForecastConfig()
    series = build_monthly_net_revenue(sales)
    validation_start = len(series) - config.test_months - config.validation_months
    test_start = len(series) - config.test_months
    if validation_start < config.minimum_initial_training_months:
        required = (
            config.minimum_initial_training_months
            + config.validation_months
            + config.test_months
        )
        raise ValueError(f"At least {required} monthly observations are required")

    validation_metrics, validation_rows = _backtest_phase(
        series, validation_start, test_start, "validation", config
    )
    selected_method = min(
        FORECAST_METHODS,
        key=lambda method: (
            validation_metrics[method]["wape"],
            validation_metrics[method]["mae"],
        ),
    )
    test_metrics, test_rows = _backtest_phase(
        series, test_start, len(series), "test", config
    )
    backtest = pd.concat([validation_rows, test_rows], ignore_index=True)
    future, fitted_model = _future_forecast(series, selected_method, config)
    selected_test = test_metrics[selected_method]
    naive_test = test_metrics["naive"]
    lag_12_value = series.autocorr(lag=12) if len(series) > 12 else np.nan
    lag_12_correlation = float(lag_12_value) if np.isfinite(lag_12_value) else None
    wape_improvement = (
        float((naive_test["wape"] - selected_test["wape"]) / naive_test["wape"])
        if naive_test["wape"] > 0
        else None
    )

    report: dict[str, Any] = {
        "company": "AMX Tech",
        "configuration": asdict(config),
        "series": {
            "metric": "monthly net revenue",
            "start_month": str(series.index.min().date()),
            "end_month": str(series.index.max().date()),
            "observations": len(series),
            "overall_growth": float(series.iloc[-1] / series.iloc[0] - 1),
            "lag_12_correlation": lag_12_correlation,
        },
        "protocol": {
            "initial_training_period": (
                f"{series.index[0].date()}/{series.index[validation_start - 1].date()}"
            ),
            "validation_period": (
                f"{series.index[validation_start].date()}/{series.index[test_start - 1].date()}"
            ),
            "test_period": (
                f"{series.index[test_start].date()}/{series.index[-1].date()}"
            ),
            "evaluation": (
                "Expanding-window, one-step-ahead forecasts. Each prediction uses only "
                "months observed before its target month."
            ),
            "selection": (
                "Lowest validation WAPE, with validation MAE as a tie-breaker. The method "
                "is fixed before the test period is scored."
            ),
        },
        "candidate_definitions": {
            "naive": "Previous observed month.",
            "moving_average": (
                f"Mean of the previous {config.moving_average_window} observed months."
            ),
            "exponential_smoothing": (
                "Additive damped trend with estimated initialization and no seasonal term."
            ),
        },
        "validation_metrics": validation_metrics,
        "selected_method": selected_method,
        "test_metrics": test_metrics,
        "selected_test_metrics": selected_test,
        "test_wape_improvement_over_naive": wape_improvement,
        "future_forecast": [
            {
                "month": str(row.month.date()),
                "horizon_month": int(row.horizon_month),
                "forecast_net_revenue": float(row.forecast_net_revenue),
            }
            for row in future.itertuples(index=False)
        ],
        "limitations": [
            "Only 24 monthly observations are available, limiting structural complexity.",
            "Annual seasonality is plausible but two cycles are insufficient for a stable seasonal model.",
            "Walk-forward results estimate next-month updates, not a six-month forecast made once.",
            "Future values are point forecasts without calibrated prediction intervals.",
            "Synthetic history cannot establish real-world forecast performance.",
            "Known future pricing, pipeline, macroeconomic, and operational drivers are absent.",
        ],
    }
    return future, backtest, fitted_model, report
