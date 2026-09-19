# Monthly Revenue Forecast Findings

## Decision objective

Phase 9 estimates AMX Tech's next monthly net revenue and tests whether a forecasting model improves on simple planning baselines. The source series contains 24 monthly observations from January 2023 through December 2024. Revenue rises 14.71% from the first to the final observation, and the 12-month lag correlation is 0.988.

## Chronological evaluation

The time series is never randomly split:

- Initial training: January-December 2023.
- Model-selection validation: January-June 2024.
- Untouched test period: July-December 2024.

Every evaluation prediction is one month ahead with an expanding window. For example, the August test forecast may use actual revenue through July, but never August or any later value. Selection uses validation WAPE only, with validation MAE as a tie-breaker.

## Candidate methods

- Naïve baseline: repeat the previous observed month.
- Moving-average baseline: average the previous three observed months.
- Exponential smoothing: additive damped trend, estimated initialization, and no seasonal component.

Annual structure is visible, but two annual cycles are not enough to estimate a stable seasonal Holt-Winters model. SARIMA and neural models would add unjustified complexity at this sample size.

## Validation results

| Method | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naïve | 259,973 | 320,786 | 1.032% |
| Three-month moving average | 575,203 | 645,322 | 2.284% |
| Exponential smoothing | **158,527** | **217,062** | **0.630%** |

Exponential smoothing has the lowest validation WAPE and is selected before the test period is examined.

## Test results

| Method | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Naïve | 174,461 | 208,741 | 0.687% |
| Three-month moving average | 233,956 | 290,482 | 0.921% |
| Exponential smoothing | **148,020** | **163,288** | **0.583%** |

The selected method improves test WAPE by 15.16% relative to the required naïve baseline. Its average one-month error is about 148 thousand against monthly revenue near 25 million. This is a useful improvement on the synthetic history, but the small absolute test set contains only six forecast origins.

## Forward point forecast

After testing, exponential smoothing is refit on all 24 observations.

| Forecast month | Net-revenue forecast |
|---|---:|
| January 2025 | 26,233,753 |
| February 2025 | 26,524,830 |
| March 2025 | 26,776,086 |

The direct answer to “What is expected revenue next month?” is approximately **26.23 million for January 2025**. This is a point estimate, not a guaranteed result or a calibrated interval.

## Reproducible outputs

Run `python scripts/run_revenue_forecast.py --source database` when PostgreSQL is available, or `--source csv` for the validated generated dataset. It writes:

- `models/revenue_forecast_model.joblib`: fitted selected model, configuration, and series contract.
- `models/revenue_forecast_report.json`: partitions, candidate definitions, metrics, selection, forecasts, and limitations.
- `models/revenue_forecast.csv`: the three future monthly point forecasts.
- `models/revenue_forecast_backtest.csv`: validation and test predictions for every candidate.

These outputs are reproducible and intentionally git-ignored.

## Limitations

- Twenty-four observations and six test origins produce uncertain performance estimates.
- The evaluation measures rolling next-month forecasts, not a six-month forecast issued once.
- Only level and damped trend are modeled; a seasonal model needs more annual cycles.
- No calibrated prediction interval is available.
- Pricing plans, sales pipeline, macroeconomic conditions, and operational events are absent.
- Synthetic performance should not be generalized to production revenue.
