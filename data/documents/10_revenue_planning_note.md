---
document_id: DOC-010
title: Revenue Planning and Forecast Use Note
document_date: 2024-12-20
category: planning_note
regions: [India, North America, Europe, APAC]
topics: [forecast, revenue, seasonality, uncertainty]
---
# Revenue Planning and Forecast Use Note

## Monthly forecast assumptions and decision use

Finance uses the monthly net-revenue forecast as a planning estimate for staffing and operating reviews. The current history is short, with only two annual cycles, so the team prefers transparent baselines and a restrained trend model over a complex seasonal specification. Every candidate must be evaluated chronologically. A random split would allow later observations to influence an earlier forecast and would not match the real decision process.

The required comparison includes the previous-month naive method, a short moving average, and additive damped-trend exponential smoothing. Validation performance selects the method before the later test period is scored. The chosen model is then refit on all observed months for the future forecast. WAPE, MAE, and RMSE describe historical point-error behavior; they do not guarantee the next outcome or create a calibrated prediction interval.

The forecast is company-wide because the persisted model was trained on total monthly net revenue. It should not inherit dashboard region or plan filters, and it should not be presented as a customer-level expectation. Pricing changes, discount mix, new sales, cancellations, and unusual events may shift realized revenue beyond what a short univariate history captures. The current output is therefore suitable for directional planning, not contractual commitments or cash guarantees.

When explaining a forecast, Sentinel should show the selected method, next-month point estimate, held-out error metrics, improvement or deterioration relative to the naive baseline, and limitations. This note provides governance context for how Finance intends to use the result. The authoritative estimate and evaluation values come from the persisted forecasting artifact, never from this document or a language-model calculation.
