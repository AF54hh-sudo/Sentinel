# Visualization Layer

## Purpose

Phase 10 converts AMX Tech's validated analytical outputs into reusable Plotly evidence views. The layer does not query PostgreSQL, recompute business definitions, fit models, or generate narrative conclusions. It receives explicit data contracts and returns `plotly.graph_objects.Figure` instances that can be rendered in notebooks, standalone HTML, or the Phase 11 Streamlit dashboard.

## Chart inventory

| Chart | Function | Evidence shown |
|---|---|---|
| Monthly revenue | `revenue_trend_chart` | Gross revenue, net revenue, and discount rate |
| Monthly churn | `churn_trend_chart` | Exposed-subscription churn and cancellation counts |
| Cloud-cost composition | `cloud_cost_chart` | Compute, storage, network, and GPU monthly costs |
| Region performance | `segment_performance_chart` | Net revenue and subscription churn on aligned scales |
| Plan performance | `segment_performance_chart` | Net revenue and subscription churn on aligned scales |
| GPU anomaly evidence | `anomaly_evidence_chart` | Daily GPU cost, robust normal band, score, and strong dates |
| Churn-model evaluation | `churn_model_evaluation_chart` | Candidate cross-validation AP and held-out confusion matrix |
| Revenue forecast | `forecast_evidence_chart` | Observed history, selected test forecasts, and future estimates |

All figures use stable trace names, explicit axis units, formatted hover evidence, responsive sizing, and AMX Tech titles. Input frames are copied and validated, so rendering cannot mutate analytical results.

## Design choices

- Revenue and churn use aligned vertical panels when percentage and absolute values have different scales.
- Cost components use one stacked chart because their part-to-whole relationship matters.
- Segment revenue and churn use side-by-side aligned panels instead of a dual-axis overlay.
- Strong anomaly diamonds are shown over the robust expected band; color is paired with labels and marker shape.
- The churn figure retains the dummy baseline and shows actual error counts, preventing a single headline metric from hiding failure modes.
- The forecast chart visually separates observed, one-step test, and genuinely future values with labeled line styles.

## Gallery generation

When PostgreSQL is available:

```powershell
python scripts/build_visualization_gallery.py --source database
```

For the validated generated CSVs:

```powershell
python scripts/build_visualization_gallery.py --source csv
```

The runner requires the completed Phase 7–9 artifacts and writes eight standalone charts, `manifest.json`, and a combined `index.html` under `models/visualizations`. Plotly JavaScript is loaded from its CDN, so opening exported HTML requires network access unless a future deployment bundles the library.

## Validation

Automated contracts verify figure types, titles, trace identities, required inputs, anomaly-marker counts, confusion-matrix values, forecast-series separation, input immutability, JSON serialization, and standalone/gallery HTML generation. The real seed-42 runner additionally confirms that every persisted analytical artifact can be rendered together.

## Limitations

- The gallery is an analytical review surface, not the Phase 11 interactive application.
- Exported HTML uses a fixed light presentation; Streamlit theme adaptation belongs to the application phase.
- Charts expose evidence already produced elsewhere and cannot repair limitations in the underlying models or data.
- Dense daily anomaly hover behavior is most usable on desktop-width displays.
- Static HTML does not provide live filters or database refresh.
