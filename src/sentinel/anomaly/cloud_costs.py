"""Robust daily cloud-cost anomaly detection for AMX Tech."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

COST_COLUMNS = [
    "compute_cost",
    "storage_cost",
    "network_cost",
    "gpu_cost",
    "total_cost",
]
SUPPORTED_METRICS = tuple(COST_COLUMNS)


@dataclass(frozen=True)
class CloudCostAnomalyConfig:
    """Configuration for retrospective daily anomaly detection."""

    metric: str = "gpu_cost"
    local_window_days: int = 29
    iqr_multiplier: float = 1.5
    strong_iqr_multiplier: float = 3.0
    isolation_contamination: float = 0.01
    isolation_estimators: int = 300
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.metric not in SUPPORTED_METRICS:
            raise ValueError(f"metric must be one of {SUPPORTED_METRICS}")
        if self.local_window_days < 7 or self.local_window_days % 2 == 0:
            raise ValueError("local_window_days must be an odd integer of at least seven")
        if self.iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier must be positive")
        if self.strong_iqr_multiplier <= self.iqr_multiplier:
            raise ValueError("strong_iqr_multiplier must exceed iqr_multiplier")
        if not 0 < self.isolation_contamination <= 0.5:
            raise ValueError("isolation_contamination must be in (0, 0.5]")
        if self.isolation_estimators < 50:
            raise ValueError("isolation_estimators must be at least 50")


def aggregate_daily_cloud_costs(cloud_costs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regional rows to one reconciled company-wide row per day."""
    frame = cloud_costs.copy()
    if frame.empty:
        raise ValueError("Cloud-cost data cannot be empty")
    frame.columns = ["".join(column) for column in frame.columns]
    required = {"date", *COST_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cloud-cost data is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in COST_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[COST_COLUMNS].lt(0).any().any():
        raise ValueError("Cloud costs cannot be negative")

    daily = (
        frame.groupby("date", as_index=False)[COST_COLUMNS]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    expected_dates = pd.date_range(daily.date.min(), daily.date.max(), freq="D")
    if len(daily) != len(expected_dates) or not daily.date.equals(pd.Series(expected_dates)):
        raise ValueError("Cloud-cost dates must form one complete daily time series")
    components = daily[["compute_cost", "storage_cost", "network_cost", "gpu_cost"]].sum(
        axis=1
    )
    if not np.allclose(components, daily.total_cost, atol=0.10, rtol=0):
        raise ValueError("Daily cloud-cost components do not reconcile to total_cost")
    return daily


def _iqr_features(
    daily: pd.DataFrame, config: CloudCostAnomalyConfig
) -> tuple[pd.DataFrame, dict[str, float]]:
    observed = daily[config.metric]
    if observed.le(0).any():
        raise ValueError(f"{config.metric} must be positive for ratio-based detection")
    minimum_periods = config.local_window_days // 2 + 1
    baseline = observed.rolling(
        config.local_window_days,
        center=True,
        min_periods=minimum_periods,
    ).median()
    if baseline.isna().any():
        raise ValueError(
            f"At least {minimum_periods} daily observations are required for the local baseline"
        )
    relative = observed / baseline
    q1, q3 = relative.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 0:
        raise ValueError("The local cost ratio has no usable interquartile range")

    bounds = {
        "ratio_q1": float(q1),
        "ratio_q3": float(q3),
        "ratio_iqr": iqr,
        "ratio_lower": max(0.0, float(q1 - config.iqr_multiplier * iqr)),
        "ratio_upper": float(q3 + config.iqr_multiplier * iqr),
        "strong_ratio_lower": max(
            0.0, float(q1 - config.strong_iqr_multiplier * iqr)
        ),
        "strong_ratio_upper": float(q3 + config.strong_iqr_multiplier * iqr),
    }
    result = daily.copy()
    result["local_baseline"] = baseline
    result["relative_to_baseline"] = relative
    result["normal_range_lower"] = baseline * bounds["ratio_lower"]
    result["normal_range_upper"] = baseline * bounds["ratio_upper"]
    result["iqr_anomaly"] = relative.lt(bounds["ratio_lower"]) | relative.gt(
        bounds["ratio_upper"]
    )
    result["strong_iqr_anomaly"] = relative.lt(
        bounds["strong_ratio_lower"]
    ) | relative.gt(bounds["strong_ratio_upper"])
    return result, bounds


def _interpretation(row: pd.Series, metric: str) -> str:
    difference = (float(row.relative_to_baseline) - 1) * 100
    direction = "above" if difference >= 0 else "below"
    if row.status == "strong_anomaly":
        return (
            f"Strong {metric} anomaly: {abs(difference):.1f}% {direction} the robust "
            "local median; both the outer IQR fence and Isolation Forest agree."
        )
    if row.status == "candidate_anomaly":
        return (
            f"Candidate {metric} anomaly: {abs(difference):.1f}% {direction} the robust "
            "local median; only one method or the standard IQR fence flags it."
        )
    return f"Within the expected robust local range for {metric}."


def detect_cloud_cost_anomalies(
    cloud_costs: pd.DataFrame,
    config: CloudCostAnomalyConfig | None = None,
) -> tuple[pd.DataFrame, IsolationForest, dict[str, Any]]:
    """Fit the two detectors and return daily scores, model, and an audit report."""
    config = config or CloudCostAnomalyConfig()
    daily = aggregate_daily_cloud_costs(cloud_costs)
    scored, bounds = _iqr_features(daily, config)
    model_input = np.log(scored[["relative_to_baseline"]].to_numpy())
    model = IsolationForest(
        n_estimators=config.isolation_estimators,
        contamination=config.isolation_contamination,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model_prediction = model.fit_predict(model_input)
    scored["anomaly_score"] = -model.decision_function(model_input)
    scored["isolation_forest_anomaly"] = model_prediction == -1
    scored["strong_anomaly"] = (
        scored.strong_iqr_anomaly & scored.isolation_forest_anomaly
    )
    scored["candidate_anomaly"] = scored.iqr_anomaly | scored.isolation_forest_anomaly
    scored["status"] = np.select(
        [scored.strong_anomaly, scored.candidate_anomaly],
        ["strong_anomaly", "candidate_anomaly"],
        default="normal",
    )
    scored["metric"] = config.metric
    scored["observed_value"] = scored[config.metric]
    scored["interpretation"] = scored.apply(
        _interpretation, axis=1, metric=config.metric
    )

    output_columns = [
        "date",
        "metric",
        "normal_range_lower",
        "normal_range_upper",
        "observed_value",
        "local_baseline",
        "relative_to_baseline",
        "iqr_anomaly",
        "strong_iqr_anomaly",
        "isolation_forest_anomaly",
        "candidate_anomaly",
        "strong_anomaly",
        "anomaly_score",
        "status",
        "interpretation",
    ]
    detections = scored[output_columns].copy()
    strong_rows = detections[detections.strong_anomaly]
    report: dict[str, Any] = {
        "company": "AMX Tech",
        "configuration": asdict(config),
        "data": {
            "start_date": str(detections.date.min().date()),
            "end_date": str(detections.date.max().date()),
            "daily_observations": len(detections),
            "source_rows": len(cloud_costs),
            "aggregation": "sum across regions to one company-wide daily observation",
        },
        "methodology": {
            "baseline": (
                f"Centered {config.local_window_days}-day rolling median for retrospective "
                "local-level adjustment."
            ),
            "iqr": (
                f"Standard {config.iqr_multiplier:.1f}-IQR candidates and "
                f"{config.strong_iqr_multiplier:.1f}-IQR strong outliers on the ratio of "
                "observed cost to its robust local baseline."
            ),
            "isolation_forest": (
                "Isolation Forest on the log local-cost ratio with a "
                f"{config.isolation_contamination:.1%} contamination prior."
            ),
            "final_rule": (
                "Strong anomaly requires agreement between the outer IQR fence and "
                "Isolation Forest."
            ),
        },
        "iqr_ratio_bounds": bounds,
        "counts": {
            "iqr_candidates": int(detections.iqr_anomaly.sum()),
            "strong_iqr_outliers": int(detections.strong_iqr_anomaly.sum()),
            "isolation_forest_anomalies": int(
                detections.isolation_forest_anomaly.sum()
            ),
            "final_strong_anomalies": int(detections.strong_anomaly.sum()),
        },
        "strong_anomalies": [
            {
                "date": str(row.date.date()),
                "metric": row.metric,
                "normal_range_lower": float(row.normal_range_lower),
                "normal_range_upper": float(row.normal_range_upper),
                "observed_value": float(row.observed_value),
                "anomaly_score": float(row.anomaly_score),
                "interpretation": row.interpretation,
            }
            for row in strong_rows.itertuples(index=False)
        ],
        "limitations": [
            "The centered baseline is retrospective and uses observations on both sides of a date.",
            "The contamination rate is an analytical prior, not a learned business frequency.",
            "Daily aggregation can hide offsetting or region-specific anomalies.",
            "Detector agreement identifies unusual values, not their operational cause.",
            "Results come from synthetic data and do not establish production performance.",
        ],
    }
    return detections, model, report


def evaluate_detection_period(
    detections: pd.DataFrame,
    expected_start: str | pd.Timestamp,
    expected_end: str | pd.Timestamp,
    *,
    flag_column: str = "strong_anomaly",
) -> dict[str, Any]:
    """Evaluate a detection flag against an externally supplied date interval."""
    if flag_column not in detections:
        raise ValueError(f"Unknown detection flag: {flag_column}")
    dates = pd.to_datetime(detections.date)
    start, end = pd.Timestamp(expected_start), pd.Timestamp(expected_end)
    if start > end:
        raise ValueError("expected_start must be on or before expected_end")
    expected = dates.between(start, end)
    actual = detections[flag_column].astype(bool).to_numpy()
    truth = expected.to_numpy()
    true_positive = int((actual & truth).sum())
    false_positive = int((actual & ~truth).sum())
    false_negative = int((~actual & truth).sum())
    true_negative = int((~actual & ~truth).sum())
    precision = true_positive / (true_positive + false_positive) if actual.sum() else 0.0
    recall = true_positive / truth.sum() if truth.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "flag": flag_column,
        "expected_period": f"{start.date()}/{end.date()}",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "detected_dates": dates[actual].dt.strftime("%Y-%m-%d").tolist(),
        "missed_expected_dates": dates[truth & ~actual].dt.strftime("%Y-%m-%d").tolist(),
    }
