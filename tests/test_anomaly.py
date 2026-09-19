import pandas as pd
import pytest

from sentinel.anomaly import (
    CloudCostAnomalyConfig,
    aggregate_daily_cloud_costs,
    detect_cloud_cost_anomalies,
    evaluate_detection_period,
)
from sentinel.data.generator import AMXTechDataGenerator, GenerationConfig


@pytest.fixture(scope="module")
def cloud_costs():
    config = GenerationConfig(n_customers=300, n_subscriptions=480, n_tickets=600)
    return AMXTechDataGenerator(config).generate()["cloud_costs"]


@pytest.fixture(scope="module")
def detection_result(cloud_costs):
    return detect_cloud_cost_anomalies(cloud_costs)


def test_daily_aggregation_is_complete_and_reconciled(cloud_costs):
    daily = aggregate_daily_cloud_costs(cloud_costs)
    assert len(daily) == 731
    assert daily.date.is_monotonic_increasing
    components = daily.compute_cost + daily.storage_cost + daily.network_cost + daily.gpu_cost
    assert (components - daily.total_cost).abs().max() <= 0.10


def test_detector_output_contract_and_method_counts(detection_result):
    detections, model, report = detection_result
    assert report["company"] == "AMX Tech"
    assert report["counts"]["iqr_candidates"] >= report["counts"][
        "strong_iqr_outliers"
    ]
    assert report["counts"]["isolation_forest_anomalies"] > 0
    assert set(detections.status) <= {"normal", "candidate_anomaly", "strong_anomaly"}
    assert detections.anomaly_score.notna().all()
    assert detections.normal_range_lower.lt(detections.normal_range_upper).all()
    assert model.n_features_in_ == 1


def test_detector_recovers_the_injected_may_window(detection_result):
    detections, _, _ = detection_result
    evaluation = evaluate_detection_period(
        detections, "2024-05-13", "2024-05-19"
    )
    assert evaluation["recall"] == 1.0
    assert evaluation["precision"] >= 0.80
    assert evaluation["confusion_matrix"]["true_positive"] == 7
    assert evaluation["missed_expected_dates"] == []


def test_ground_truth_is_not_required_for_detection(cloud_costs):
    detections, _, _ = detect_cloud_cost_anomalies(cloud_costs)
    expected_columns = {
        "date",
        "metric",
        "normal_range_lower",
        "normal_range_upper",
        "observed_value",
        "anomaly_score",
        "interpretation",
    }
    assert expected_columns.issubset(detections.columns)
    assert not any("ground_truth" in column for column in detections.columns)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metric": "memory_cost"}, "metric must be"),
        ({"local_window_days": 10}, "odd integer"),
        ({"isolation_contamination": 0}, "contamination"),
    ],
)
def test_configuration_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CloudCostAnomalyConfig(**kwargs)


def test_aggregation_rejects_missing_dates(cloud_costs):
    frame = cloud_costs[pd.to_datetime(cloud_costs.date).ne(pd.Timestamp("2024-01-15"))]
    with pytest.raises(ValueError, match="complete daily time series"):
        aggregate_daily_cloud_costs(frame)


def test_evaluation_rejects_unknown_flag(detection_result):
    detections, _, _ = detection_result
    with pytest.raises(ValueError, match="Unknown detection flag"):
        evaluate_detection_period(
            detections,
            "2024-05-13",
            "2024-05-19",
            flag_column="oracle_anomaly",
        )
