import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from sentinel.data.generator import AMXTechDataGenerator, GenerationConfig
from sentinel.eda.cleaning import clean_dataset
from sentinel.features.customer_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    ChurnSnapshotConfig,
    build_churn_snapshot,
)
from sentinel.ml.churn import build_candidate_pipelines
from sentinel.ml.evaluation import (
    classification_metrics,
    positive_probabilities,
    select_threshold_for_recall,
)
from sentinel.ml.model_selection import cross_validate_candidates, tune_candidate
from sentinel.ml.training import ChurnTrainingConfig, train_churn_model


@pytest.fixture(scope="module")
def tables():
    raw = AMXTechDataGenerator(
        GenerationConfig(n_customers=800, n_subscriptions=1_200, n_tickets=1_600)
    ).generate()
    return clean_dataset(raw)[0]


@pytest.fixture(scope="module")
def cohort(tables):
    return build_churn_snapshot(tables)


def test_churn_snapshot_is_unique_deterministic_and_leakage_safe(tables, cohort):
    second = build_churn_snapshot(tables)
    pd.testing.assert_frame_equal(cohort, second)
    assert cohort.customer_id.is_unique
    assert all(type(column) is str for column in cohort.columns)
    assert cohort[TARGET_COLUMN].isin([0, 1]).all()
    assert cohort[TARGET_COLUMN].nunique() == 2
    assert set(CATEGORICAL_FEATURES + NUMERIC_FEATURES).issubset(cohort.columns)
    forbidden = {"customer_status", "subscription_status", "end_date", "churn_probability"}
    assert forbidden.isdisjoint(CATEGORICAL_FEATURES + NUMERIC_FEATURES)


def test_snapshot_target_only_uses_subscriptions_active_at_snapshot(tables):
    config = ChurnSnapshotConfig()
    cohort = build_churn_snapshot(tables, config).set_index("customer_id")
    subscriptions = tables["subscriptions"].copy()
    subscriptions["start_date"] = pd.to_datetime(subscriptions.start_date)
    subscriptions["end_date"] = pd.to_datetime(subscriptions.end_date)
    eligible = subscriptions[
        subscriptions.start_date.le(config.snapshot_date)
        & (subscriptions.end_date.isna() | subscriptions.end_date.gt(config.snapshot_date))
    ]
    expected = set(
        eligible.loc[
            eligible.subscription_status.eq("cancelled")
            & eligible.end_date.gt(config.snapshot_date)
            & eligible.end_date.le(config.prediction_end_date),
            "customer_id",
        ]
    )
    actual = set(cohort.index[cohort[TARGET_COLUMN].eq(1)])
    assert actual == expected


def test_all_candidate_pipelines_train_and_return_probabilities(cohort):
    features = cohort[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    target = cohort[TARGET_COLUMN]
    train_x, test_x, train_y, _ = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=42
    )
    for name, pipeline in build_candidate_pipelines().items():
        fitted = pipeline.fit(train_x, train_y)
        probabilities = positive_probabilities(fitted, test_x)
        assert len(probabilities) == len(test_x), name
        assert np.all((0 <= probabilities) & (probabilities <= 1)), name


def test_cross_validation_includes_required_baseline_and_metrics(cohort):
    features = cohort[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    target = cohort[TARGET_COLUMN]
    reports, pipelines = cross_validate_candidates(features, target, folds=3)
    assert set(reports) == {
        "dummy",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
    assert set(reports) == set(pipelines)
    for metrics in reports.values():
        assert set(metrics) == {"average_precision", "roc_auc", "recall", "precision", "f1"}
        for result in metrics.values():
            assert 0 <= result["mean"] <= 1
            assert result["std"] >= 0


def test_moderate_tuning_returns_fitted_estimator(cohort):
    features = cohort[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    target = cohort[TARGET_COLUMN]
    pipeline = build_candidate_pipelines()["logistic_regression"]
    model, report = tune_candidate(
        "logistic_regression", pipeline, features, target, folds=3, iterations=3
    )
    probabilities = positive_probabilities(model, features.head(10))
    assert len(probabilities) == 10
    assert report["candidates_evaluated"] == 3
    assert report["best_cross_validation_average_precision"] > 0


def test_metrics_and_threshold_contract():
    target = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.05, 0.2, 0.8, 0.6, 0.9])
    report = classification_metrics(target, probabilities, threshold=0.5)
    assert set(report) >= {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "confusion_matrix",
    }
    assert report["confusion_matrix"]["true_positive"] == 2
    threshold = select_threshold_for_recall(target, probabilities, minimum_recall=1.0)
    assert classification_metrics(target, probabilities, threshold=threshold)["recall"] == 1
    boundary_threshold = select_threshold_for_recall(
        target,
        np.array([0.0, 0.2, 0.8, 0.6, 1.0]),
        minimum_recall=1.0,
    )
    assert 0 < boundary_threshold < 1
    with pytest.raises(ValueError):
        classification_metrics(target, probabilities, threshold=1.0)


def test_snapshot_config_rejects_invalid_dates():
    with pytest.raises(ValueError):
        ChurnSnapshotConfig("2024-12-31", "2024-03-31")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_size": 0.5},
        {"cross_validation_folds": 1},
        {"tuning_iterations": 0},
        {"minimum_recall": 0.0},
    ],
)
def test_training_configuration_rejects_unsafe_values(kwargs):
    with pytest.raises(ValueError):
        ChurnTrainingConfig(**kwargs)


def test_end_to_end_training_returns_auditable_held_out_evidence(tables, cohort):
    model, report, ranked = train_churn_model(
        tables,
        training_config=ChurnTrainingConfig(
            test_size=0.25,
            cross_validation_folds=2,
            tuning_iterations=1,
            minimum_recall=0.5,
        ),
    )

    assert report["company"] == "AMX Tech"
    assert report["selected_model"] != "dummy"
    assert report["selection_metric"] == "average_precision"
    assert report["threshold_selection"]["source"] == "out-of-fold training predictions"
    assert report["cohort"]["training_customers"] + report["cohort"]["test_customers"] == len(
        cohort
    )
    assert 0 <= report["held_out_test_metrics"]["average_precision"] <= 1
    assert ranked.churn_probability.is_monotonic_decreasing

    features = cohort[CATEGORICAL_FEATURES + NUMERIC_FEATURES].head(5)
    probabilities = positive_probabilities(model, features)
    assert np.all((0 <= probabilities) & (probabilities <= 1))
