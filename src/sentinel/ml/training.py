"""End-to-end, leakage-safe training workflow for AMX Tech customer churn."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from sentinel.features.customer_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    ChurnSnapshotConfig,
    build_churn_snapshot,
)
from sentinel.ml.evaluation import (
    classification_metrics,
    error_analysis,
    logistic_coefficients,
    model_feature_importance,
    positive_probabilities,
    select_threshold_for_recall,
)
from sentinel.ml.model_selection import cross_validate_candidates, tune_candidate


@dataclass(frozen=True)
class ChurnTrainingConfig:
    random_state: int = 42
    test_size: float = 0.20
    cross_validation_folds: int = 5
    tuning_iterations: int = 12
    minimum_recall: float = 0.70

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 0.5:
            raise ValueError("test_size must be between zero and 0.5")
        if self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds must be at least two")
        if self.tuning_iterations < 1:
            raise ValueError("tuning_iterations must be positive")
        if not 0 < self.minimum_recall <= 1:
            raise ValueError("minimum_recall must be in (0, 1]")


def train_churn_model(
    tables: dict[str, pd.DataFrame],
    *,
    snapshot_config: ChurnSnapshotConfig | None = None,
    training_config: ChurnTrainingConfig | None = None,
) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    snapshot_config = snapshot_config or ChurnSnapshotConfig()
    training_config = training_config or ChurnTrainingConfig()
    cohort = build_churn_snapshot(tables, snapshot_config)
    feature_columns = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
    features = cohort[feature_columns]
    target = cohort[TARGET_COLUMN]

    train_index, test_index = train_test_split(
        np.arange(len(cohort)),
        test_size=training_config.test_size,
        stratify=target,
        random_state=training_config.random_state,
    )
    train_features, test_features = features.iloc[train_index], features.iloc[test_index]
    train_target, test_target = target.iloc[train_index], target.iloc[test_index]

    cv_reports, pipelines = cross_validate_candidates(
        train_features,
        train_target,
        random_state=training_config.random_state,
        folds=training_config.cross_validation_folds,
    )
    baseline_ap = cv_reports["dummy"]["average_precision"]["mean"]
    selected_name = max(
        (name for name in pipelines if name != "dummy"),
        key=lambda name: cv_reports[name]["average_precision"]["mean"],
    )
    tuned_model, tuning_report = tune_candidate(
        selected_name,
        pipelines[selected_name],
        train_features,
        train_target,
        random_state=training_config.random_state,
        folds=training_config.cross_validation_folds,
        iterations=training_config.tuning_iterations,
    )

    splitter = StratifiedKFold(
        n_splits=training_config.cross_validation_folds,
        shuffle=True,
        random_state=training_config.random_state,
    )
    out_of_fold = cross_val_predict(
        clone(tuned_model),
        train_features,
        train_target,
        cv=splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold = select_threshold_for_recall(
        train_target,
        out_of_fold,
        minimum_recall=training_config.minimum_recall,
    )
    tuned_model.fit(train_features, train_target)
    test_probabilities = positive_probabilities(tuned_model, test_features)
    held_out = classification_metrics(test_target, test_probabilities, threshold=threshold)
    default_threshold = classification_metrics(test_target, test_probabilities, threshold=0.5)

    base_test_metrics: dict[str, Any] = {}
    for name, pipeline in pipelines.items():
        fitted = clone(pipeline).fit(train_features, train_target)
        base_test_metrics[name] = classification_metrics(
            test_target, positive_probabilities(fitted, test_features), threshold=0.5
        )

    test_cohort = cohort.iloc[test_index].reset_index(drop=True)
    ranked = test_cohort[["customer_id", "region", "primary_plan", TARGET_COLUMN]].copy()
    ranked["churn_probability"] = test_probabilities
    ranked = ranked.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    report = {
        "company": "AMX Tech",
        "snapshot": {
            "snapshot_date": str(snapshot_config.snapshot_date.date()),
            "prediction_end_date": str(snapshot_config.prediction_end_date.date()),
            "trailing_window_days": snapshot_config.trailing_window_days,
            "target_definition": (
                "Customer has an eligible subscription active at snapshot that is cancelled "
                "after snapshot and on or before prediction end."
            ),
        },
        "training_configuration": asdict(training_config),
        "cohort": {
            "customers": len(cohort),
            "churners": int(target.sum()),
            "churn_rate": float(target.mean()),
            "training_customers": len(train_index),
            "test_customers": len(test_index),
            "training_churn_rate": float(train_target.mean()),
            "test_churn_rate": float(test_target.mean()),
        },
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "leakage_exclusions": [
                "customer_status at dataset end",
                "subscription_status as of dataset end",
                "subscription end dates as predictive features",
                "sales or tickets after snapshot",
                "ground-truth scenario metadata",
            ],
        },
        "cross_validation": cv_reports,
        "selection_metric": "average_precision",
        "selection_protocol": (
            "Candidate selection, tuning, and threshold selection use training data only. "
            "The held-out test set is evaluated once after these choices are fixed."
        ),
        "baseline_cross_validation_average_precision": baseline_ap,
        "selected_model": selected_name,
        "tuning": tuning_report,
        "threshold_selection": {
            "source": "out-of-fold training predictions",
            "minimum_recall": training_config.minimum_recall,
            "selected_threshold": threshold,
        },
        "held_out_test_metrics": held_out,
        "held_out_default_threshold_metrics": default_threshold,
        "untuned_candidate_test_metrics": base_test_metrics,
        "average_precision_lift_over_held_out_dummy": (
            held_out["average_precision"] / base_test_metrics["dummy"]["average_precision"]
            if base_test_metrics["dummy"]["average_precision"]
            else None
        ),
        "error_analysis": error_analysis(
            test_cohort,
            test_target.reset_index(drop=True),
            test_probabilities,
            threshold,
        ),
        "permutation_importance": model_feature_importance(
            tuned_model,
            test_features,
            test_target,
            random_state=training_config.random_state,
        ),
        "logistic_coefficients": (
            logistic_coefficients(tuned_model)
            if selected_name == "logistic_regression"
            else []
        ),
        "limitations": [
            "Predictions are probabilities for a synthetic cohort, not certainties.",
            "A random customer split estimates within-period generalization, not temporal drift.",
            "The target concerns subscriptions active at the snapshot only.",
            "Feature importance describes predictive contribution, not causation.",
            "Probability calibration is not separately evaluated in Phase 7.",
        ],
    }
    deployment_model = clone(tuned_model).fit(features, target)
    return deployment_model, report, ranked
