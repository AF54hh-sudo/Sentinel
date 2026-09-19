"""Stratified cross-validation, moderate tuning, and final churn model selection."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

from sentinel.ml.churn import build_candidate_pipelines

SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def cross_validate_candidates(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    random_state: int = 42,
    folds: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    pipelines = build_candidate_pipelines(random_state)
    reports: dict[str, Any] = {}
    for name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=splitter,
            scoring=SCORING,
            n_jobs=-1,
            return_train_score=False,
        )
        reports[name] = {
            metric: {
                "mean": float(scores[f"test_{metric}"].mean()),
                "std": float(scores[f"test_{metric}"].std(ddof=1)),
            }
            for metric in SCORING
        }
    return reports, pipelines


def tune_candidate(
    name: str,
    pipeline: Any,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    random_state: int = 42,
    folds: int = 5,
    iterations: int = 12,
) -> tuple[Any, dict[str, Any]]:
    spaces = {
        "logistic_regression": {
            "classifier__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        },
        "random_forest": {
            "classifier__n_estimators": [200, 350, 500],
            "classifier__max_depth": [None, 6, 10, 16],
            "classifier__min_samples_leaf": [2, 5, 10, 20],
            "classifier__max_features": ["sqrt", 0.5, 0.8],
        },
        "gradient_boosting": {
            "classifier__learning_rate": [0.03, 0.05, 0.08, 0.12],
            "classifier__max_iter": [120, 200, 300],
            "classifier__max_leaf_nodes": [7, 15, 31],
            "classifier__min_samples_leaf": [10, 20, 40],
            "classifier__l2_regularization": [0.0, 0.5, 1.0, 3.0],
        },
    }
    if name not in spaces:
        raise ValueError(f"No tuning space for {name!r}")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipeline,
        spaces[name],
        n_iter=min(iterations, _parameter_space_size(spaces[name])),
        scoring="average_precision",
        cv=splitter,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(features, target)
    return search.best_estimator_, {
        "best_parameters": search.best_params_,
        "best_cross_validation_average_precision": float(search.best_score_),
        "candidates_evaluated": len(search.cv_results_["params"]),
    }


def _parameter_space_size(space: dict[str, list[Any]]) -> int:
    size = 1
    for values in space.values():
        size *= len(values)
    return size
