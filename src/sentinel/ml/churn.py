"""Candidate churn pipelines with a required dummy baseline."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentinel.ml.preprocessing import build_preprocessor


def build_candidate_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "dummy": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("classifier", DummyClassifier(strategy="prior", random_state=random_state)),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=200,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
