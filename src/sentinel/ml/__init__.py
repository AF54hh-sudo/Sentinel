"""Churn feature engineering, candidate models, selection, and evaluation."""

from sentinel.ml.churn import build_candidate_pipelines
from sentinel.ml.evaluation import classification_metrics
from sentinel.ml.model_selection import cross_validate_candidates
from sentinel.ml.training import ChurnTrainingConfig, train_churn_model

__all__ = [
    "ChurnTrainingConfig",
    "build_candidate_pipelines",
    "classification_metrics",
    "cross_validate_candidates",
    "train_churn_model",
]
