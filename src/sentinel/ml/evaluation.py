"""Classification metrics, thresholding, error analysis, and interpretation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    return probabilities[:, classes.index(1)]


def classification_metrics(
    target: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    truth = np.asarray(target, dtype=int)
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "average_precision": float(average_precision_score(truth, probabilities)),
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
    }


def select_threshold_for_recall(
    target: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_recall: float = 0.70,
) -> float:
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be in (0, 1]")
    candidates = np.unique(np.r_[0.01, probabilities, 0.99])
    candidates = candidates[(candidates > 0) & (candidates < 1)]
    valid: list[tuple[float, float, float]] = []
    for threshold in candidates:
        metrics = classification_metrics(target, probabilities, threshold=float(threshold))
        if metrics["recall"] >= minimum_recall:
            valid.append((metrics["precision"], metrics["f1"], float(threshold)))
    return max(valid)[2] if valid else 0.5


def error_analysis(
    cohort: pd.DataFrame,
    target: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    analysis = cohort[["customer_id", "region", "primary_plan"]].copy()
    analysis["actual"] = np.asarray(target, dtype=int)
    analysis["probability"] = probabilities
    analysis["predicted"] = (analysis.probability >= threshold).astype(int)
    analysis["error_type"] = np.select(
        [
            analysis.actual.eq(1) & analysis.predicted.eq(0),
            analysis.actual.eq(0) & analysis.predicted.eq(1),
        ],
        ["false_negative", "false_positive"],
        default="correct",
    )
    errors = analysis[analysis.error_type.ne("correct")]
    return {
        "by_region": errors.groupby(["region", "error_type"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "by_plan": errors.groupby(["primary_plan", "error_type"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "highest_risk_false_positives": (
            errors[errors.error_type.eq("false_positive")]
            .nlargest(10, "probability")[["customer_id", "probability", "region", "primary_plan"]]
            .to_dict("records")
        ),
        "lowest_scored_false_negatives": (
            errors[errors.error_type.eq("false_negative")]
            .nsmallest(10, "probability")[["customer_id", "probability", "region", "primary_plan"]]
            .to_dict("records")
        ),
    }


def model_feature_importance(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    random_state: int = 42,
    repeats: int = 8,
) -> list[dict[str, float | str]]:
    result = permutation_importance(
        model,
        features,
        target,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    rows = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in zip(features.columns, result.importances_mean, result.importances_std)
    ]
    return sorted(rows, key=lambda row: row["importance_mean"], reverse=True)


def logistic_coefficients(model: Any) -> list[dict[str, float | str]]:
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "coef_"):
        return []
    names = model.named_steps["preprocessor"].get_feature_names_out()
    rows = [
        {"feature": str(name), "coefficient": float(value)}
        for name, value in zip(names, classifier.coef_[0])
    ]
    return sorted(rows, key=lambda row: abs(row["coefficient"]), reverse=True)
