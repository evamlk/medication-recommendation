"""Evaluation utilities for multi-label medication prediction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import average_precision_score


def _validate_targets_and_probabilities(
    targets: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if targets.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("targets and probabilities must be two-dimensional")
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must have identical shapes")
    if not np.isin(targets, (0, 1, False, True)).all():
        raise ValueError("targets must be binary")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")

    return targets.astype(bool, copy=False), probabilities


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute set-based and probability-based multi-label metrics."""
    targets, probabilities = _validate_targets_and_probabilities(targets, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")

    predictions = probabilities >= threshold
    intersection = np.logical_and(targets, predictions).sum(axis=1)
    union = np.logical_or(targets, predictions).sum(axis=1)
    sample_jaccard = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union > 0,
    )

    sample_denominator = targets.sum(axis=1) + predictions.sum(axis=1)
    sample_f1 = np.divide(
        2 * intersection,
        sample_denominator,
        out=np.ones_like(intersection, dtype=np.float64),
        where=sample_denominator > 0,
    )

    true_positive = np.logical_and(targets, predictions).sum()
    false_positive = np.logical_and(~targets, predictions).sum()
    false_negative = np.logical_and(targets, ~predictions).sum()
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    micro_f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)

    class_true_positive = np.logical_and(targets, predictions).sum(axis=0)
    class_denominator = targets.sum(axis=0) + predictions.sum(axis=0)
    class_f1 = np.divide(
        2 * class_true_positive,
        class_denominator,
        out=np.zeros_like(class_true_positive, dtype=np.float64),
        where=class_denominator > 0,
    )

    return {
        "jaccard": float(sample_jaccard.mean()),
        "sample_f1": float(sample_f1.mean()),
        "micro_precision": float(precision),
        "micro_recall": float(recall),
        "micro_f1": float(micro_f1),
        "macro_f1": float(class_f1.mean()),
        "micro_average_precision": float(average_precision_score(targets, probabilities)),
        "macro_average_precision": float(
            average_precision_score(targets, probabilities, average="macro")
        ),
        "avg_predicted_medications": float(predictions.sum(axis=1).mean()),
        "avg_true_medications": float(targets.sum(axis=1).mean()),
        "empty_prediction_rate": float((predictions.sum(axis=1) == 0).mean()),
    }


def tune_global_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Select the threshold with the highest validation Jaccard score."""
    targets, probabilities = _validate_targets_and_probabilities(targets, probabilities)
    candidates = np.asarray(
        list(thresholds) if thresholds is not None else np.arange(0.05, 0.96, 0.01),
        dtype=np.float64,
    )
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional sequence")
    if ((candidates < 0) | (candidates > 1)).any():
        raise ValueError("all thresholds must lie in [0, 1]")

    scored = [multilabel_metrics(targets, probabilities, float(value)) for value in candidates]
    best_index = max(range(len(scored)), key=lambda index: scored[index]["jaccard"])
    return float(candidates[best_index]), scored[best_index]
