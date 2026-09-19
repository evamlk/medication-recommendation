import numpy as np

from medrec.metrics import multilabel_metrics, tune_global_threshold


def test_perfect_predictions_score_one() -> None:
    targets = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    probabilities = np.array([[0.9, 0.1, 0.8], [0.2, 0.9, 0.1]])

    metrics = multilabel_metrics(targets, probabilities, threshold=0.5)

    assert metrics["jaccard"] == 1.0
    assert metrics["sample_f1"] == 1.0
    assert metrics["micro_f1"] == 1.0


def test_threshold_selection_uses_jaccard() -> None:
    targets = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    probabilities = np.array([[0.7, 0.4], [0.3, 0.6]])

    threshold, metrics = tune_global_threshold(targets, probabilities, [0.25, 0.5, 0.75])

    assert threshold == 0.5
    assert metrics["jaccard"] == 1.0
