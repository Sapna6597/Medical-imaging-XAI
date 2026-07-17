import numpy as np
import pandas as pd

from med_image_xai.fairness import disparity_summary, subgroup_metrics, synthetic_subgroups


def _binary_probs(preds):
    return np.array([[1 - p, p] for p in preds])


def test_subgroup_metrics_binary():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_prob = _binary_probs([0.1, 0.9, 0.8, 0.2, 0.6, 0.3])
    groups = pd.Series(["A", "A", "A", "B", "B", "B"])
    metrics = subgroup_metrics(y_true, y_prob, groups)
    assert set(metrics.index) == {"A", "B"}
    assert {"n", "accuracy", "roc_auc", "selection_rate"}.issubset(metrics.columns)


def test_disparity_summary_non_negative():
    y_true = np.array([0, 1, 1, 0])
    y_prob = _binary_probs([0.2, 0.8, 0.7, 0.1])
    groups = pd.Series(["A", "B", "A", "B"])
    summary = disparity_summary(subgroup_metrics(y_true, y_prob, groups))
    assert all(v >= 0 for v in summary.values())


def test_synthetic_subgroups_reproducible():
    a = synthetic_subgroups(50, seed=1)
    b = synthetic_subgroups(50, seed=1)
    assert (a == b).all()
    assert len(a) == 50
