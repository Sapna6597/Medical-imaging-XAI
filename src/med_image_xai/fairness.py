"""Subgroup fairness / bias analysis for image classifiers.

Medical imaging models can underperform for specific subpopulations or
acquisition conditions. This module reports performance per subgroup so
disparities are visible.

NOTE ON DEMOGRAPHICS: MedMNIST does not ship demographic labels. For a runnable
demonstration, ``synthetic_subgroups`` creates a reproducible PLACEHOLDER
attribute -- it is NOT real demographic data and must not be interpreted
clinically. To perform a genuine demographic audit, supply real metadata (e.g.,
Patient Age / Patient Gender from NIH ChestX-ray14) as the ``groups`` argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def synthetic_subgroups(n: int, seed: int = 42) -> pd.Series:
    """Return a reproducible PLACEHOLDER subgroup attribute (not real demographics)."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice(["group_A", "group_B"], size=n), name="subgroup")


def subgroup_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, groups: pd.Series, positive_class: int = 1
) -> pd.DataFrame:
    """Per-subgroup accuracy, ROC-AUC, and selection rate for the positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_prob).argmax(axis=1)
    y_prob = np.asarray(y_prob)
    groups = pd.Series(groups).reset_index(drop=True)

    rows = []
    for name, idx in groups.groupby(groups).groups.items():
        mask = groups.index.isin(idx)
        yt, yp = y_true[mask], y_pred[mask]
        try:
            if y_prob.shape[1] == 2:
                auc = roc_auc_score(yt, y_prob[mask, 1])
            else:
                auc = roc_auc_score(
                    yt, y_prob[mask], multi_class="ovr", average="macro"
                )
        except ValueError:
            auc = float("nan")
        rows.append(
            {
                "subgroup": name,
                "n": int(mask.sum()),
                "accuracy": float((yt == yp).mean()),
                "roc_auc": float(auc),
                "selection_rate": float((yp == positive_class).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("subgroup")


def disparity_summary(metrics: pd.DataFrame) -> dict[str, float]:
    """Max-minus-min gaps across subgroups."""
    summary = {}
    for col in ["accuracy", "roc_auc", "selection_rate"]:
        values = metrics[col].dropna()
        if not values.empty:
            summary[f"{col}_gap"] = float(values.max() - values.min())
    return summary
