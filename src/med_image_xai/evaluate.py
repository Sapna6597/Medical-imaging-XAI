"""Model evaluation: accuracy, ROC-AUC, confusion matrix, and ROC curves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from .config import REPORTS_DIR, get_device


@torch.no_grad()
def collect_predictions(model, loader, device=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) where y_prob is the softmax probability matrix."""
    device = device or get_device()
    model = model.to(device).eval()
    probs, targets = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(labels.reshape(-1).numpy())
    return np.concatenate(targets), np.concatenate(probs)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> dict[str, float]:
    """Accuracy and ROC-AUC (binary or macro one-vs-rest)."""
    y_pred = y_prob.argmax(axis=1)
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    try:
        if n_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def plot_confusion(y_true, y_prob, label_names: dict, out_dir: Path | None = None) -> Path:
    """Save a confusion matrix heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    labels = [label_names[str(i)] for i in range(len(label_names))]

    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path = out_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_roc(y_true, y_prob, n_classes: int, out_dir: Path | None = None) -> Path:
    """Save ROC curve(s)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        ax.plot(fpr, tpr, label="positive class")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
            ax.plot(fpr, tpr, label=f"class {c}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "roc_curve.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
