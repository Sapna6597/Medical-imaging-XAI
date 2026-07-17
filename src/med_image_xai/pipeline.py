"""End-to-end demo: train, evaluate, Grad-CAM, and fairness on one dataset.

Uses a small image size and sample cap so it runs quickly on CPU. Increase
``EPOCHS`` / remove ``LIMIT`` for real results.
"""

from __future__ import annotations

from .config import REPORTS_DIR
from .data import get_dataloaders, get_test_arrays
from .evaluate import collect_predictions, compute_metrics, plot_confusion, plot_roc
from .fairness import disparity_summary, subgroup_metrics, synthetic_subgroups
from .predict import explain_image, load_checkpoint
from .train import checkpoint_path, train

DATASET = "pneumonia"
EPOCHS = 3
SIZE = 64
LIMIT = 500  # samples per split; set to None for the full dataset


def main() -> None:
    print(f"=== Training on {DATASET} (size={SIZE}, limit={LIMIT}) ===")
    train(DATASET, epochs=EPOCHS, size=SIZE, limit=LIMIT)

    model, ckpt = load_checkpoint(checkpoint_path(DATASET))
    _, _, test_loader, meta = get_dataloaders(DATASET, size=SIZE, limit=LIMIT)
    y_true, y_prob = collect_predictions(model, test_loader)

    print("\n=== Test metrics ===")
    print(compute_metrics(y_true, y_prob, meta.n_classes))
    print(f"Saved: {plot_confusion(y_true, y_prob, meta.label_names)}")
    print(f"Saved: {plot_roc(y_true, y_prob, meta.n_classes)}")

    print("\n=== Grad-CAM explanation (test image 0) ===")
    images, _ = get_test_arrays(DATASET, size=SIZE, limit=1)
    out = explain_image(model, ckpt, images[0], REPORTS_DIR / f"gradcam_{DATASET}_0.png")
    print(f"Saved: {out}")

    print("\n=== Fairness audit (PLACEHOLDER subgroups) ===")
    groups = synthetic_subgroups(len(y_true))
    metrics = subgroup_metrics(y_true, y_prob, groups)
    print(metrics.round(3).to_string())
    print(f"Disparities: {disparity_summary(metrics)}")


if __name__ == "__main__":
    main()
