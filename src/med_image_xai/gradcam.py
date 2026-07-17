"""Grad-CAM: visual explanations for CNN predictions.

Implements Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017):
the gradient of the target class flowing into the last conv layer is used to
weight the layer's activation maps, highlighting image regions that drive the
prediction. Essential for clinical trust in imaging models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class GradCAM:
    """Compute Grad-CAM heatmaps for a target convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model.eval()
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def remove(self) -> None:
        """Detach hooks. Call when finished to avoid leaks."""
        for handle in self._handles:
            handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return a normalized (H, W) heatmap in [0, 1] for a single image tensor."""
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global-average-pool gradients to get per-channel weights.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def save_overlay(
    image: np.ndarray, heatmap: np.ndarray, out_path: str | Path, title: str = ""
) -> Path:
    """Save the original image, heatmap, and overlay side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gray = image if image.ndim == 2 else image.mean(axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(gray, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
