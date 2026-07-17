"""Load a trained checkpoint and run inference / Grad-CAM on a single image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import get_device
from .data import to_tensor
from .gradcam import GradCAM, save_overlay
from .model import build_model


def load_checkpoint(path: str | Path):
    """Load a saved model and its metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}. Train a model first.")
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["n_channels"], ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_image(model, ckpt, image: np.ndarray) -> dict:
    """Return predicted class, label name, and class probabilities for one image."""
    device = get_device()
    tensor = to_tensor(image, ckpt["n_channels"]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model.to(device)(tensor), dim=1).cpu().numpy()[0]
    class_idx = int(probs.argmax())
    return {
        "class_idx": class_idx,
        "label": ckpt["label_names"][str(class_idx)],
        "probabilities": probs.tolist(),
    }


def explain_image(model, ckpt, image: np.ndarray, out_path: str | Path) -> Path:
    """Generate a Grad-CAM overlay for one image and save it."""
    tensor = to_tensor(image, ckpt["n_channels"])
    with GradCAM(model, model.target_layer) as cam:
        heatmap = cam.generate(tensor)
    pred = predict_image(model, ckpt, image)
    return save_overlay(image, heatmap, out_path, title=f"Prediction: {pred['label']}")
