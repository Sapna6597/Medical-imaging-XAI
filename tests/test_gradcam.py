import numpy as np
import torch

from med_image_xai.gradcam import GradCAM
from med_image_xai.model import build_model


def test_gradcam_heatmap_shape_and_range():
    model = build_model(n_channels=1, n_classes=2)
    x = torch.randn(1, 1, 64, 64)
    with GradCAM(model, model.target_layer) as cam:
        heatmap = cam.generate(x)
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_gradcam_specific_class():
    model = build_model(n_channels=3, n_classes=7)
    x = torch.randn(1, 3, 64, 64)
    with GradCAM(model, model.target_layer) as cam:
        heatmap = cam.generate(x, class_idx=3)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (64, 64)
