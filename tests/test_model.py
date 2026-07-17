import torch

from med_image_xai.model import build_model


def test_forward_shape_binary():
    model = build_model(n_channels=1, n_classes=2)
    out = model(torch.randn(4, 1, 64, 64))
    assert out.shape == (4, 2)


def test_forward_shape_multiclass_rgb():
    model = build_model(n_channels=3, n_classes=7)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 7)


def test_has_target_layer():
    model = build_model(n_channels=1, n_classes=2)
    assert isinstance(model.target_layer, torch.nn.Conv2d)
