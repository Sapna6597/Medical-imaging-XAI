"""Dataset loading via MedMNIST (open, standardized medical imaging benchmarks).

Datasets are downloaded on first use into the MedMNIST cache. All are openly
licensed (CC BY 4.0) and de-identified by their curators. See docs/data_ethics.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .config import DEFAULT_IMAGE_SIZE, resolve_flag


@dataclass
class DatasetMeta:
    """Metadata describing a loaded dataset."""

    flag: str
    task: str
    n_channels: int
    n_classes: int
    label_names: dict[str, str]


def _build_transform(n_channels: int) -> transforms.Compose:
    mean = [0.5] * n_channels
    std = [0.5] * n_channels
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def get_meta(name: str) -> DatasetMeta:
    """Return metadata for a dataset without downloading images."""
    from medmnist import INFO

    flag = resolve_flag(name)
    info = INFO[flag]
    return DatasetMeta(
        flag=flag,
        task=info["task"],
        n_channels=info["n_channels"],
        n_classes=len(info["label"]),
        label_names=info["label"],
    )


def _load_split(flag: str, split: str, size: int, download: bool, transform):
    import medmnist
    from medmnist import INFO

    data_class = getattr(medmnist, INFO[flag]["python_class"])
    return data_class(split=split, transform=transform, download=download, size=size)


def get_dataloaders(
    name: str,
    batch_size: int = 64,
    size: int = DEFAULT_IMAGE_SIZE,
    download: bool = True,
    limit: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, DatasetMeta]:
    """Return (train, val, test) DataLoaders and dataset metadata.

    ``limit`` caps the number of samples per split (useful for quick smoke tests).
    """
    meta = get_meta(name)
    transform = _build_transform(meta.n_channels)

    loaders = []
    for split, shuffle in (("train", True), ("val", False), ("test", False)):
        dataset = _load_split(meta.flag, split, size, download, transform)
        if limit is not None:
            dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
    train_loader, val_loader, test_loader = loaders
    return train_loader, val_loader, test_loader, meta


def get_test_arrays(
    name: str, size: int = DEFAULT_IMAGE_SIZE, download: bool = True, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw test images (N, H, W[, C]) and integer labels (N,)."""
    meta = get_meta(name)
    dataset = _load_split(meta.flag, "test", size, download, transform=None)
    images = dataset.imgs
    labels = dataset.labels.reshape(-1)
    if limit is not None:
        images, labels = images[:limit], labels[:limit]
    return images, labels


def to_tensor(image: np.ndarray, n_channels: int) -> torch.Tensor:
    """Convert a single raw uint8 image to a normalized (1, C, H, W) tensor."""
    from PIL import Image

    pil = Image.fromarray(image)
    tensor = _build_transform(n_channels)(pil)
    return tensor.unsqueeze(0)
