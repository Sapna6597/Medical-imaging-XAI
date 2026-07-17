"""Configuration: dataset registry, paths, and device selection."""

from __future__ import annotations

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42
DEFAULT_IMAGE_SIZE = 64  # MedMNIST supports 28 / 64 / 128 / 224

# Friendly name -> MedMNIST dataset flag.
DATASETS: dict[str, str] = {
    "pneumonia": "pneumoniamnist",  # chest X-ray: normal vs pneumonia (binary)
    "retina": "retinamnist",  # fundus: diabetic retinopathy grade (5 classes)
    "derma": "dermamnist",  # dermatoscope: skin lesion type (7 classes)
}


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_flag(name: str) -> str:
    """Map a friendly dataset name to its MedMNIST flag."""
    if name in DATASETS:
        return DATASETS[name]
    if name in DATASETS.values():
        return name
    raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
