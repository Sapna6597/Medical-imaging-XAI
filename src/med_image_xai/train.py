"""Training loop with checkpointing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .config import MODEL_DIR, RANDOM_SEED, get_device, resolve_flag
from .data import get_dataloaders
from .model import build_model


def _run_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.reshape(-1).long().to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def _val_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.reshape(-1).long().to(device)
        preds = model(images).argmax(dim=1)
        correct += int((preds == labels).sum())
        total += labels.size(0)
    return correct / max(total, 1)


def train(
    dataset: str,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    size: int = 64,
    limit: int | None = None,
    download: bool = True,
) -> tuple[nn.Module, dict]:
    """Train the classifier on a dataset and save the best checkpoint."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = get_device()

    train_loader, val_loader, _, meta = get_dataloaders(
        dataset, batch_size=batch_size, size=size, download=download, limit=limit
    )
    model = build_model(meta.n_channels, meta.n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_acc": []}
    best_acc = -1.0
    ckpt_path = checkpoint_path(dataset)

    for epoch in range(1, epochs + 1):
        loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        acc = _val_accuracy(model, val_loader, device)
        history["train_loss"].append(loss)
        history["val_acc"].append(acc)
        print(f"epoch {epoch}/{epochs}  train_loss={loss:.4f}  val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            save_checkpoint(model, meta, size, ckpt_path)

    print(f"Best val accuracy: {best_acc:.4f}. Saved checkpoint to {ckpt_path}")
    return model, history


def checkpoint_path(dataset: str) -> Path:
    return MODEL_DIR / f"{resolve_flag(dataset)}_smallcnn.pt"


def save_checkpoint(model, meta, size: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "flag": meta.flag,
            "task": meta.task,
            "n_channels": meta.n_channels,
            "n_classes": meta.n_classes,
            "label_names": meta.label_names,
            "size": size,
        },
        path,
    )
