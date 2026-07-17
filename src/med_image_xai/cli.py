"""Command-line interface for the medical imaging XAI toolkit.

Examples:
    med-image-xai download --dataset pneumonia
    med-image-xai train --dataset pneumonia --epochs 5
    med-image-xai evaluate --dataset pneumonia
    med-image-xai gradcam --dataset pneumonia --index 0
    med-image-xai fairness --dataset pneumonia
"""

from __future__ import annotations

import typer

from .config import DATASETS, REPORTS_DIR

app = typer.Typer(add_completion=False, help="Explainable medical image classifier.")

_DATASET_HELP = f"Dataset: one of {list(DATASETS)}"


@app.command()
def download(dataset: str = typer.Option("pneumonia", help=_DATASET_HELP)) -> None:
    """Download a MedMNIST dataset into the local cache."""
    from .data import get_dataloaders

    get_dataloaders(dataset, download=True)
    typer.echo(f"Downloaded '{dataset}'.")


@app.command()
def train(
    dataset: str = typer.Option("pneumonia", help=_DATASET_HELP),
    epochs: int = typer.Option(5, help="Training epochs."),
    batch_size: int = typer.Option(64),
    size: int = typer.Option(64, help="Image size (28/64/128/224)."),
    limit: int = typer.Option(0, help="Cap samples per split for quick runs (0 = all)."),
) -> None:
    """Train the classifier and save the best checkpoint."""
    from .train import train as train_model

    train_model(dataset, epochs=epochs, batch_size=batch_size, size=size, limit=limit or None)


@app.command()
def evaluate(
    dataset: str = typer.Option("pneumonia", help=_DATASET_HELP),
    size: int = typer.Option(64),
    limit: int = typer.Option(0, help="Cap test samples (0 = all)."),
) -> None:
    """Evaluate the saved model and write confusion matrix + ROC curves."""
    from .data import get_dataloaders
    from .evaluate import collect_predictions, compute_metrics, plot_confusion, plot_roc
    from .predict import load_checkpoint
    from .train import checkpoint_path

    model, ckpt = load_checkpoint(checkpoint_path(dataset))
    _, _, test_loader, meta = get_dataloaders(dataset, size=size, limit=limit or None)
    y_true, y_prob = collect_predictions(model, test_loader)

    metrics = compute_metrics(y_true, y_prob, meta.n_classes)
    typer.echo(f"Metrics: {metrics}")
    typer.echo(f"Saved: {plot_confusion(y_true, y_prob, meta.label_names)}")
    typer.echo(f"Saved: {plot_roc(y_true, y_prob, meta.n_classes)}")


@app.command()
def gradcam(
    dataset: str = typer.Option("pneumonia", help=_DATASET_HELP),
    index: int = typer.Option(0, help="Test-set image index to explain."),
    size: int = typer.Option(64),
) -> None:
    """Generate a Grad-CAM overlay for one test image."""
    from .data import get_test_arrays
    from .predict import explain_image, load_checkpoint
    from .train import checkpoint_path

    model, ckpt = load_checkpoint(checkpoint_path(dataset))
    images, labels = get_test_arrays(dataset, size=size, limit=index + 1)
    out = REPORTS_DIR / f"gradcam_{dataset}_{index}.png"
    path = explain_image(model, ckpt, images[index], out)
    typer.echo(f"True label: {ckpt['label_names'][str(int(labels[index]))]}")
    typer.echo(f"Saved Grad-CAM overlay to {path}")


@app.command()
def fairness(
    dataset: str = typer.Option("pneumonia", help=_DATASET_HELP),
    size: int = typer.Option(64),
    limit: int = typer.Option(0, help="Cap test samples (0 = all)."),
) -> None:
    """Audit subgroup performance (placeholder subgroups unless real metadata is supplied)."""
    from .data import get_dataloaders
    from .evaluate import collect_predictions
    from .fairness import disparity_summary, subgroup_metrics, synthetic_subgroups
    from .predict import load_checkpoint
    from .train import checkpoint_path

    model, ckpt = load_checkpoint(checkpoint_path(dataset))
    _, _, test_loader, meta = get_dataloaders(dataset, size=size, limit=limit or None)
    y_true, y_prob = collect_predictions(model, test_loader)

    typer.secho(
        "WARNING: using PLACEHOLDER subgroups (not real demographics). "
        "Supply real metadata for a genuine audit.",
        fg="yellow",
    )
    groups = synthetic_subgroups(len(y_true))
    metrics = subgroup_metrics(y_true, y_prob, groups)
    typer.echo(metrics.round(3).to_string())
    typer.echo(f"Disparities: {disparity_summary(metrics)}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(8000, help="Port to bind."),
) -> None:
    """Launch the interactive web UI (open http://HOST:PORT in a browser)."""
    from .web import serve as serve_app

    typer.echo(f"Serving web UI at http://{host}:{port}")
    serve_app(host=host, port=port)


if __name__ == "__main__":
    app()
