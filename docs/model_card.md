# Model Card: Medical Image Classifier (SmallCNN)

## Overview

- **Task:** Image classification for one of: pneumonia (binary), diabetic
  retinopathy grade (5-class), or skin lesion type (7-class).
- **Architecture:** 3-block convolutional network (`SmallCNN`) with global
  average pooling and a linear head; last conv block exposed for Grad-CAM.
- **Version:** 0.1.0

## Intended use

- **Intended:** Educational / research demonstration of an explainable, fairness-
  aware medical imaging pipeline.
- **Out of scope:** Any real diagnostic or clinical use.

## Training data

- MedMNIST v2 datasets (PneumoniaMNIST / RetinaMNIST / DermaMNIST), official
  train/val/test splits. Default configuration downsamples to 64px and may cap
  samples for speed.

## Evaluation

- Reported metrics: accuracy and ROC-AUC (binary or macro one-vs-rest), plus a
  confusion matrix and ROC curves saved to `reports/`.

## Explainability

- Grad-CAM heatmaps localize the image regions driving each prediction.

## Ethical considerations

- Subgroup fairness is auditable via `fairness.py`. The bundled demo uses
  **placeholder** subgroups (see `docs/data_ethics.md`); real demographic audits
  require external metadata.
- Class imbalance (e.g., DermaMNIST) can bias accuracy; consult ROC-AUC and the
  confusion matrix, not accuracy alone.

## Caveats & recommendations

- Increase image size, epochs, and use the full dataset for meaningful accuracy.
- Consider transfer learning (e.g., ImageNet-pretrained backbones) and
  calibration before interpreting probabilities.
