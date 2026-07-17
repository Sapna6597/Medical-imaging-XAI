# Data Ethics & Responsible Use

## Datasets

This project uses **MedMNIST v2** (Yang et al., *Scientific Data*, 2023) — a
collection of standardized, openly licensed (CC BY 4.0) biomedical image
datasets derived from established sources and de-identified by their curators.

| Task | MedMNIST dataset | Source domain |
|------|------------------|---------------|
| Pneumonia detection | PneumoniaMNIST | Pediatric chest X-rays |
| Diabetic retinopathy | RetinaMNIST | Retinal fundus images |
| Skin lesion classification | DermaMNIST | Dermatoscopic images (HAM10000) |

Data is downloaded on demand into a local cache (git-ignored). No data is
committed to the repository.

## Principles

1. **No PHI committed.** Only code is versioned; images live in the local cache.
2. **Transparency.** Every prediction can be explained with Grad-CAM.
3. **Fairness.** Subgroup performance is reported before any performance claim.
4. **Honest limitations.** See below.

## Important limitations

- **This is an educational demonstrator, not a medical device.** It must never be
  used for clinical decisions.
- **Fairness demo uses PLACEHOLDER subgroups.** MedMNIST does not include
  demographic labels, so `fairness.synthetic_subgroups` generates a reproducible
  but *artificial* attribute. For a genuine demographic audit, supply real
  metadata — e.g., Patient Age / Patient Gender from
  [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) — as the
  `groups` argument to `subgroup_metrics`.
- **Small default image size (64px)** favors speed over accuracy. Use `--size 224`
  and the full dataset for research-grade results.

## Scaling to full clinical datasets

- **NIH ChestX-ray14** — 112k images with age/gender metadata (enables real
  demographic fairness analysis).
- **APTOS 2019 / EyePACS** — diabetic retinopathy grading.
- **ISIC Archive** — dermoscopic skin lesion images.

These require accepting dataset-specific licenses and, in some cases,
registration. Follow each provider's terms.
