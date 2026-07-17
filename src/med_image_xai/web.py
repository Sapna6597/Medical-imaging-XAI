"""Interactive web UI (FastAPI) for the medical imaging XAI toolkit.

Serves a single-page app that lets you:
  * pick a dataset (pneumonia / retina / derma),
  * run a prediction on a test image and view its Grad-CAM explanation,
  * evaluate the saved model (metrics + confusion matrix + ROC curve),
  * run the subgroup fairness audit (placeholder subgroups).

Run with:  ``med-image-xai serve``  or  ``python -m med_image_xai.web``
Then open http://127.0.0.1:8000 in your browser.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .config import DATASETS, REPORTS_DIR
from .data import get_dataloaders, get_meta, get_test_arrays
from .evaluate import collect_predictions, compute_metrics, plot_confusion, plot_roc
from .fairness import disparity_summary, subgroup_metrics, synthetic_subgroups
from .predict import explain_image, load_checkpoint, predict_image
from .train import checkpoint_path

app = FastAPI(title="Medical Imaging XAI", version="0.1.0")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _png_to_data_uri(path: str | Path) -> str:
    """Read a PNG file and return it as a base64 data URI for inline display."""
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


@lru_cache(maxsize=len(DATASETS))
def _load_model(dataset: str):
    """Load (and cache) a checkpoint. Raises HTTPException if missing."""
    try:
        return load_checkpoint(checkpoint_path(dataset))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No trained model for '{dataset}'. Train one first, e.g. "
                f"`med-image-xai train --dataset {dataset} --epochs 3 --limit 500`."
            ),
        ) from exc


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    dataset: str = "pneumonia"
    index: int = 0
    size: int = 64


class EvalRequest(BaseModel):
    dataset: str = "pneumonia"
    size: int = 64
    limit: int = 200


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/datasets")
def list_datasets() -> dict:
    """Return available datasets and whether a trained checkpoint exists."""
    items = []
    for name, flag in DATASETS.items():
        meta = get_meta(name)
        items.append(
            {
                "name": name,
                "flag": flag,
                "task": meta.task,
                "n_classes": meta.n_classes,
                "labels": meta.label_names,
                "trained": checkpoint_path(name).exists(),
            }
        )
    return {"datasets": items}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    """Predict a single test image and return its Grad-CAM overlay."""
    model, ckpt = _load_model(req.dataset)
    images, labels = get_test_arrays(req.dataset, size=req.size, limit=req.index + 1)
    if req.index >= len(images):
        raise HTTPException(status_code=400, detail=f"Index {req.index} out of range.")

    image = images[req.index]
    pred = predict_image(model, ckpt, image)
    out = REPORTS_DIR / f"gradcam_{req.dataset}_{req.index}.png"
    explain_image(model, ckpt, image, out)

    labels_map = ckpt["label_names"]
    return {
        "dataset": req.dataset,
        "index": req.index,
        "prediction": pred["label"],
        "true_label": labels_map[str(int(labels[req.index]))],
        "probabilities": {labels_map[str(i)]: p for i, p in enumerate(pred["probabilities"])},
        "gradcam": _png_to_data_uri(out),
    }


@app.post("/api/evaluate")
def evaluate(req: EvalRequest) -> dict:
    """Evaluate the saved model: metrics + confusion matrix + ROC curve."""
    model, ckpt = _load_model(req.dataset)
    _, _, test_loader, meta = get_dataloaders(req.dataset, size=req.size, limit=req.limit or None)
    y_true, y_prob = collect_predictions(model, test_loader)

    metrics = compute_metrics(y_true, y_prob, meta.n_classes)
    cm_path = plot_confusion(y_true, y_prob, meta.label_names)
    roc_path = plot_roc(y_true, y_prob, meta.n_classes)
    return {
        "metrics": metrics,
        "confusion_matrix": _png_to_data_uri(cm_path),
        "roc_curve": _png_to_data_uri(roc_path),
    }


@app.post("/api/fairness")
def fairness(req: EvalRequest) -> dict:
    """Run the subgroup fairness audit (placeholder subgroups)."""
    model, ckpt = _load_model(req.dataset)
    _, _, test_loader, _ = get_dataloaders(req.dataset, size=req.size, limit=req.limit or None)
    y_true, y_prob = collect_predictions(model, test_loader)

    groups = synthetic_subgroups(len(y_true))
    metrics = subgroup_metrics(y_true, y_prob, groups)
    return {
        "warning": (
            "PLACEHOLDER subgroups (not real demographics). "
            "Supply real metadata for a genuine audit."
        ),
        "subgroups": metrics.reset_index().round(4).to_dict(orient="records"),
        "disparities": disparity_summary(metrics),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-page UI."""
    return _INDEX_HTML


# --------------------------------------------------------------------------- #
# Front-end (single self-contained page)
# --------------------------------------------------------------------------- #
_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Medical Imaging XAI</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --accent:#38bdf8; --text:#e2e8f0; --muted:#94a3b8; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:var(--bg); color:var(--text); }
  header { padding:24px 32px; border-bottom:1px solid #334155; }
  header h1 { margin:0 0 4px; font-size:22px; }
  header p { margin:0; color:var(--muted); font-size:13px; }
  .disclaimer { background:#78350f; color:#fde68a; padding:10px 32px; font-size:13px; }
  main { max-width:1080px; margin:0 auto; padding:24px 32px 64px; }
  .controls { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-end;
              background:var(--card); padding:18px; border-radius:12px; margin-bottom:24px; }
  .field { display:flex; flex-direction:column; gap:4px; }
  .field label { font-size:12px; color:var(--muted); }
  select, input { background:#0b1220; color:var(--text); border:1px solid #334155;
                  border-radius:8px; padding:8px 10px; font-size:14px; }
  button { background:var(--accent); color:#082f49; border:none; border-radius:8px;
           padding:9px 16px; font-weight:600; cursor:pointer; font-size:14px; }
  button:hover { filter:brightness(1.08); }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .btn-row { display:flex; gap:10px; flex-wrap:wrap; }
  .card { background:var(--card); border-radius:12px; padding:20px; margin-bottom:20px; }
  .card h2 { margin:0 0 12px; font-size:16px; }
  img { max-width:100%; border-radius:8px; background:#000; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:6px 10px; border-bottom:1px solid #334155; }
  .bars { display:flex; flex-direction:column; gap:6px; }
  .bar { display:flex; align-items:center; gap:8px; font-size:13px; }
  .bar .track { flex:1; background:#0b1220; border-radius:6px; overflow:hidden; height:16px; }
  .bar .fill { height:100%; background:var(--accent); }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; }
  .ok { background:#065f46; color:#a7f3d0; }
  .no { background:#7f1d1d; color:#fecaca; }
  .status { color:var(--muted); font-size:13px; margin-left:8px; }
  .hidden { display:none; }
  pre { background:#0b1220; padding:12px; border-radius:8px; overflow:auto; font-size:13px; }
</style>
</head>
<body>
<header>
  <h1>🩻 Medical Imaging XAI</h1>
  <p>Explainable, fairness-aware deep learning for medical image classification.</p>
</header>
<div class="disclaimer">⚕️ Educational demonstrator — not a medical device. Uses openly licensed MedMNIST data.</div>

<main>
  <div class="controls">
    <div class="field">
      <label for="dataset">Dataset</label>
      <select id="dataset"></select>
    </div>
    <div class="field">
      <label for="index">Test image index</label>
      <input id="index" type="number" value="0" min="0" style="width:120px" />
    </div>
    <div class="field">
      <label for="limit">Eval sample cap</label>
      <input id="limit" type="number" value="200" min="10" style="width:120px" />
    </div>
    <div class="btn-row">
      <button id="btnPredict">Predict &amp; Explain</button>
      <button id="btnEval">Evaluate</button>
      <button id="btnFair">Fairness audit</button>
      <span class="status" id="status"></span>
    </div>
  </div>

  <div id="datasetInfo" class="card"></div>

  <div id="predictCard" class="card hidden">
    <h2>Prediction &amp; Grad-CAM</h2>
    <div id="predictSummary"></div>
    <div class="bars" id="probBars"></div>
    <img id="gradcamImg" alt="Grad-CAM overlay" style="margin-top:14px" />
  </div>

  <div id="evalCard" class="card hidden">
    <h2>Evaluation</h2>
    <pre id="evalMetrics"></pre>
    <div style="display:flex; gap:16px; flex-wrap:wrap">
      <div style="flex:1; min-width:280px"><img id="cmImg" alt="Confusion matrix" /></div>
      <div style="flex:1; min-width:280px"><img id="rocImg" alt="ROC curve" /></div>
    </div>
  </div>

  <div id="fairCard" class="card hidden">
    <h2>Fairness audit</h2>
    <div class="disclaimer" id="fairWarn" style="padding:8px 12px; border-radius:8px; margin-bottom:12px"></div>
    <table id="fairTable"></table>
    <pre id="fairGaps" style="margin-top:12px"></pre>
  </div>
</main>

<script>
const $ = (id) => document.getElementById(id);
const status = (msg) => { $("status").textContent = msg || ""; };

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

let DATASETS = {};

async function loadDatasets() {
  const data = await api("/api/datasets");
  const sel = $("dataset");
  sel.innerHTML = "";
  data.datasets.forEach((d) => {
    DATASETS[d.name] = d;
    const opt = document.createElement("option");
    opt.value = d.name;
    opt.textContent = `${d.name} — ${d.task} (${d.n_classes} classes)`;
    sel.appendChild(opt);
  });
  renderDatasetInfo();
}

function renderDatasetInfo() {
  const d = DATASETS[$("dataset").value];
  if (!d) return;
  const pill = d.trained
    ? '<span class="pill ok">checkpoint available</span>'
    : '<span class="pill no">not trained yet</span>';
  const labels = Object.values(d.labels).join(", ");
  $("datasetInfo").innerHTML =
    `<h2>${d.name} ${pill}</h2>` +
    `<div style="color:var(--muted); font-size:13px">Task: ${d.task} · ` +
    `Classes: ${labels}</div>`;
}

function currentReq() {
  return {
    dataset: $("dataset").value,
    index: parseInt($("index").value || "0", 10),
    limit: parseInt($("limit").value || "200", 10),
    size: 64,
  };
}

function probBars(probs) {
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  return entries
    .map(([label, p]) => {
      const pct = (p * 100).toFixed(1);
      return `<div class="bar"><span style="width:120px">${label}</span>` +
             `<div class="track"><div class="fill" style="width:${pct}%"></div></div>` +
             `<span style="width:54px; text-align:right">${pct}%</span></div>`;
    })
    .join("");
}

$("btnPredict").onclick = async () => {
  try {
    status("Running prediction & Grad-CAM…");
    const r = await api("/api/predict", currentReq());
    const correct = r.prediction === r.true_label;
    $("predictSummary").innerHTML =
      `<div>Predicted: <b>${r.prediction}</b> · True: <b>${r.true_label}</b> ` +
      (correct ? '<span class="pill ok">correct</span>' : '<span class="pill no">incorrect</span>') +
      `</div>`;
    $("probBars").innerHTML = probBars(r.probabilities);
    $("gradcamImg").src = r.gradcam;
    $("predictCard").classList.remove("hidden");
    status("");
  } catch (e) { status("⚠ " + e.message); }
};

$("btnEval").onclick = async () => {
  try {
    status("Evaluating model…");
    const r = await api("/api/evaluate", currentReq());
    $("evalMetrics").textContent = JSON.stringify(r.metrics, null, 2);
    $("cmImg").src = r.confusion_matrix;
    $("rocImg").src = r.roc_curve;
    $("evalCard").classList.remove("hidden");
    status("");
  } catch (e) { status("⚠ " + e.message); }
};

$("btnFair").onclick = async () => {
  try {
    status("Running fairness audit…");
    const r = await api("/api/fairness", currentReq());
    $("fairWarn").textContent = "⚠ " + r.warning;
    const rows = r.subgroups;
    if (rows.length) {
      const cols = Object.keys(rows[0]);
      const head = "<tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr>";
      const body = rows
        .map((row) => "<tr>" + cols.map((c) => `<td>${row[c]}</td>`).join("") + "</tr>")
        .join("");
      $("fairTable").innerHTML = head + body;
    }
    $("fairGaps").textContent = "Disparities: " + JSON.stringify(r.disparities, null, 2);
    $("fairCard").classList.remove("hidden");
    status("");
  } catch (e) { status("⚠ " + e.message); }
};

$("dataset").onchange = renderDatasetInfo;
loadDatasets().catch((e) => status("⚠ " + e.message));
</script>
</body>
</html>"""


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the web UI with uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
