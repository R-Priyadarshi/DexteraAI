# DexteraAI Onboarding

## 1. Clone and set up the Python environment

```bash
git clone <repo-url>
cd DexteraAI
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,training]"
```

## 2. Fetch the MediaPipe model bundle

Hand detection needs a `.task` bundle that is not committed to the repo:

```bash
make fetch-models
```

This writes `models/mediapipe/hand_landmarker.task`. Without it, anything that
detects hands raises a `FileNotFoundError` explaining this step.

## 3. Verify the install

```bash
python dextera.py info                      # versions, CUDA, ORT providers
pytest tests/ -m "not slow"                 # unit + wiring tests
python dextera.py train --synthetic --epochs 2   # training loop smoke test
```

## 4. Run the web app

```bash
cd apps/web
npm install
npm run dev        # http://localhost:3000
```

The web app is self-contained: it runs MediaPipe and ONNX Runtime in the browser
and does not need the Python backend.

## 5. Optional: run the API server

```bash
python dextera.py serve          # http://localhost:8000/docs
curl http://localhost:8000/api/health
```

The server is optional and exists for demos and integration tests. It exposes
three endpoints (`/api/health`, `/api/predict`, `/api/ws/stream`). Inference is
on-device either way.

## 6. Train on real data

```bash
# 1. Get a dataset (see docs/DATASET_LICENSES.md before using one commercially)
# 2. Turn images into landmark sequences
python -m training.datasets.extract_landmarks \
    --parquet-dir data/raw/asl_parquet \
    --output data/sequences/asl_alphabet

# 3. Train, evaluate on a held-out split, and export a deployable bundle
python dextera.py train \
    --dataset data/sequences/asl_alphabet \
    --epochs 80 --export models/asl_alphabet
```

`--export` writes `gesture.onnx` plus `labels.json`. Both the Python pipeline and
the browser engine read the label set from that bundle, so the vocabulary is never
hardcoded in two places.

## Where things live

| Path | What it is |
|---|---|
| `core/` | The on-device pipeline: vision, landmarks, temporal model, inference |
| `training/` | Dataset extraction, training loop, evaluation, ONNX export |
| `apps/web/` | Next.js app; runs the whole pipeline client-side |
| `backend/` | Optional FastAPI server |
| `backend/experimental/` | Unmounted sketches. Not shipped. Read its README first. |
| `docs/` | Architecture, API reference, model card, dataset licensing |
