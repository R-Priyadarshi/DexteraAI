# Deployment Guide

For first-time setup see [ONBOARDING.md](ONBOARDING.md). This document covers
shipping DexteraAI to users.

## What actually gets deployed

DexteraAI runs inference on-device, so "deployment" usually means shipping a static
web app plus a model bundle, not standing up an inference server.

| Target | What ships | Inference runs |
|---|---|---|
| **Web** (primary) | Static Next.js export + `gesture.onnx` + `labels.json` + MediaPipe WASM | In the browser (WebGPU, WASM fallback) |
| **Desktop** | Tauri shell wrapping the same web build | On the user's machine |
| **API server** (optional) | Docker image running FastAPI | On the server, for demos and integration tests |

## 1. Web deployment (primary path)

```bash
cd apps/web
npm ci
npm run build          # static export to apps/web/out/
```

Serve `apps/web/out/` from any static host. Requirements:

- **HTTPS is mandatory.** `getUserMedia` (camera access) is refused on plain HTTP
  from anything but `localhost`.
- Serve the `.wasm` and `.onnx` files with correct MIME types and long-lived cache
  headers. They are large and immutable per release.
- If you enable cross-origin isolation for WASM threading, set
  `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`. Without it ONNX Runtime falls back
  to single-threaded WASM, which is slower but still works.

### Shipping a model bundle

`python dextera.py train ... --export models/<name>` produces:

```
models/<name>/
├── gesture.onnx
└── labels.json
```

Copy both into `apps/web/public/onnx/`. The browser engine fetches `labels.json`
next to the `.onnx` file, so the vocabulary and sequence length travel with the
model. Do not hardcode labels in the client.

## 2. Desktop (Tauri)

```bash
cd apps/desktop
cargo tauri build
```

The Tauri config builds `apps/web` first and bundles `apps/web/out`. The CSP in
`tauri.conf.json` already allows `mediastream:` for camera access; keep it that way
or the camera will silently fail.

## 3. API server (optional)

```bash
docker build -t dextera-ai .
docker run -p 8000:8000 dextera-ai
curl http://localhost:8000/api/health
```

Note the container's `HEALTHCHECK` targets `/api/health`. The router is mounted
under the `/api` prefix, so a bare `/health` returns 404 and marks the container
unhealthy.

The image does not include the MediaPipe `.task` bundle. Either bake it in by
adding a fetch step to the Dockerfile, mount it at runtime, or set
`DEXTERA_HAND_LANDMARKER` to its path.

Before exposing this server to a network, note that it has **no authentication and
no rate limiting**. It is built for local demos. Do not put it on the public
internet without putting real auth in front of it.

## 4. Pre-release checklist

- [ ] `ruff check .` and `mypy core/` clean
- [ ] `pytest tests/` passing, including `tests/test_pipeline.py` and `tests/test_api.py`
- [ ] `npx tsc --noEmit` and `npm run build` clean in `apps/web`
- [ ] Model bundle exported, and `labels.json` matches the ONNX output dimension
- [ ] `python dextera.py benchmark --checkpoint models/<name>/gesture.onnx --onnx`
      run on hardware comparable to the target, with results recorded
- [ ] Every training dataset listed in [docs/DATASET_LICENSES.md](docs/DATASET_LICENSES.md)
      with commercial-use status confirmed
- [ ] `docs/model_card.md` updated with the measured accuracy of the shipped model
- [ ] Camera permission flow tested on the real target browsers

## 5. Performance expectations

Measure on your own target hardware. Latency is dominated by MediaPipe hand
detection, not by the gesture classifier, and it varies widely by device:

- The classifier is ~0.1 ms per window on CPU, effectively free.
- Hand detection is tens of milliseconds on a CPU, and much faster with GPU/WebGPU
  delegation in the browser.

Do not quote numbers you have not measured on the hardware you are shipping to.
