# DexteraAI — On-Device Gesture Recognition

> Real-time hand gesture recognition that runs entirely on the user's device.
> No images leave the machine. No cloud inference. No accounts.

[![CI](https://github.com/R-Priyadarshi/DexteraAI/actions/workflows/ultra_ci.yml/badge.svg)](https://github.com/R-Priyadarshi/DexteraAI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## What it does

A camera frame goes in; a gesture label comes out. Between those two points:

1. **MediaPipe** extracts 21 3D hand landmarks per hand.
2. **Normalization** makes those landmarks invariant to position, scale and rotation.
3. **Feature extraction** turns them into an 86-dim vector per frame: coordinates,
   fingertip distances, finger-curl ratios, palm normal.
4. A **Transformer encoder** classifies a 30-frame window of those vectors.

Only landmarks are ever persisted or processed downstream, never pixels. That is
what makes the privacy claim structural rather than a policy promise: after step 1,
there is no image left in the system to leak.

The same pipeline exists twice, deliberately: once in Python (`core/`) for training
and desktop/CLI use, and once in TypeScript (`apps/web/src/lib/gesture-engine.ts`)
so the browser runs it with no server at all.

---

## Vocabulary

Two model tracks, each shipped as a self-describing bundle (`gesture.onnx` +
`labels.json`). The client reads its vocabulary from the bundle, so Python and the
browser can never drift apart.

| Track | Classes | Data | Status |
|---|---|---|---|
| **General gestures** | 18 (call, dislike, fist, four, like, mute, ok, one, palm, peace, peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted) | HaGRID, 65,802 landmark samples | **98.1% test accuracy** |
| **ASL fingerspelling** | 26 (A–Z) | ASL alphabet, 8,638 landmark samples | **93.8% test accuracy** |
| **Custom, per-user** | Unlimited | Recorded in-browser by the user | Working |

### On "recognizing every gesture"

A closed-set classifier is finite by construction, so breadth comes from two places:

- **The trained vocabulary** above, as wide as licensing and data allow.
- **Teach-it-yourself.** Anything the model does not know, a user records in the
  browser in a few seconds. Each taught gesture becomes a prototype — a mean
  vector plus the spread of its own samples — and a query is scored in units of
  that spread, so tight and loose gestures are not held to one absolute
  threshold. See `apps/web/src/lib/few-shot.ts`. Gestures export as a JSON pack
  of landmark coordinates, so they move between machines.
- **Two-handed combinations.** Both hands are recognised independently, each with
  its own temporal window and segmenter, and composed into an ordered pair. This
  is not a two-hand *model* — the shipped models are trained on one hand — but it
  squares the command surface: 18 labels give 324 ordered pairs.

Confidence is calibrated (temperature scaling) so gestures outside the vocabulary
are reported as unrecognized rather than confidently mislabeled. Both shipped
bundles are calibrated: general gestures at T=0.736, ASL at T=0.843. See
`training/evaluation/calibrate_confidence.py`, and `scripts/calibrate_bundle.py`
to calibrate an existing checkpoint without retraining.

### From label to action

Recognition emits a label per frame; acting on that requires knowing when a
gesture *starts*. `apps/web/src/lib/gesture-segmenter.ts` converts the
per-frame stream into onset/hold/offset events with entry debounce, exit
hysteresis and a refractory period — without it, a held pose fires its bound
action ~30 times a second.

Bindings are keyed by gesture label rather than class index, since an index
belongs to whichever bundle is loaded and would silently re-point at an
unrelated gesture on a bundle swap.

Three ways to act on a gesture:

- **In-page actions** — scroll, deck navigation, plugin handlers.
- **Desktop actions** via the local bridge (`python -m bridge.server`) — media
  keys, volume, slides, window switching. Loopback-only and token-authenticated;
  see [bridge/README.md](bridge/README.md) for the threat model.
- **Hands-free pointer** — the index fingertip drives a cursor, and holding
  still for ~900ms clicks. This is the accessibility path: a vocabulary of
  discrete poses cannot reach an arbitrary target, only trigger a fixed binding.

### Known scope limits

Stated plainly, because the alternative is a promise the code does not keep:

- **Dynamic gestures are not in the shipped models.** Both bundles report
  `frames_per_sequence: 1` — every training sample is a still image replicated
  across the 30-frame window. The architecture is a temporal Transformer, but it
  has never been shown motion, so swipes and waves have no class to fall under
  and are handled by velocity heuristics instead. The obvious corpus (Jester) is
  licensed non-commercial, so the path provided is to record your own: the
  console's **Motion** panel captures clips at the model's window length, and
  `training/datasets/import_recordings.py` converts them into a trainable
  dataset. See [docs/DATASET_LICENSES.md](docs/DATASET_LICENSES.md).
- **Word-level sign language** has the input path built but no trained model.
  `core/vision/holistic_detector.py` and `core/landmarks/holistic_features.py`
  produce a 254-dim body-relative feature vector — both hands, upper-body pose,
  and the face points carrying non-manual markers, all scaled by shoulder span.
  ASL marks yes/no questions with raised brows, wh-questions with lowered brows,
  and negation with the mouth, so a hands-only model has a ceiling regardless of
  hand data. What is missing is training data that can actually ship: WLASL, the
  standard corpus, is research-licensed.
- **Continuous sign-language translation** is out of scope. Recognizing isolated
  signs is a different and much easier problem than translating fluent signing.

---

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,training]"
make fetch-models          # MediaPipe hand + pose + face bundles
python dextera.py info
```

Web app (self-contained, no backend needed):

```bash
cd apps/web && npm install && npm run dev
```

Train on real data end to end:

```bash
python -m training.datasets.extract_landmarks \
    --parquet-dir data/raw/asl_parquet --output data/sequences/asl_alphabet

python dextera.py train --dataset data/sequences/asl_alphabet \
    --epochs 80 --calibrate --export models/asl_alphabet
```

Train dynamic gestures from your own recordings (the console's **Motion** panel
exports the pack):

```bash
python training/datasets/import_recordings.py \
    --pack ~/Downloads/dextera-motion-*.json --out data/sequences/motion --mirror

python dextera.py train --dataset data/sequences/motion \
    --epochs 120 --calibrate --export models/motion
```

Control the desktop rather than just the tab:

```bash
pip install -e ".[bridge]"
python -m bridge.server        # prints a token; paste it into the Desktop panel
```

Full walkthrough in [ONBOARDING.md](ONBOARDING.md).

---

## Measured performance

Measured on this repo's own hardware, not targets. Numbers vary widely by device,
so measure on yours before quoting any of them.

The ONNX figures below were re-measured after fixing an inverted padding mask in
the benchmark and eval paths: both marked every frame as padding, so the encoder
attended to nothing and the old numbers timed a degenerate case. **Browser
latency is not listed, because it has not been measured on real hardware** —
figures from a headless software rasteriser would be meaningless, and the WebGPU
path is unavailable there by construction.

| Stage | Cost | Notes |
|---|---|---|
| MediaPipe hand detection | ~47 ms p50, ~84 ms p95 | CPU, 720p frame. Dominates the pipeline. |
| Normalize + feature extraction | ~0.7 ms | CPU |
| Transformer classification (PyTorch) | ~0.1 ms | CPU, 30x86 window |
| Transformer classification (ONNX Runtime) | 0.74 ms p50, 1.11 ms p95 | CPU, includes session overhead |
| Model size | 2.1 MB fp32 | ~551k parameters |

### Accuracy

Held-out test splits, never trained or validated on. Splits are random rather
than subject-disjoint (the datasets carry no subject IDs), so accuracy on new
people will be lower.

| Model | Classes | Test accuracy | Macro F1 | Top-3 |
|---|---|---|---|---|
| General gestures | 18 | 98.1% | 98.0% | 99.4% |
| ASL fingerspelling | 26 | 93.8% | 93.6% | 97.6% |

Confidence is calibrated by temperature scaling on the validation split. For the
general-gesture model this cut expected calibration error from 0.089 to 0.017,
and the fitted rejection threshold holds 97.9% accuracy at full coverage.

The classifier is effectively free; hand detection is the entire latency budget.
On CPU this lands well above a 60 FPS budget. The browser path uses MediaPipe's
WASM/GPU build and ONNX Runtime Web with WebGPU, which is substantially faster than
the CPU numbers above, but it has not been benchmarked here and no number is quoted
for it.

Reproduce with:

```bash
python dextera.py benchmark --checkpoint models/<name>/gesture.onnx --onnx
```

---

## Architecture

```
Camera ─► Preprocess ─► MediaPipe Hands ─► Normalize ─► 86-d features
                                                              │
                                              30-frame sliding buffer
                                                              │
                                          Transformer encoder (CLS token)
                                                              │
                                            label + calibrated confidence
```

| Module | Responsibility |
|---|---|
| `core/vision` | MediaPipe Tasks hand detection, holistic hands+pose detection, preprocessing |
| `core/landmarks` | Normalization, augmentation, 86-dim hand and 208-dim holistic features |
| `core/temporal` | Transformer classifier, static MLP baseline, sequence buffer |
| `core/inference` | Pipeline orchestration, ONNX runtime |
| `core/calibration` | Per-user personalization |
| `training/datasets` | Landmark extraction from image corpora, dataset merging |
| `training/trainers` | Training loop: AMP, warmup + cosine LR, early stopping, MLflow |
| `training/evaluation` | Metrics, latency benchmarks, confidence calibration |
| `training/export` | ONNX export and quantization |
| `apps/web` | Next.js app running the full pipeline client-side |
| `apps/desktop` | Tauri shell around the web build |
| `backend/` | Optional FastAPI server (3 endpoints) |

More detail in [docs/architecture.md](docs/architecture.md).

---

## Deployment

The primary artifact is a static web build plus a model bundle. See
[ROBUST_DEPLOYMENT_GUIDE.md](ROBUST_DEPLOYMENT_GUIDE.md). The API server is optional
and has no authentication; it is for local demos, not public hosting.

```bash
docker build -t dextera-ai . && docker run -p 8000:8000 dextera-ai
```

---

## Development

```bash
make lint          # ruff + mypy
make test          # pytest
cd apps/web && npx tsc --noEmit && npm test && npm run build
```

CI runs lint, types, tests, an end-to-end wiring smoke test, and the web build.
The smoke test exists because unit tests alone previously missed three modules that
passed in isolation while calling each other with mismatched APIs.

---

Built by [R-Priyadarshi](https://github.com/R-Priyadarshi). MIT licensed.
