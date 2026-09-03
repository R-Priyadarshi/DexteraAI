# Dextera AI — API Reference

> **NOTE**: The API server is OPTIONAL. The core ML pipeline runs entirely on-device.
> Use `python dextera.py serve` to start the server.

## Base URL

```
http://localhost:8000/api
```

---

## Endpoints

### `GET /api/health`

Health check / readiness probe.

**Response** `200 OK`

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "pipeline_running": true,
  "uptime_seconds": 123.45,
  "privacy": "all-inference-on-device"
}
```

---

### `POST /api/predict`

Predict gesture from an uploaded image.

**Request**: `multipart/form-data` with `file` field (JPEG/PNG).

**Response** `200 OK`

```json
{
  "success": true,
  "predictions": [
    {
      "gesture_name": "open_palm",
      "gesture_id": 1,
      "confidence": 0.95
    }
  ],
  "num_hands": 1,
  "inference_ms": 12.3,
  "privacy_mode": "on-device"
}
```

---

### `WS /api/ws/stream`

Real-time gesture prediction over WebSocket.

**Protocol**:
- Client → Server: base64-encoded JPEG frame (text message)
- Server → Client: JSON with gesture predictions

```json
{
  "frame_id": 42,
  "gestures": [
    {
      "gesture_name": "peace",
      "gesture_id": 5,
      "confidence": 0.91
    }
  ],
  "num_hands": 1,
  "inference_ms": 8.7,
  "privacy_mode": "on-device"
}
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `python dextera.py train` | Train a gesture model |
| `python dextera.py eval` | Evaluate a trained model |
| `python dextera.py export` | Export to ONNX/TFLite |
| `python dextera.py demo` | Real-time webcam demo |
| `python dextera.py benchmark` | Latency benchmark |
| `python dextera.py serve` | Start FastAPI server |
| `python dextera.py info` | Show system info |

---

## Core Modules

### Training
- `GestureTrainer`: Production-grade training loop with distributed, plugin/callback, MLflow tracking, Optuna tuning.
- `TrainConfig`: Hyperparameters, reproducibility, checkpointing.
- `TrainResult`: Metrics, history, artifact saving.

### Inference
- `GesturePipeline`: Modular, extensible pipeline for real-time gesture recognition.
- `ONNXInferenceRuntime`: Cross-platform ONNX inference, privacy, batch/streaming, plugin/callbacks.

### Calibration
- `UserCalibrator`: Per-user calibration, plugin/callbacks, privacy, metrics.
- `CalibrationProfile`: User-specific reference data.

### Export
- `export_to_onnx`: ONNX export, plugin/callbacks, privacy, validation, versioning.
- `export_to_tflite`: TFLite export, plugin/callbacks, privacy, quantization, validation.

### Integration
- Web, mobile, desktop, edge: API, CLI, SDK, plugin marketplace.

---

## Privacy & Security
- All inference can run fully on-device.
- No user data leaves device unless explicitly enabled.
- Calibration and export support privacy-preserving modes.
- Enterprise-grade logging, error handling, and metrics.

---

## Developer Onboarding
- See `README.md` for quickstart, CLI, and integration.
- All modules are extensible via plugin/callback architecture.
- MLflow, DVC, GitHub Actions, benchmarks for MLOps.
- API and CLI are documented and auto-generated.

---

## Performance Targets
- Real-time inference (<20ms per frame on modern devices)
- Distributed training, robust experiment tracking
- Modular, scalable, privacy-preserving architecture

---

## Plugin & Callback API

DexteraAI supports a plugin/callback architecture for extensibility:

### Callback Hooks
- `on_train_start(trainer)`
- `on_epoch_start(epoch, trainer)`
- `on_epoch_end(epoch, metrics, trainer)`
- `on_train_end(trainer, result)`

### Custom Plugin Example
```python
class MyPlugin(Callback):
    def on_epoch_end(self, epoch, metrics, trainer):
        print(f"Epoch {epoch} finished. Metrics: {metrics}")
```

### Register Plugins
```python
trainer = GestureTrainer(model, dataset, config, callbacks=[MyPlugin()])
```

---

## Global Integration Examples

- **Web**: Use ONNX.js or WebGPU for browser inference
- **Mobile**: TFLite integration for Android/iOS
- **Desktop**: ONNX runtime, plugin/callbacks
- **Edge/Robotics**: TFLite/ONNX, real-time, low-power

See `/apps/web`, `/apps/mobile`, `/apps/desktop` for starter templates.

---

## Accessibility

- **Hands-free pointer**: a fingertip cursor with dwell-to-click, for use
  without a mouse. See `pointer-engine.ts`.
- **ASL fingerspelling**: a 26-letter model, with the holistic feature path
  built for word-level signs (no trained model yet).
- **Teach-your-own gestures**: few-shot prototypes for anything the shipped
  vocabulary does not cover.

Not present, though earlier drafts of this document claimed otherwise: there is
no multi-language UI and no locale-aware onboarding — `core/accessibility.py`
held a translation table covering two of the eight languages it advertised, and
has been deleted rather than left to imply a feature. No formal GDPR or CCPA
compliance work has been done either; what is true is narrower and stronger —
inference runs on-device and no image data leaves the machine.

---

## Not Currently Served

Earlier drafts of this document listed endpoints for plugins, retraining, cloud/edge sync,
multimodal ingest, notifications, custom gestures, integrations, analytics, and RBAC.

**None of those are mounted, and the sketches have now been deleted.** The application
serves only the three endpoints above. Several of the sketches contained security defects
(path traversal, SSRF) and none had authentication, so leaving them in the tree was a
liability even unreachable. `backend/experimental/README.md` records what they were, what
was wrong with each, and how to recover them from git history.

Two of them have since been answered without a server: custom-gesture sync is handled by
gesture packs (a JSON export that imports on another machine), and plugins are TypeScript
objects registered at startup rather than records in a server registry.

The product runs inference on-device; the API server is an optional convenience for demos
and integration tests, not the primary delivery path.

---

## Model Bundles

A trained model ships as a directory containing the ONNX graph and its label set:

```
models/<name>/
├── gesture.onnx     # exported graph, inputs: input (1, seq_len, 86), mask (1, seq_len)
└── labels.json      # {"labels": [...], "seq_len": 30, "feature_dim": 86, ...}
```

`GesturePipeline` reads `labels.json` from the directory next to the checkpoint, so the
label set travels with the model instead of being hardcoded in the Python and TypeScript
clients separately.
