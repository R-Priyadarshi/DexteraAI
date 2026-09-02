# DexteraAI Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                       │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────────┐ │
│  │   Web   │  │  Mobile  │  │ Desktop │  │   Embedded   │ │
│  │ WebGPU  │  │  TFLite  │  │  ONNX   │  │ TFLite/ONNX  │ │
│  └────┬────┘  └────┬─────┘  └────┬────┘  └──────┬───────┘ │
└───────┼────────────┼────────────┼───────────────┼──────────┘
        │            │            │               │
┌───────┴────────────┴────────────┴───────────────┴──────────┐
│                    INFERENCE RUNTIME LAYER                   │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ ONNX Runtime   │  │  TFLite RT   │  │  PyTorch (dev) │  │
│  │ (Web/Desktop)  │  │  (Mobile)    │  │  (training)    │  │
│  └────────┬───────┘  └──────┬───────┘  └───────┬────────┘  │
└───────────┼─────────────────┼──────────────────┼───────────┘
            │                 │                  │
┌───────────┴─────────────────┴──────────────────┴───────────┐
│                      CORE ML PIPELINE                       │
│                                                             │
│  ┌──────────┐  ┌────────────┐  ┌───────────┐  ┌─────────┐ │
│  │  Vision  │→ │ Landmarks  │→ │ Temporal  │→ │ Gesture │ │
│  │MediaPipe │  │ Normalize  │  │Transformer│  │ Output  │ │
│  │  Hands   │  │ + Features │  │  (PyTorch)│  │         │ │
│  └──────────┘  └────────────┘  └───────────┘  └─────────┘ │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────────────┐        │
│  │   Calibration    │  │   Sequence Buffer         │        │
│  │   (per-user)     │  │   (sliding window)        │        │
│  └──────────────────┘  └──────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
            │
┌───────────┴─────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Datasets │  │ Trainers │  │  Eval    │  │   Export   │ │
│  │ (DVC)    │  │ (PyTorch)│  │ (Metrics)│  │(ONNX/TFLite│ │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │                    MLOps                          │      │
│  │    MLflow · DVC · GitHub Actions · Benchmarks     │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
---

## Module Overview

- **Training**: Distributed, plugin/callback, MLflow, Optuna, robust experiment tracking.
- **Inference**: Modular pipeline, ONNX/TFLite, batch/streaming, privacy, plugin/callbacks.
- **Calibration**: Per-user, plugin/callbacks, privacy, metrics.
- **Export**: ONNX/TFLite, plugin/callbacks, privacy, validation, versioning.
- **Integration**: Web, mobile, desktop, edge, plugin marketplace, API/CLI/SDK.

---

## Privacy & Security

- All inference and calibration can run fully on-device.
- No user data leaves device unless explicitly enabled.
- Enterprise logging, error handling, and metrics.

---

## Developer Experience

- Modular, extensible architecture (plugin/callbacks everywhere)
- CLI, API, SDK, and plugin marketplace
- Automated onboarding, documentation, and integration
- MLflow, DVC, GitHub Actions, benchmarks for MLOps

---

## Performance & Scale

- Real-time inference (<20ms per frame)
- Distributed training, robust experiment tracking
- Global deployment, accessibility, internationalization

## Data Flow

1. **Frame Capture** → Camera feed (webcam, phone camera, Raspberry Pi camera)
2. **Preprocessing** → Resize, flip, normalize via `FramePreprocessor`
3. **Hand Detection** → MediaPipe Hands extracts 21 3D landmarks per hand
4. **Normalization** → `LandmarkNormalizer` makes landmarks position/scale/rotation invariant
5. **Feature Extraction** → `LandmarkFeatureExtractor` produces 86-dim feature vector
6. **Temporal Buffering** → `SequenceBuffer` maintains sliding window of 30 frames
7. **Classification** → `GestureTransformer` classifies gesture from sequence
8. **Output** → `GestureResult` with label, confidence, and explainability data

## Key Design Decisions

### Why Landmark-Based (not pixel-based)?
- **Privacy**: Only 21×3 floats leave the detection stage, not images
- **Performance**: 86 features vs millions of pixels
- **Portability**: Same features work on all platforms
- **Augmentation**: Geometric transforms are trivial in landmark space

### Why Transformer (not LSTM/CNN)?
- **Self-attention** captures non-local temporal dependencies
- **Parallel** computation (no sequential bottleneck like LSTM)
- **CLS token** provides natural sequence-level representation
- **ONNX exportable** with fixed sequence length

### Why ONNX as Universal Format?
- Single export, runs everywhere: Web (WASM/WebGPU), Desktop, Edge
- Hardware-agnostic acceleration (CUDA, DirectML, CoreML, XNNPACK)
- Quantization support (int8, float16)
- < 1MB model after quantization

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `core/vision` | MediaPipe Tasks hand detection, holistic (hands + pose) detection, frame preprocessing |
| `core/landmarks` | Normalization, augmentation, 86-dim hand features, 208-dim holistic features |
| `core/temporal` | Transformer model, sequence buffering |
| `core/inference` | ONNX/TFLite runtime, full pipeline orchestration |
| `core/calibration` | Per-user gesture calibration |
| `training/datasets` | Landmark extraction from image corpora, dataset merging, DVC integration |
| `training/trainers` | PyTorch training loop with AMP, MLflow |
| `training/evaluation` | Metrics, latency benchmarks, confidence calibration |
| `training/export` | ONNX and TFLite conversion + quantization |
| `apps/web` | Browser-based demo (React + WebGPU) |
| `apps/desktop` | Native desktop app (Tauri) |
| `apps/mobile` | Mobile app (Flutter — planned) |

## Performance Budget

| Component | Budget | Actual (target) |
|-----------|--------|-----------------|
| MediaPipe detection | < 10ms | ~8ms (CPU) |
| Feature extraction | < 1ms | ~0.2ms |
| Transformer inference | < 5ms | ~3ms (ONNX CPU) |
| **Total pipeline** | **< 20ms** | **< 15ms** |
| Model size (quantized) | < 5MB | < 1MB |
| Memory usage | < 100MB | ~60MB |
