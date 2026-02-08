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
```

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
| `core/vision` | MediaPipe hand detection, frame preprocessing |
| `core/landmarks` | Normalization, augmentation, feature extraction |
| `core/temporal` | Transformer model, sequence buffering |
| `core/inference` | ONNX/TFLite runtime, full pipeline orchestration |
| `core/calibration` | Per-user gesture calibration |
| `training/datasets` | Dataset loading, DVC integration |
| `training/trainers` | PyTorch training loop with AMP, MLflow |
| `training/evaluation` | Metrics, benchmarks, model cards |
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
