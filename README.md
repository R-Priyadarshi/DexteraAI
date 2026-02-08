# DexteraAI — Gesture Intelligence Platform

> Real-time, on-device, privacy-preserving hand-gesture recognition platform.
> Web · Mobile · Desktop · Embedded · Robotics

[![CI](https://github.com/R-Priyadarshi/DexteraAI/actions/workflows/ci.yml/badge.svg)](https://github.com/R-Priyadarshi/DexteraAI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🎯 What This Is

DexteraAI is a **gesture intelligence platform** — not a demo. It provides:

- **Real-time hand landmark detection** via MediaPipe
- **Transformer-based temporal gesture modeling** for sequence recognition
- **On-device inference** (zero cloud, zero data leakage)
- **Cross-platform**: Web (WebGPU), Mobile (TFLite), Desktop (ONNX), Edge, Robotics
- **Few-shot & zero-shot gesture learning**
- **Accessibility-first** design (motor disability support)

## 🏗 Project Structure

```
DexteraAI/
├── core/                  # Core ML pipeline (platform-agnostic)
│   ├── types.py           # Shared types, protocols, constants
│   ├── vision/            # MediaPipe hand detection + preprocessing
│   │   ├── detector.py
│   │   └── preprocessor.py
│   ├── landmarks/         # Landmark normalization, augmentation, features
│   │   ├── normalizer.py
│   │   ├── augmentor.py
│   │   └── features.py
│   ├── temporal/          # Transformer gesture sequence model
│   │   ├── model.py       # GestureTransformer (temporal, CLS token)
│   │   ├── static_model.py# StaticGestureClassifier (single-frame MLP)
│   │   └── sequence_buffer.py
│   ├── inference/         # ONNX runtime + full pipeline orchestration
│   │   ├── onnx_runtime.py
│   │   └── pipeline.py
│   └── calibration/       # Per-user gesture calibration
│       └── calibrator.py
├── training/              # Training pipeline
│   ├── datasets/          # Dataset loaders (real + synthetic)
│   │   └── gesture_dataset.py
│   ├── trainers/          # PyTorch training loop (AMP, MLflow, early stopping)
│   │   └── train_gesture.py
│   ├── evaluation/        # Metrics, confusion matrix, latency benchmarks
│   │   └── metrics.py
│   └── export/            # ONNX / TFLite export + quantization
│       ├── to_onnx.py
│       └── to_tflite.py
├── backend/               # FastAPI server (OPTIONAL — for remote inference)
│   ├── config.py          # Pydantic settings
│   ├── logging_config.py  # Loguru structured logging
│   └── apps/api/          # REST + WebSocket endpoints
│       ├── main.py
│       ├── routes.py
│       ├── schemas.py
│       ├── middleware.py
│       └── dependencies.py
├── apps/                  # Application layer
│   ├── web/               # Next.js + ONNX Runtime Web + WebGPU
│   ├── desktop/           # Tauri (Rust + Web frontend)
│   └── mobile/            # Flutter (planned)
├── tests/                 # pytest + hypothesis
├── docs/                  # Architecture docs, model cards, API reference
├── dextera.py             # CLI: train / eval / export / demo / benchmark / serve / info
├── pyproject.toml         # Python project config + all dependencies
├── Makefile               # Developer convenience commands
├── Dockerfile             # Multi-stage production container
├── dvc.yaml               # DVC pipeline (data → train → eval → export)
└── .github/workflows/     # CI/CD (lint, test, benchmark)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+ (web app only)

### Install & Test
```bash
git clone https://github.com/R-Priyadarshi/DexteraAI.git
cd DexteraAI

# Install all dependencies
make dev
# — or manually —
pip install -e ".[dev,training]"

# Run tests
make test-fast

# See all available commands
make help
```

### CLI Usage
```bash
# Train with synthetic data (pipeline test)
python dextera.py train --synthetic --epochs 10

# Train on real data
python dextera.py train --dataset data/gestures --epochs 100 --device auto

# Evaluate
python dextera.py eval --checkpoint checkpoints/best.pt --dataset data/test

# Export to ONNX (+ quantization)
python dextera.py export --checkpoint checkpoints/best.pt --format onnx --quantize

# Webcam demo (detection-only, no trained model needed)
python dextera.py demo

# Latency benchmark
python dextera.py benchmark --checkpoint checkpoints/best.pt

# Start FastAPI server (optional — for remote inference)
python dextera.py serve --port 8000

# Show system info
python dextera.py info
```

### Web App
```bash
cd apps/web
npm install
npm run dev
# Open http://localhost:3000
```

## ⚡ Performance Targets

| Metric | Target |
|--------|--------|
| End-to-end latency | < 20ms |
| FPS | 60 FPS real-time |
| CPU fallback | ✅ No GPU required |
| Model size (quantized) | < 1MB |
| Memory usage | < 100MB |

## 🛡 Privacy

- **Zero cloud inference** — all processing on-device
- **No image/video leaves the device** — ever
- **Landmark-only pipeline** — only 21×3 floats, not pixels
- GDPR / DPDP / HIPAA-safe by design

## 📜 License

MIT — see [LICENSE](LICENSE)
