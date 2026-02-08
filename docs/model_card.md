# Model Card: GestureTransformer v0.1

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | GestureTransformer |
| **Version** | 0.1.0 |
| **Architecture** | Transformer Encoder + CLS Token |
| **Parameters** | ~500K (default config) |
| **Input** | (batch, seq_len, 86) landmark features |
| **Output** | (batch, num_classes) logits + confidence |
| **License** | MIT |

## Intended Use

- **Primary**: Real-time hand gesture recognition from webcam/camera
- **Secondary**: Sign language detection, accessibility interfaces, AR/VR input
- **Out of scope**: Full sign language translation, surgical hand tracking

## Architecture

```
Input (B, S, 86)
  → Linear Projection (86 → 128)
  → LayerNorm + GELU
  → Prepend CLS Token
  → Sinusoidal Positional Encoding
  → 4× TransformerEncoderLayer (d=128, heads=4, ff=256, pre-norm)
  → CLS Token Output
  → Classification Head (128 → 64 → num_classes)
  → Confidence Head (128 → 1, sigmoid)
```

## Training Data

- **Dataset**: To be trained on HaGRID, custom collected data
- **Classes**: open_palm, closed_fist, thumbs_up, thumbs_down, peace, pointing_up, ok_sign, pinch, wave
- **Augmentation**: Rotation ±15°, scale 0.85–1.15, translation ±0.05, Gaussian noise σ=0.005

## Evaluation

| Metric | Target | Actual |
|--------|--------|--------|
| Accuracy | > 95% | TBD |
| F1 (macro) | > 93% | TBD |
| Latency (CPU) | < 5ms | TBD |
| Model Size | < 5MB | ~2MB (FP32) |

## Limitations

- Requires well-lit conditions for MediaPipe detection
- Performance degrades with heavy occlusion
- Currently supports single-hand per gesture (multi-hand WIP)
- Not validated for medical/safety-critical applications

## Ethical Considerations

- **Privacy**: No images stored or transmitted. Landmark-only processing.
- **Bias**: Model should be evaluated across diverse hand shapes, skin tones, and motor abilities.
- **Accessibility**: Calibration system allows adaptation for users with motor disabilities.

## Environmental Impact

- Training: < 1 GPU-hour on consumer hardware
- Inference: CPU-only, < 5W power consumption
- No cloud infrastructure required
