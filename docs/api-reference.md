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
