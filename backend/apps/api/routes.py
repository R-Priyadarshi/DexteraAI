"""Dextera AI — API Routes.

REST + WebSocket endpoints for gesture inference.
All inference uses the core/ pipeline modules.
"""

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from loguru import logger

from backend.apps.api.dependencies import get_settings
from backend.apps.api.schemas import HealthResponse, PredictionResponse, PredictionResult
from core.inference.pipeline import GesturePipeline, PipelineConfig

if TYPE_CHECKING:
    from backend.config import Settings

router = APIRouter()

# Lazy-loaded pipeline singleton
_pipeline: GesturePipeline | None = None


def _get_pipeline() -> GesturePipeline:
    """Get or create the global gesture pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = GesturePipeline(PipelineConfig())
        _pipeline.start()
    return _pipeline


# ── Health ───────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Liveness / readiness probe."""
    from backend.apps.api.main import get_uptime

    pipeline = _get_pipeline()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        pipeline_running=pipeline.is_running,
        uptime_seconds=round(get_uptime(), 2),
    )


# ── Prediction ───────────────────────────────────────────────


@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG/PNG)"),
) -> PredictionResponse:
    """Predict hand gesture from an uploaded image."""
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {file.content_type}",
        )

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr_image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image",
        )

    try:
        pipeline = _get_pipeline()
        result = pipeline.process_frame(bgr_image)
    except Exception as e:
        logger.error("Prediction failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference error",
        ) from None

    predictions = [
        PredictionResult(
            gesture_name=g.gesture_name,
            gesture_id=g.gesture_id,
            confidence=g.confidence,
        )
        for g in result.gestures
    ]

    return PredictionResponse(
        success=True,
        predictions=predictions,
        num_hands=len(result.hands),
        inference_ms=round(result.inference_time_ms, 2),
    )


# ── WebSocket Real-Time Stream ───────────────────────────────


@router.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket) -> None:
    """Real-time gesture prediction over WebSocket.

    Protocol:
      Client → Server: base64-encoded JPEG frame
      Server → Client: JSON prediction result
    """
    await ws.accept()
    pipeline = _get_pipeline()
    logger.info("WebSocket client connected")
    frame_count = 0

    try:
        while True:
            data = await ws.receive_text()
            frame_count += 1

            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                await ws.send_json({"error": "Invalid image data"})
                continue

            if bgr_image is None:
                await ws.send_json({"error": "Failed to decode frame"})
                continue

            t0 = time.perf_counter()
            result = pipeline.process_frame(bgr_image)
            latency = (time.perf_counter() - t0) * 1000

            await ws.send_json(
                {
                    "frame_id": frame_count,
                    "gestures": [
                        {
                            "gesture_name": g.gesture_name,
                            "gesture_id": g.gesture_id,
                            "confidence": round(g.confidence, 4),
                        }
                        for g in result.gestures
                    ],
                    "num_hands": len(result.hands),
                    "inference_ms": round(latency, 2),
                    "privacy_mode": "on-device",
                }
            )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected | frames={}", frame_count)
    except Exception as e:
        logger.error("WebSocket error: {}", e)
        await ws.close(code=1011, reason="Internal error")
