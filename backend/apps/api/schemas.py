# ============================================================
#  Dextera AI — Pydantic API Schemas
# ============================================================
"""Dextera AI — Pydantic API Schemas.

Updated to use core.types data structures.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Prediction ───────────────────────────────────────────────


class PredictionResult(BaseModel):
    gesture_name: str = Field(..., description="Recognized gesture label")
    gesture_id: int = Field(..., ge=-1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    success: bool = True
    predictions: list[PredictionResult] = Field(default_factory=list)
    num_hands: int = 0
    inference_ms: float = 0.0
    privacy_mode: str = "on-device"


# ── Health ───────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    pipeline_running: bool
    uptime_seconds: float
    privacy: str = "all-inference-on-device"
