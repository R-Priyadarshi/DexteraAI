"""Dextera AI — Centralised Settings (Pydantic v2).

Single source of truth for all configuration.
Loads from .env, environment variables, or defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────
    app_name: str = "Dextera AI"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # ── Server ───────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = ["*"]

    # ── Model paths ──────────────────────────────────────────
    onnx_model_path: str = "models/gesture_classifier.onnx"
    pytorch_model_path: str = "models/gesture_classifier.pt"
    dataset_path: str = "data/processed/dataset.joblib"
    training_data_dir: str = "data/raw"

    # ── Inference ────────────────────────────────────────────
    confidence_threshold: float = 0.70
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5

    # ── MLflow ───────────────────────────────────────────────
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "dextera-ai"

    # ── Camera ───────────────────────────────────────────────
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    # ── Privacy ──────────────────────────────────────────────
    enable_cloud_inference: bool = False
    log_images: bool = False

    # ── Gesture model ────────────────────────────────────────
    num_landmarks: int = 21
    landmark_dim: int = 3
    sequence_length: int = 30
    num_gesture_classes: int = 10

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return json.loads(v)
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def feature_dim(self) -> int:
        """86-dim features from LandmarkFeatureExtractor."""
        return 86


settings = Settings()
