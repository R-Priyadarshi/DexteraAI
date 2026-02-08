"""Shared test fixtures for DexteraAI."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core.types import HandLandmarks, Handedness, NUM_HAND_LANDMARKS, LANDMARK_DIMS


@pytest.fixture
def random_landmarks() -> HandLandmarks:
    """Generate random hand landmarks."""
    rng = np.random.default_rng(42)
    return HandLandmarks(
        landmarks=rng.random((NUM_HAND_LANDMARKS, LANDMARK_DIMS)).astype(np.float32),
        handedness=Handedness.RIGHT,
        confidence=0.95,
    )


@pytest.fixture
def batch_landmarks() -> list[HandLandmarks]:
    """Generate a batch of random hand landmarks (sequence of 30 frames)."""
    rng = np.random.default_rng(42)
    return [
        HandLandmarks(
            landmarks=rng.random((NUM_HAND_LANDMARKS, LANDMARK_DIMS)).astype(np.float32),
            handedness=Handedness.RIGHT,
            confidence=0.9,
        )
        for _ in range(30)
    ]


@pytest.fixture
def dummy_bgr_frame() -> np.ndarray:
    """Generate a dummy 480x640 BGR frame."""
    return np.random.default_rng(42).integers(
        0, 256, (480, 640, 3), dtype=np.uint8
    )


@pytest.fixture
def gesture_model() -> torch.nn.Module:
    """Create a small GestureTransformer for testing."""
    from core.temporal.model import GestureTransformer

    return GestureTransformer(
        input_dim=86,
        num_classes=10,
        d_model=32,
        nhead=2,
        num_layers=2,
        dim_feedforward=64,
        max_seq_len=30,
        dropout=0.0,
    )
