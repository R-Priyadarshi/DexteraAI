"""Tests for core.types — data structures and protocols."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from core.types import (
    LANDMARK_DIMS,
    NUM_HAND_LANDMARKS,
    FrameResult,
    GestureResult,
    GestureState,
    Handedness,
    HandLandmarks,
)


class TestHandLandmarks:
    """Tests for HandLandmarks dataclass."""

    def test_valid_creation(self) -> None:
        landmarks = np.zeros((21, 3), dtype=np.float32)
        hand = HandLandmarks(landmarks=landmarks, handedness=Handedness.RIGHT, confidence=0.9)
        assert hand.landmarks.shape == (21, 3)
        assert hand.handedness == Handedness.RIGHT
        assert hand.confidence == 0.9

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(AssertionError):
            HandLandmarks(landmarks=np.zeros((10, 3), dtype=np.float32))

    def test_invalid_dims_raises(self) -> None:
        with pytest.raises(AssertionError):
            HandLandmarks(landmarks=np.zeros((21, 2), dtype=np.float32))

    @given(
        arrays(
            dtype=np.float32,
            shape=(NUM_HAND_LANDMARKS, LANDMARK_DIMS),
            elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=50)
    def test_any_valid_landmarks(self, landmarks: np.ndarray) -> None:
        hand = HandLandmarks(landmarks=landmarks)
        assert hand.landmarks.shape == (NUM_HAND_LANDMARKS, LANDMARK_DIMS)

    def test_frozen(self) -> None:
        landmarks = np.zeros((21, 3), dtype=np.float32)
        hand = HandLandmarks(landmarks=landmarks)
        with pytest.raises(AttributeError):
            hand.confidence = 0.5  # type: ignore[misc]


class TestGestureResult:
    """Tests for GestureResult dataclass."""

    def test_creation(self) -> None:
        result = GestureResult(
            gesture_id=1,
            gesture_name="thumbs_up",
            confidence=0.95,
        )
        assert result.gesture_id == 1
        assert result.gesture_name == "thumbs_up"
        assert result.state == GestureState.RECOGNIZED

    def test_with_landmarks(self, random_landmarks: HandLandmarks) -> None:
        result = GestureResult(
            gesture_id=0,
            gesture_name="open_palm",
            confidence=0.8,
            landmarks=random_landmarks,
        )
        assert result.landmarks is not None
        assert result.landmarks.landmarks.shape == (21, 3)


class TestFrameResult:
    """Tests for FrameResult dataclass."""

    def test_empty_frame(self) -> None:
        result = FrameResult()
        assert len(result.hands) == 0
        assert len(result.gestures) == 0
        assert result.inference_time_ms == 0.0

    def test_frame_with_hands(self, random_landmarks: HandLandmarks) -> None:
        result = FrameResult(
            hands=[random_landmarks],
            gestures=[],
            timestamp_ms=1000.0,
            inference_time_ms=5.0,
        )
        assert len(result.hands) == 1
        assert result.inference_time_ms == 5.0


class TestHandedness:
    """Tests for Handedness enum."""

    def test_values(self) -> None:
        assert Handedness.LEFT.value == "left"
        assert Handedness.RIGHT.value == "right"
        assert Handedness.UNKNOWN.value == "unknown"

    def test_all_variants(self) -> None:
        assert len(Handedness) == 3
