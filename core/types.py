"""Shared types, protocols, and constants for DexteraAI core."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_HAND_LANDMARKS = 21
LANDMARK_DIMS = 3  # x, y, z
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Handedness(Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class GestureState(Enum):
    """Lifecycle of a gesture detection."""
    IDLE = auto()
    DETECTING = auto()
    RECOGNIZED = auto()
    HELD = auto()
    RELEASED = auto()


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """Normalized 3D hand landmarks for a single hand.

    Attributes:
        landmarks: (21, 3) array of [x, y, z] in normalized coordinates.
        handedness: Left or right hand.
        confidence: Detection confidence [0, 1].
    """
    landmarks: NDArray[np.float32]  # shape (21, 3)
    handedness: Handedness = Handedness.UNKNOWN
    confidence: float = 0.0

    def __post_init__(self) -> None:
        assert self.landmarks.shape == (NUM_HAND_LANDMARKS, LANDMARK_DIMS), (
            f"Expected shape ({NUM_HAND_LANDMARKS}, {LANDMARK_DIMS}), "
            f"got {self.landmarks.shape}"
        )


@dataclass(frozen=True, slots=True)
class GestureResult:
    """Result of gesture classification.

    Attributes:
        gesture_id: Integer class ID (-1 = unknown).
        gesture_name: Human-readable gesture label.
        confidence: Classification confidence [0, 1].
        state: Current gesture lifecycle state.
        landmarks: Source hand landmarks (optional).
    """
    gesture_id: int
    gesture_name: str
    confidence: float
    state: GestureState = GestureState.RECOGNIZED
    landmarks: HandLandmarks | None = None


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Complete result for a single frame.

    Attributes:
        hands: Detected hands with landmarks.
        gestures: Recognized gestures.
        timestamp_ms: Frame timestamp in milliseconds.
        inference_time_ms: Total inference latency in milliseconds.
    """
    hands: list[HandLandmarks] = field(default_factory=list)
    gestures: list[GestureResult] = field(default_factory=list)
    timestamp_ms: float = 0.0
    inference_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Protocols (Interfaces)
# ---------------------------------------------------------------------------

class HandDetector(Protocol):
    """Protocol for hand landmark detection backends."""

    def detect(self, frame: NDArray[np.uint8]) -> list[HandLandmarks]:
        """Detect hand landmarks in a BGR frame."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class GestureClassifier(Protocol):
    """Protocol for gesture classification backends."""

    def classify(
        self,
        landmarks_sequence: Sequence[HandLandmarks],
    ) -> list[GestureResult]:
        """Classify a sequence of landmark frames into gestures."""
        ...

    @property
    def gesture_labels(self) -> list[str]:
        """Return ordered list of gesture class names."""
        ...


class InferenceRuntime(Protocol):
    """Protocol for cross-platform inference runtimes (ONNX, TFLite)."""

    def load(self, model_path: str) -> None:
        """Load a model from disk."""
        ...

    def predict(self, inputs: dict[str, NDArray[np.float32]]) -> dict[str, NDArray[np.float32]]:
        """Run inference."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...
