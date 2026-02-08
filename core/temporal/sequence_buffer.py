"""Temporal sequence buffer for real-time gesture recognition.

Maintains a sliding window of landmark features for continuous
gesture classification. Thread-safe and memory-efficient.
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from core.landmarks.features import LandmarkFeatureExtractor

if TYPE_CHECKING:
    from core.types import HandLandmarks


class SequenceBuffer:
    """Fixed-length sliding window buffer for landmark sequences.

    Maintains the last N frames of landmark features for
    feeding into the temporal gesture model.

    Thread-safe for use in real-time pipelines.

    Usage:
        >>> buffer = SequenceBuffer(max_len=30)
        >>> buffer.push(hand_landmarks)
        >>> if buffer.is_ready:
        ...     features = buffer.get_features()  # (30, feature_dim)
    """

    def __init__(
        self,
        max_len: int = 30,
        feature_extractor: LandmarkFeatureExtractor | None = None,
    ) -> None:
        """Initialize the sequence buffer.

        Args:
            max_len: Number of frames to keep in the buffer.
            feature_extractor: Feature extractor instance (creates default if None).
        """
        self._max_len = max_len
        self._extractor = feature_extractor or LandmarkFeatureExtractor()
        self._buffer: deque[np.ndarray] = deque(maxlen=max_len)
        self._lock = Lock()

    @property
    def max_len(self) -> int:
        return self._max_len

    @property
    def length(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True when buffer is full."""
        with self._lock:
            return len(self._buffer) == self._max_len

    @property
    def feature_dim(self) -> int:
        return self._extractor.feature_dim

    def push(self, hand: HandLandmarks) -> None:
        """Add a frame's landmarks to the buffer.

        Args:
            hand: Hand landmarks for the current frame.
        """
        features = self._extractor.extract(hand)
        with self._lock:
            self._buffer.append(features)

    def push_empty(self) -> None:
        """Push a zero frame (no hand detected)."""
        with self._lock:
            self._buffer.append(np.zeros(self._extractor.feature_dim, dtype=np.float32))

    def get_features(self) -> np.ndarray:
        """Get the buffered features as a 2D array.

        Returns:
            Array of shape (current_len, feature_dim).
        """
        with self._lock:
            if not self._buffer:
                return np.zeros((0, self._extractor.feature_dim), dtype=np.float32)
            return np.stack(list(self._buffer))

    def get_padded_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Get features padded to max_len with a mask.

        Returns:
            Tuple of:
                - features: (max_len, feature_dim) array
                - mask: (max_len,) boolean array (True = padded)
        """
        with self._lock:
            current = list(self._buffer)

        features = np.zeros((self._max_len, self._extractor.feature_dim), dtype=np.float32)
        mask = np.ones(self._max_len, dtype=bool)

        for i, feat in enumerate(current):
            features[i] = feat
            mask[i] = False

        return features, mask

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
