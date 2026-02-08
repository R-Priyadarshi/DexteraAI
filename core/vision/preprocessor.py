"""Frame preprocessing utilities for the vision pipeline.

Handles resizing, normalization, color space conversion,
and validation of input frames before hand detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Configuration for frame preprocessing.

    Attributes:
        target_width: Target frame width (0 = preserve aspect ratio).
        target_height: Target frame height (0 = preserve aspect ratio).
        max_dimension: Maximum dimension; downscale if exceeded.
        normalize: Whether to normalize pixel values to [0, 1].
        flip_horizontal: Mirror the frame (useful for selfie mode).
    """
    target_width: int = 0
    target_height: int = 0
    max_dimension: int = 1280
    normalize: bool = False
    flip_horizontal: bool = True


class FramePreprocessor:
    """Production frame preprocessor.

    Handles all frame transformations before feeding to the detector.
    Stateless and thread-safe.

    Usage:
        >>> preprocessor = FramePreprocessor(PreprocessConfig(max_dimension=640))
        >>> processed = preprocessor.process(raw_frame)
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self._config = config or PreprocessConfig()

    @property
    def config(self) -> PreprocessConfig:
        return self._config

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to a BGR frame.

        Args:
            frame: Raw BGR frame (H, W, 3), dtype uint8.

        Returns:
            Preprocessed BGR frame.

        Raises:
            ValueError: If frame is invalid.
        """
        self._validate(frame)
        result = frame.copy()

        if self._config.flip_horizontal:
            result = cv2.flip(result, 1)

        result = self._resize(result)

        if self._config.normalize:
            result = result.astype(np.float32) / 255.0

        return result

    def _validate(self, frame: np.ndarray) -> None:
        """Validate input frame."""
        if frame is None:
            raise ValueError("Frame is None")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) BGR frame, got shape {frame.shape}")
        if frame.size == 0:
            raise ValueError("Frame is empty")

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame according to config."""
        h, w = frame.shape[:2]

        # Fixed target size
        if self._config.target_width > 0 and self._config.target_height > 0:
            return cv2.resize(
                frame,
                (self._config.target_width, self._config.target_height),
                interpolation=cv2.INTER_LINEAR,
            )

        # Max dimension constraint
        max_dim = max(h, w)
        if max_dim > self._config.max_dimension:
            scale = self._config.max_dimension / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return frame
