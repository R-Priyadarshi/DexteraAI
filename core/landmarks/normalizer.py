"""Landmark normalization for translation, scale, and rotation invariance.

All normalization is purely geometric — no ML involved.
This ensures landmarks are comparable across different hand positions,
distances from camera, and orientations.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from core.types import HandLandmarks


class NormalizationMode(Enum):
    """Available normalization strategies."""

    WRIST_CENTERED = auto()  # Translate wrist to origin
    PALM_CENTERED = auto()  # Translate palm center to origin
    SCALE_INVARIANT = auto()  # + normalize to unit scale
    ROTATION_INVARIANT = auto()  # + align wrist-middle finger axis
    FULL = auto()  # All of the above


class LandmarkNormalizer:
    """Normalize hand landmarks for model-ready representation.

    Supports multiple normalization modes for different use cases.
    All operations are vectorized with numpy for performance.

    Usage:
        >>> normalizer = LandmarkNormalizer(mode=NormalizationMode.FULL)
        >>> normalized = normalizer.normalize(hand_landmarks)
    """

    # Landmark indices
    WRIST = 0
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17

    def __init__(self, mode: NormalizationMode = NormalizationMode.FULL) -> None:
        self._mode = mode

    @property
    def mode(self) -> NormalizationMode:
        return self._mode

    def normalize(self, hand: HandLandmarks) -> HandLandmarks:
        """Normalize landmarks according to the configured mode.

        Args:
            hand: Raw hand landmarks.

        Returns:
            New HandLandmarks with normalized coordinates.
        """
        lm = hand.landmarks.copy()

        if self._mode in (
            NormalizationMode.WRIST_CENTERED,
            NormalizationMode.SCALE_INVARIANT,
            NormalizationMode.ROTATION_INVARIANT,
            NormalizationMode.FULL,
        ):
            lm = self._center_on_wrist(lm)

        if self._mode == NormalizationMode.PALM_CENTERED:
            lm = self._center_on_palm(lm)

        if self._mode in (
            NormalizationMode.SCALE_INVARIANT,
            NormalizationMode.ROTATION_INVARIANT,
            NormalizationMode.FULL,
        ):
            lm = self._normalize_scale(lm)

        if self._mode in (
            NormalizationMode.ROTATION_INVARIANT,
            NormalizationMode.FULL,
        ):
            lm = self._align_rotation(lm)

        return HandLandmarks(
            landmarks=lm,
            handedness=hand.handedness,
            confidence=hand.confidence,
        )

    def normalize_batch(self, hands: list[HandLandmarks]) -> list[HandLandmarks]:
        """Normalize a batch of hand landmarks."""
        return [self.normalize(h) for h in hands]

    # ----- Private transforms -----

    def _center_on_wrist(self, lm: np.ndarray) -> np.ndarray:
        """Translate so wrist is at origin."""
        result: np.ndarray = lm - lm[self.WRIST]
        return result

    def _center_on_palm(self, lm: np.ndarray) -> np.ndarray:
        """Translate so palm center is at origin."""
        palm_indices = [self.WRIST, self.INDEX_MCP, self.MIDDLE_MCP, self.RING_MCP, self.PINKY_MCP]
        palm_center = lm[palm_indices].mean(axis=0)
        result: np.ndarray = lm - palm_center
        return result

    def _normalize_scale(self, lm: np.ndarray) -> np.ndarray:
        """Scale landmarks to unit bounding box."""
        max_dist = np.max(np.linalg.norm(lm - lm[self.WRIST], axis=1))
        if max_dist < 1e-6:
            return lm
        result: np.ndarray = lm / max_dist
        return result

    def _align_rotation(self, lm: np.ndarray) -> np.ndarray:
        """Rotate so wrist→middle_mcp axis aligns with Y-axis (2D plane)."""
        direction = lm[self.MIDDLE_MCP, :2] - lm[self.WRIST, :2]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return lm

        direction = direction / norm
        # Rotation matrix to align direction with [0, 1] (Y-up)
        cos_a = direction[1]
        sin_a = direction[0]
        rot = np.array(
            [
                [cos_a, sin_a, 0],
                [-sin_a, cos_a, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        result: np.ndarray = (rot @ lm.T).T
        return result
