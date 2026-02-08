"""Landmark augmentation for training data diversity.

Applies geometric transformations to hand landmarks to improve
model generalization. All augmentations operate on landmark coordinates,
NOT on images — this is much faster and avoids pixel artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.types import HandLandmarks


@dataclass(slots=True)
class AugmentationConfig:
    """Augmentation parameters.

    Attributes:
        rotation_range_deg: Max rotation in degrees (±).
        scale_range: Scale factor range (min, max).
        translation_range: Max translation offset (±).
        noise_std: Gaussian noise standard deviation.
        flip_probability: Probability of horizontal flip.
        drop_landmark_probability: Probability of dropping each landmark (set to 0).
    """

    rotation_range_deg: float = 15.0
    scale_range: tuple[float, float] = (0.85, 1.15)
    translation_range: float = 0.05
    noise_std: float = 0.005
    flip_probability: float = 0.0
    drop_landmark_probability: float = 0.0


class LandmarkAugmentor:
    """Augment hand landmarks for training.

    All augmentations are applied in landmark space (not pixel space),
    making them extremely fast and resolution-independent.

    Usage:
        >>> augmentor = LandmarkAugmentor(AugmentationConfig(rotation_range_deg=20))
        >>> augmented = augmentor.augment(hand_landmarks)
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self._config = config or AugmentationConfig()
        self._rng = np.random.default_rng(seed)

    @property
    def config(self) -> AugmentationConfig:
        return self._config

    def augment(self, hand: HandLandmarks) -> HandLandmarks:
        """Apply random augmentations to landmarks.

        Args:
            hand: Input hand landmarks.

        Returns:
            New HandLandmarks with augmented coordinates.
        """
        lm = hand.landmarks.copy()

        # Random rotation (2D, preserving z)
        if self._config.rotation_range_deg > 0:
            lm = self._random_rotation(lm)

        # Random scale
        if self._config.scale_range != (1.0, 1.0):
            lm = self._random_scale(lm)

        # Random translation
        if self._config.translation_range > 0:
            lm = self._random_translation(lm)

        # Gaussian noise
        if self._config.noise_std > 0:
            lm = self._add_noise(lm)

        # Random horizontal flip
        if self._config.flip_probability > 0 and self._rng.random() < self._config.flip_probability:
            lm[:, 0] = -lm[:, 0]

        # Random landmark dropout
        if self._config.drop_landmark_probability > 0:
            lm = self._drop_landmarks(lm)

        return HandLandmarks(
            landmarks=lm.astype(np.float32),
            handedness=hand.handedness,
            confidence=hand.confidence,
        )

    def augment_batch(self, hands: list[HandLandmarks]) -> list[HandLandmarks]:
        """Augment a batch of landmarks."""
        return [self.augment(h) for h in hands]

    # ----- Private augmentations -----

    def _random_rotation(self, lm: np.ndarray) -> np.ndarray:
        angle_deg = self._rng.uniform(
            -self._config.rotation_range_deg,
            self._config.rotation_range_deg,
        )
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rot = np.array(
            [
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        result: np.ndarray = (rot @ lm.T).T
        return result

    def _random_scale(self, lm: np.ndarray) -> np.ndarray:
        scale = self._rng.uniform(*self._config.scale_range)
        return lm * scale

    def _random_translation(self, lm: np.ndarray) -> np.ndarray:
        offset = self._rng.uniform(
            -self._config.translation_range,
            self._config.translation_range,
            size=(1, 3),
        ).astype(np.float32)
        offset[0, 2] = 0  # Don't translate z
        return lm + offset

    def _add_noise(self, lm: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(0, self._config.noise_std, size=lm.shape).astype(np.float32)
        return lm + noise

    def _drop_landmarks(self, lm: np.ndarray) -> np.ndarray:
        mask = self._rng.random(lm.shape[0]) > self._config.drop_landmark_probability
        result = lm.copy()
        result[~mask] = 0.0
        return result
