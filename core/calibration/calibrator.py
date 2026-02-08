"""Per-user gesture calibration.

Captures reference gestures from a user and builds a personalized
embedding space for improved recognition accuracy, especially
for users with motor disabilities or atypical hand shapes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from core.landmarks.features import LandmarkFeatureExtractor
from core.landmarks.normalizer import LandmarkNormalizer, NormalizationMode

if TYPE_CHECKING:
    from core.types import HandLandmarks


@dataclass
class CalibrationProfile:
    """Stored calibration data for a user.

    Attributes:
        user_id: Unique user identifier.
        gesture_references: Dict mapping gesture_name → list of feature vectors.
        created_at: ISO timestamp of creation.
        hand_scale: Estimated hand scale factor.
    """

    user_id: str
    gesture_references: dict[str, list[np.ndarray]] = field(default_factory=dict)
    created_at: str = ""
    hand_scale: float = 1.0


class UserCalibrator:
    """Manages per-user gesture calibration.

    Workflow:
        1. User performs each gesture N times during calibration
        2. System stores reference feature vectors
        3. During recognition, confidence is adjusted based on
           similarity to user's calibration data

    Usage:
        >>> calibrator = UserCalibrator(user_id="user_123")
        >>> calibrator.start_calibration()
        >>> calibrator.add_sample("thumbs_up", hand_landmarks)
        >>> calibrator.add_sample("thumbs_up", hand_landmarks_2)
        >>> calibrator.finish_calibration()
        >>> calibrator.save("profiles/user_123.json")
    """

    MIN_SAMPLES_PER_GESTURE = 3

    def __init__(
        self,
        user_id: str,
        normalizer: LandmarkNormalizer | None = None,
        feature_extractor: LandmarkFeatureExtractor | None = None,
    ) -> None:
        self._user_id = user_id
        self._normalizer = normalizer or LandmarkNormalizer(NormalizationMode.FULL)
        self._extractor = feature_extractor or LandmarkFeatureExtractor()
        self._profile = CalibrationProfile(user_id=user_id)
        self._is_calibrating = False

    @property
    def profile(self) -> CalibrationProfile:
        return self._profile

    @property
    def is_calibrating(self) -> bool:
        return self._is_calibrating

    def start_calibration(self) -> None:
        """Begin a calibration session."""
        self._profile.gesture_references.clear()
        self._is_calibrating = True
        logger.info(f"Calibration started for user: {self._user_id}")

    def add_sample(self, gesture_name: str, hand: HandLandmarks) -> int:
        """Add a calibration sample for a gesture.

        Args:
            gesture_name: Name of the gesture being calibrated.
            hand: Hand landmarks for this sample.

        Returns:
            Number of samples collected for this gesture.

        Raises:
            RuntimeError: If calibration hasn't been started.
        """
        if not self._is_calibrating:
            raise RuntimeError("Call start_calibration() first.")

        normalized = self._normalizer.normalize(hand)
        features = self._extractor.extract(normalized)

        if gesture_name not in self._profile.gesture_references:
            self._profile.gesture_references[gesture_name] = []

        self._profile.gesture_references[gesture_name].append(features)
        count = len(self._profile.gesture_references[gesture_name])
        logger.debug(f"Calibration sample {count} for '{gesture_name}'")
        return count

    def finish_calibration(self) -> CalibrationProfile:
        """Finish calibration and validate the profile.

        Returns:
            The completed calibration profile.

        Raises:
            ValueError: If any gesture has insufficient samples.
        """
        for name, samples in self._profile.gesture_references.items():
            if len(samples) < self.MIN_SAMPLES_PER_GESTURE:
                raise ValueError(
                    f"Gesture '{name}' has {len(samples)} samples, "
                    f"need at least {self.MIN_SAMPLES_PER_GESTURE}."
                )

        import datetime

        self._profile.created_at = datetime.datetime.now(datetime.UTC).isoformat()
        self._is_calibrating = False
        logger.info(
            f"Calibration complete for {self._user_id}: "
            f"{len(self._profile.gesture_references)} gestures"
        )
        return self._profile

    def compute_similarity(self, gesture_name: str, hand: HandLandmarks) -> float:
        """Compute similarity between a hand and calibrated reference.

        Args:
            gesture_name: Gesture to compare against.
            hand: Current hand landmarks.

        Returns:
            Cosine similarity score [0, 1].
        """
        if gesture_name not in self._profile.gesture_references:
            return 0.0

        normalized = self._normalizer.normalize(hand)
        features = self._extractor.extract(normalized)

        references = self._profile.gesture_references[gesture_name]
        mean_ref = np.mean(references, axis=0)

        # Cosine similarity
        dot = np.dot(features, mean_ref)
        norm_a = np.linalg.norm(features)
        norm_b = np.linalg.norm(mean_ref)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def save(self, path: str | Path) -> None:
        """Save calibration profile to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "user_id": self._profile.user_id,
            "created_at": self._profile.created_at,
            "hand_scale": self._profile.hand_scale,
            "gestures": {
                name: [arr.tolist() for arr in samples]
                for name, samples in self._profile.gesture_references.items()
            },
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Calibration profile saved: {path}")

    def load(self, path: str | Path) -> CalibrationProfile:
        """Load calibration profile from JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Profile not found: {path}")

        data = json.loads(path.read_text())
        self._profile = CalibrationProfile(
            user_id=data["user_id"],
            created_at=data.get("created_at", ""),
            hand_scale=data.get("hand_scale", 1.0),
            gesture_references={
                name: [np.array(arr, dtype=np.float32) for arr in samples]
                for name, samples in data["gestures"].items()
            },
        )
        logger.info(f"Calibration profile loaded: {path}")
        return self._profile
