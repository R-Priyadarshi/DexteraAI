"""Per-user gesture calibration.

Captures reference gestures from a user and builds a personalized
embedding space for improved recognition accuracy, especially
for users with motor disabilities or atypical hand shapes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from core.landmarks.features import LandmarkFeatureExtractor
from core.landmarks.normalizer import LandmarkNormalizer

if TYPE_CHECKING:
    from pathlib import Path

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

    Supports plugin/callback hooks, privacy controls, metrics, and robust error handling.

    Usage:
        >>> calibrator = UserCalibrator(user_id="user_123", callbacks=[...], privacy_mode=True)
        >>> calibrator.start_calibration()
        >>> calibrator.add_sample("thumbs_up", hand_landmarks)
        >>> calibrator.finish_calibration()
        >>> calibrator.save("profiles/user_123.json")
    """

    MIN_SAMPLES_PER_GESTURE = 3

    def __init__(
        self,
        user_id: str,
        callbacks: list[Any] | None = None,
        privacy_mode: bool = False,
        log: Any = logger,
    ) -> None:
        self.user_id = user_id
        self.profile = CalibrationProfile(user_id=user_id)
        self.callbacks = callbacks or []
        self.privacy_mode = privacy_mode
        self.log = log
        self.feature_extractor = LandmarkFeatureExtractor()
        self.normalizer = LandmarkNormalizer()
        self._calibrating = False
        self.metrics: dict[str, Any] = {}

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    def start_calibration(self) -> None:
        """Begin a calibration session."""
        self._calibrating = True
        self.profile.gesture_references.clear()
        self.profile.created_at = self._now()
        for cb in self.callbacks:
            cb.on_calibration_start(self)
        self.log.info(f"Calibration started for user: {self.user_id}")

    def add_sample(self, gesture_name: str, hand_landmarks: HandLandmarks) -> int:
        if not self._calibrating:
            self.log.error("Calibration not started.")
            for cb in self.callbacks:
                cb.on_calibration_error(self, "Calibration not started.")
            raise RuntimeError("start_calibration must be called before add_sample")
        try:
            if self.privacy_mode:
                hand_landmarks = self._mask_landmarks(hand_landmarks)
            normalized = self.normalizer.normalize(hand_landmarks)
            features = self.feature_extractor.extract(normalized)
            self.profile.gesture_references.setdefault(gesture_name, []).append(features)
            for cb in self.callbacks:
                cb.on_sample_added(self, gesture_name, features)
            self.log.info(f"Sample added for gesture: {gesture_name}")
            return len(self.profile.gesture_references[gesture_name])
        except Exception as e:
            self.log.error(f"Calibration sample error: {e}")
            for cb in self.callbacks:
                cb.on_calibration_error(self, e)
            raise

    def finish_calibration(self) -> CalibrationProfile:
        for gesture, refs in self.profile.gesture_references.items():
            if len(refs) < self.MIN_SAMPLES_PER_GESTURE:
                raise ValueError(
                    f"need at least {self.MIN_SAMPLES_PER_GESTURE} samples for gesture '{gesture}'"
                )
        self._calibrating = False
        self.profile.hand_scale = self._estimate_hand_scale()
        for cb in self.callbacks:
            cb.on_calibration_end(self)
        self.log.info(f"Calibration finished for user: {self.user_id}")
        return self.profile

    def compute_similarity(self, gesture_name: str, hand_landmarks: HandLandmarks) -> float:
        """Compute similarity between input landmarks and calibrated reference."""
        if gesture_name not in self.profile.gesture_references:
            raise ValueError(f"Gesture '{gesture_name}' not calibrated")
        normalized = self.normalizer.normalize(hand_landmarks)
        features = self.feature_extractor.extract(normalized)
        refs = self.profile.gesture_references[gesture_name]
        # Example: cosine similarity
        sims = [self._cosine_similarity(features, ref) for ref in refs]
        return float(np.mean(sims)) if sims else 0.0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def load(self, path: str | Path) -> CalibrationProfile | None:
        try:
            with open(path) as f:
                data = json.load(f)
            self._dict_to_profile(data)
            self.log.info(f"Calibration profile loaded: {path}")
            for cb in self.callbacks:
                cb.on_profile_loaded(self, path)
            return self.profile
        except FileNotFoundError as e:
            self.log.error(f"Calibration load error: {e}")
            for cb in self.callbacks:
                cb.on_calibration_error(self, e)
            raise
        except Exception as e:
            self.log.error(f"Calibration load error: {e}")
            for cb in self.callbacks:
                cb.on_calibration_error(self, e)
            return None

    def save(self, path: str | Path) -> None:
        try:
            with open(path, "w") as f:
                json.dump(self._profile_to_dict(), f, indent=2)
            self.log.info(f"Calibration profile saved: {path}")
            for cb in self.callbacks:
                cb.on_profile_saved(self, path)
        except Exception as e:
            self.log.error(f"Calibration save error: {e}")
            for cb in self.callbacks:
                cb.on_calibration_error(self, e)

    def _mask_landmarks(self, hand_landmarks: HandLandmarks) -> HandLandmarks:
        """Privacy: return landmarks with coordinates zeroed.

        Previously this returned ``np.zeros_like(hand_landmarks)``, which passed a
        dataclass to numpy and produced an ndarray, so every downstream call
        failed as soon as privacy_mode was enabled.
        """
        from dataclasses import replace

        return replace(hand_landmarks, landmarks=np.zeros_like(hand_landmarks.landmarks))

    def _estimate_hand_scale(self) -> float:
        # Example: estimate hand scale from reference samples
        scales = []
        for refs in self.profile.gesture_references.values():
            for f in refs:
                scales.append(np.linalg.norm(f))
        return float(np.median(scales)) if scales else 1.0

    def _now(self) -> str:
        import datetime
        return datetime.datetime.now(datetime.UTC).isoformat()

    def _profile_to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.profile.user_id,
            "gesture_references": {
                k: [f.tolist() for f in v] for k, v in self.profile.gesture_references.items()
            },
            "created_at": self.profile.created_at,
            "hand_scale": self.profile.hand_scale,
        }

    def _dict_to_profile(self, data: dict[str, Any]) -> None:
        self.profile.user_id = data.get("user_id", "")
        self.profile.gesture_references = {
            k: [np.array(f) for f in v] for k, v in data.get("gesture_references", {}).items()
        }
        self.profile.created_at = data.get("created_at", "")
        self.profile.hand_scale = data.get("hand_scale", 1.0)

    def enable_privacy(self) -> None:
        self.privacy_mode = True
        self.log.info("Privacy mode enabled for calibration.")

    def disable_privacy(self) -> None:
        self.privacy_mode = False
        self.log.info("Privacy mode disabled for calibration.")
