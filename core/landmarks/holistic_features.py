"""Feature extraction for sign-language input (hands + upper-body pose).

The 86-dim hand feature vector in `core/landmarks/features.py` deliberately
discards absolute position: for "is this a fist or a peace sign" that is exactly
right. Sign language is the opposite case. *Where* a sign is made relative to the
body, and which hand makes it, are part of the meaning, so those must survive
into the features.

This extractor therefore keeps:
    - Per-hand shape features (reusing LandmarkFeatureExtractor)
    - Each hand's position relative to the body, scaled by shoulder span
    - Upper-body pose geometry (arms, shoulders, head)

Everything is expressed in units of shoulder span and measured from the shoulder
midpoint, which makes the representation invariant to camera distance and signer
size while preserving spatial meaning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from core.landmarks.features import LandmarkFeatureExtractor
from core.landmarks.normalizer import LandmarkNormalizer, NormalizationMode
from core.types import Handedness
from core.vision.holistic_detector import (
    LEFT_SHOULDER,
    NUM_FACE_KEY_POINTS,
    RIGHT_SHOULDER,
    UPPER_BODY_INDICES,
)

if TYPE_CHECKING:
    from core.vision.holistic_detector import HolisticResult

# Per-hand: 86 shape features + 3 position + 1 presence flag
HAND_BLOCK_DIM = 86 + 3 + 1
# Upper body: 9 landmarks x 3 coords, relative and scaled
POSE_BLOCK_DIM = len(UPPER_BODY_INDICES) * 3
# Face-mesh key points x 3 coords, relative and scaled
FACE_BLOCK_DIM = NUM_FACE_KEY_POINTS * 3
# Both hands + pose + a pose-presence flag + face + a face-presence flag
HOLISTIC_FEATURE_DIM = HAND_BLOCK_DIM * 2 + POSE_BLOCK_DIM + 1 + FACE_BLOCK_DIM + 1


class HolisticFeatureExtractor:
    """Build a fixed-width feature vector from a HolisticResult.

    The layout is fixed so missing parts (a hand out of frame, no body detected)
    become zeros plus a presence flag rather than a shorter vector. The model
    then learns to read the flags instead of being fed ragged input.

    Layout:
        [0:90]     left hand:  86 shape + 3 position + 1 presence
        [90:180]   right hand: 86 shape + 3 position + 1 presence
        [180:207]  upper-body pose: 9 landmarks x 3
        [207]      pose presence flag
        [208:253]  face key points: 15 landmarks x 3   (when track_face)
        [253]      face presence flag

    The face block is always allocated, even when face tracking is off, so the
    feature width is a property of the extractor rather than of whatever the
    detector happened to be configured with. A width that changed with runtime
    configuration would silently produce a model whose input shape depends on a
    flag set somewhere else entirely.

    Usage:
        >>> extractor = HolisticFeatureExtractor()
        >>> features = extractor.extract(holistic_result)
        >>> features.shape
        (254,)
    """

    def __init__(self, normalization_mode: NormalizationMode = NormalizationMode.FULL) -> None:
        self._hand_extractor = LandmarkFeatureExtractor()
        self._normalizer = LandmarkNormalizer(normalization_mode)

    @property
    def feature_dim(self) -> int:
        """Total feature width."""
        return HOLISTIC_FEATURE_DIM

    def extract(self, result: HolisticResult) -> np.ndarray:
        """Extract a fixed-width feature vector for one frame."""
        hand_dim = HAND_BLOCK_DIM
        features = np.zeros(self.feature_dim, dtype=np.float32)

        origin, scale = self._reference_frame(result)

        # ── Hands ────────────────────────────────────────────
        for hand in result.hands:
            slot = 0 if hand.handedness == Handedness.LEFT else 1
            base = slot * hand_dim

            shape = self._hand_extractor.extract(self._normalizer.normalize(hand))
            features[base : base + 86] = shape

            # Wrist position relative to the body, in shoulder-span units.
            wrist = hand.landmarks[0]
            features[base + 86 : base + 89] = (wrist - origin) / scale
            features[base + 89] = 1.0  # presence

        # ── Upper-body pose ──────────────────────────────────
        pose_base = hand_dim * 2
        if result.pose is not None:
            selected = result.pose[list(UPPER_BODY_INDICES)]
            features[pose_base : pose_base + POSE_BLOCK_DIM] = (
                (selected - origin) / scale
            ).flatten()
            features[pose_base + POSE_BLOCK_DIM] = 1.0  # pose presence

        # ── Face (non-manual markers) ────────────────────────
        # ASL marks yes/no questions with raised brows, wh-questions with
        # lowered brows, and negation and degree with the mouth. Those are
        # grammar, not decoration, so a word-level model that cannot see them
        # has a ceiling no amount of hand data lifts.
        face_base = pose_base + POSE_BLOCK_DIM + 1
        if result.face is not None:
            features[face_base : face_base + FACE_BLOCK_DIM] = (
                (result.face - origin) / scale
            ).flatten()
            features[face_base + FACE_BLOCK_DIM] = 1.0  # face presence

        return features

    def extract_sequence(self, results: list[HolisticResult]) -> np.ndarray:
        """Extract features for a temporal sequence of frames."""
        if not results:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.stack([self.extract(r) for r in results])

    def _reference_frame(self, result: HolisticResult) -> tuple[np.ndarray, float]:
        """Return the (origin, scale) used to make features body-relative.

        Origin is the shoulder midpoint and scale is the shoulder span. Without a
        detected body, the frame falls back to the image origin and unit scale so
        that hand-shape features still work.
        """
        if result.pose is None or result.shoulder_span <= 1e-6:
            return np.zeros(3, dtype=np.float32), 1.0

        origin = ((result.pose[LEFT_SHOULDER] + result.pose[RIGHT_SHOULDER]) / 2.0).astype(
            np.float32
        )
        return origin, float(result.shoulder_span)
