"""Tests for the sign-language holistic path (hands + upper-body pose)."""

from __future__ import annotations

import numpy as np
import pytest

from core.landmarks.holistic_features import (
    HAND_BLOCK_DIM,
    POSE_BLOCK_DIM,
    HolisticFeatureExtractor,
)
from core.types import Handedness, HandLandmarks
from core.vision.holistic_detector import (
    LEFT_SHOULDER,
    NUM_POSE_LANDMARKS,
    RIGHT_SHOULDER,
    HolisticResult,
    resolve_pose_landmarker_path,
)


def _hand(handedness: Handedness, wrist_xyz: tuple[float, float, float]) -> HandLandmarks:
    """A hand whose wrist sits at a known position."""
    rng = np.random.default_rng(0)
    landmarks = rng.random((21, 3)).astype(np.float32) * 0.1
    landmarks[0] = np.array(wrist_xyz, dtype=np.float32)
    return HandLandmarks(landmarks=landmarks, handedness=handedness, confidence=0.9)


def _pose(shoulder_span: float = 0.4) -> np.ndarray:
    """An upright body with shoulders a known distance apart."""
    pose = np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float32)
    pose[LEFT_SHOULDER] = [0.5 - shoulder_span / 2, 0.4, 0.0]
    pose[RIGHT_SHOULDER] = [0.5 + shoulder_span / 2, 0.4, 0.0]
    pose[0] = [0.5, 0.2, 0.0]  # nose
    return pose


class TestHolisticFeatures:
    def test_feature_dim_layout(self) -> None:
        fx = HolisticFeatureExtractor()
        assert fx.feature_dim == HAND_BLOCK_DIM * 2 + POSE_BLOCK_DIM + 1
        assert fx.feature_dim == 208

    def test_empty_result_is_all_zeros(self) -> None:
        fx = HolisticFeatureExtractor()
        features = fx.extract(HolisticResult())
        assert features.shape == (fx.feature_dim,)
        assert not features.any()

    def test_presence_flags_mark_which_hand_is_present(self) -> None:
        fx = HolisticFeatureExtractor()

        left_only = fx.extract(HolisticResult(hands=[_hand(Handedness.LEFT, (0.4, 0.5, 0.0))]))
        right_only = fx.extract(HolisticResult(hands=[_hand(Handedness.RIGHT, (0.6, 0.5, 0.0))]))

        # Presence flag is the last element of each hand block.
        assert left_only[HAND_BLOCK_DIM - 1] == 1.0
        assert left_only[HAND_BLOCK_DIM * 2 - 1] == 0.0
        assert right_only[HAND_BLOCK_DIM - 1] == 0.0
        assert right_only[HAND_BLOCK_DIM * 2 - 1] == 1.0

    def test_hands_land_in_separate_slots(self) -> None:
        fx = HolisticFeatureExtractor()
        both = fx.extract(
            HolisticResult(
                hands=[
                    _hand(Handedness.LEFT, (0.4, 0.5, 0.0)),
                    _hand(Handedness.RIGHT, (0.6, 0.5, 0.0)),
                ]
            )
        )
        assert both[HAND_BLOCK_DIM - 1] == 1.0
        assert both[HAND_BLOCK_DIM * 2 - 1] == 1.0

    def test_pose_presence_flag(self) -> None:
        fx = HolisticFeatureExtractor()
        without = fx.extract(HolisticResult())
        with_pose = fx.extract(HolisticResult(pose=_pose(), shoulder_span=0.4))

        assert without[-1] == 0.0
        assert with_pose[-1] == 1.0

    def test_position_is_relative_to_shoulder_midpoint(self) -> None:
        """A hand at the shoulder midpoint must encode as position zero."""
        fx = HolisticFeatureExtractor()
        pose = _pose(shoulder_span=0.4)
        midpoint = (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER]) / 2.0

        result = HolisticResult(
            hands=[_hand(Handedness.LEFT, tuple(midpoint))],
            pose=pose,
            shoulder_span=0.4,
        )
        features = fx.extract(result)
        position = features[86:89]  # left hand position block
        np.testing.assert_allclose(position, np.zeros(3), atol=1e-6)

    def test_position_is_scale_invariant(self) -> None:
        """The same gesture, nearer or further from the camera, encodes the same."""
        fx = HolisticFeatureExtractor()

        def encode(span: float) -> np.ndarray:
            pose = _pose(shoulder_span=span)
            midpoint = (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER]) / 2.0
            # Hand held one full shoulder-span to the left of centre.
            wrist = midpoint + np.array([-span, 0.0, 0.0], dtype=np.float32)
            return fx.extract(
                HolisticResult(
                    hands=[_hand(Handedness.LEFT, tuple(wrist))],
                    pose=pose,
                    shoulder_span=span,
                )
            )[86:89]

        near = encode(0.6)
        far = encode(0.3)
        np.testing.assert_allclose(near, far, atol=1e-5)
        assert near[0] == pytest.approx(-1.0, abs=1e-5)

    def test_sequence_extraction_shape(self) -> None:
        fx = HolisticFeatureExtractor()
        frames = [HolisticResult(pose=_pose(), shoulder_span=0.4) for _ in range(7)]
        assert fx.extract_sequence(frames).shape == (7, fx.feature_dim)
        assert fx.extract_sequence([]).shape == (0, fx.feature_dim)


class TestPoseBundleResolution:
    def test_explicit_bad_path_raises_rather_than_falling_back(self, tmp_path) -> None:
        """An explicit path must not silently resolve to a different bundle."""
        with pytest.raises(FileNotFoundError, match="not found at the given path"):
            resolve_pose_landmarker_path(tmp_path / "absent.task")

    def test_no_bundle_anywhere_explains_the_fix(self, tmp_path, monkeypatch) -> None:
        """With nothing installed, the error tells the user how to fix it."""
        monkeypatch.delenv("DEXTERA_POSE_LANDMARKER", raising=False)
        monkeypatch.setattr(
            "core.vision.holistic_detector.DEFAULT_POSE_MODEL_PATHS",
            (str(tmp_path / "nope.task"),),
        )
        with pytest.raises(FileNotFoundError, match="make fetch-models"):
            resolve_pose_landmarker_path()

    def test_explicit_valid_path_is_used(self, tmp_path) -> None:
        bundle = tmp_path / "custom.task"
        bundle.write_bytes(b"stub")
        assert resolve_pose_landmarker_path(bundle) == bundle
