"""Tests for core.landmarks — normalizer, augmentor, feature extractor."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from core.landmarks.augmentor import AugmentationConfig, LandmarkAugmentor
from core.landmarks.features import LandmarkFeatureExtractor
from core.landmarks.normalizer import LandmarkNormalizer, NormalizationMode
from core.types import LANDMARK_DIMS, NUM_HAND_LANDMARKS, HandLandmarks


class TestLandmarkNormalizer:
    """Tests for LandmarkNormalizer."""

    def test_wrist_centered(self, random_landmarks: HandLandmarks) -> None:
        normalizer = LandmarkNormalizer(NormalizationMode.WRIST_CENTERED)
        result = normalizer.normalize(random_landmarks)
        # Wrist should be at origin
        np.testing.assert_allclose(result.landmarks[0], [0, 0, 0], atol=1e-6)

    def test_scale_invariant(self, random_landmarks: HandLandmarks) -> None:
        normalizer = LandmarkNormalizer(NormalizationMode.SCALE_INVARIANT)
        result = normalizer.normalize(random_landmarks)
        # Max distance from wrist should be <= 1.0
        distances = np.linalg.norm(result.landmarks - result.landmarks[0], axis=1)
        assert np.max(distances) <= 1.0 + 1e-6

    def test_full_normalization(self, random_landmarks: HandLandmarks) -> None:
        normalizer = LandmarkNormalizer(NormalizationMode.FULL)
        result = normalizer.normalize(random_landmarks)
        assert result.landmarks.shape == (21, 3)
        assert result.handedness == random_landmarks.handedness

    def test_preserves_handedness(self, random_landmarks: HandLandmarks) -> None:
        normalizer = LandmarkNormalizer()
        result = normalizer.normalize(random_landmarks)
        assert result.handedness == random_landmarks.handedness
        assert result.confidence == random_landmarks.confidence

    def test_batch_normalize(self, batch_landmarks: list[HandLandmarks]) -> None:
        normalizer = LandmarkNormalizer()
        results = normalizer.normalize_batch(batch_landmarks[:5])
        assert len(results) == 5

    @given(
        arrays(
            dtype=np.float32,
            shape=(NUM_HAND_LANDMARKS, LANDMARK_DIMS),
            elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=30)
    def test_output_shape_preserved(self, landmarks: np.ndarray) -> None:
        hand = HandLandmarks(landmarks=landmarks)
        normalizer = LandmarkNormalizer(NormalizationMode.FULL)
        result = normalizer.normalize(hand)
        assert result.landmarks.shape == (NUM_HAND_LANDMARKS, LANDMARK_DIMS)


class TestLandmarkAugmentor:
    """Tests for LandmarkAugmentor."""

    def test_default_augmentation(self, random_landmarks: HandLandmarks) -> None:
        augmentor = LandmarkAugmentor(seed=42)
        result = augmentor.augment(random_landmarks)
        assert result.landmarks.shape == (21, 3)
        # Should be different from original
        assert not np.allclose(result.landmarks, random_landmarks.landmarks)

    def test_no_augmentation(self, random_landmarks: HandLandmarks) -> None:
        config = AugmentationConfig(
            rotation_range_deg=0,
            scale_range=(1.0, 1.0),
            translation_range=0,
            noise_std=0,
            flip_probability=0,
        )
        augmentor = LandmarkAugmentor(config, seed=42)
        result = augmentor.augment(random_landmarks)
        np.testing.assert_allclose(result.landmarks, random_landmarks.landmarks, atol=1e-6)

    def test_reproducibility(self, random_landmarks: HandLandmarks) -> None:
        aug1 = LandmarkAugmentor(seed=123)
        aug2 = LandmarkAugmentor(seed=123)
        r1 = aug1.augment(random_landmarks)
        r2 = aug2.augment(random_landmarks)
        np.testing.assert_array_equal(r1.landmarks, r2.landmarks)

    def test_batch_augmentation(self, batch_landmarks: list[HandLandmarks]) -> None:
        augmentor = LandmarkAugmentor(seed=42)
        results = augmentor.augment_batch(batch_landmarks[:5])
        assert len(results) == 5


class TestLandmarkFeatureExtractor:
    """Tests for LandmarkFeatureExtractor."""

    def test_feature_dim(self) -> None:
        extractor = LandmarkFeatureExtractor()
        assert extractor.feature_dim == 86  # 63 + 5 + 10 + 5 + 3

    def test_extract(self, random_landmarks: HandLandmarks) -> None:
        extractor = LandmarkFeatureExtractor()
        features = extractor.extract(random_landmarks)
        assert features.shape == (86,)
        assert features.dtype == np.float32

    def test_extract_sequence(self, batch_landmarks: list[HandLandmarks]) -> None:
        extractor = LandmarkFeatureExtractor()
        seq_features = extractor.extract_sequence(batch_landmarks)
        assert seq_features.shape == (30, 86)

    def test_empty_sequence(self) -> None:
        extractor = LandmarkFeatureExtractor()
        seq_features = extractor.extract_sequence([])
        assert seq_features.shape == (0, 86)

    def test_finger_curl_range(self, random_landmarks: HandLandmarks) -> None:
        extractor = LandmarkFeatureExtractor()
        features = extractor.extract(random_landmarks)
        # Curl ratios are features[78:83] (after 63+5+10)
        curl_ratios = features[78:83]
        assert np.all(curl_ratios >= 0.0)
        assert np.all(curl_ratios <= 1.0)

    def test_palm_normal_unit(self, random_landmarks: HandLandmarks) -> None:
        extractor = LandmarkFeatureExtractor()
        features = extractor.extract(random_landmarks)
        palm_normal = features[83:86]
        norm = np.linalg.norm(palm_normal)
        # Should be unit vector or zero
        assert norm < 1.0 + 1e-5 or norm < 1e-6
