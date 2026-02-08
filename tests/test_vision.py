"""Tests for core.vision — preprocessor (detector requires camera/MediaPipe)."""

from __future__ import annotations

import numpy as np
import pytest

from core.vision.preprocessor import FramePreprocessor, PreprocessConfig


class TestFramePreprocessor:
    """Tests for FramePreprocessor."""

    def test_default_config(self, dummy_bgr_frame: np.ndarray) -> None:
        preprocessor = FramePreprocessor()
        result = preprocessor.process(dummy_bgr_frame)
        assert result.shape[2] == 3  # Still BGR
        assert result.dtype == np.uint8

    def test_horizontal_flip(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :100] = 255  # Left half white

        config = PreprocessConfig(flip_horizontal=True)
        preprocessor = FramePreprocessor(config)
        result = preprocessor.process(frame)

        # After flip, right half should be white
        assert result[:, 100:].mean() > 200
        assert result[:, :100].mean() < 50

    def test_no_flip(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :100] = 255

        config = PreprocessConfig(flip_horizontal=False)
        preprocessor = FramePreprocessor(config)
        result = preprocessor.process(frame)

        # Left half should still be white
        assert result[:, :100].mean() > 200

    def test_max_dimension_downscale(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        config = PreprocessConfig(max_dimension=640, flip_horizontal=False)
        preprocessor = FramePreprocessor(config)
        result = preprocessor.process(frame)

        assert max(result.shape[:2]) <= 640

    def test_no_downscale_if_small(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config = PreprocessConfig(max_dimension=1280, flip_horizontal=False)
        preprocessor = FramePreprocessor(config)
        result = preprocessor.process(frame)

        assert result.shape == (480, 640, 3)

    def test_fixed_target_size(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config = PreprocessConfig(
            target_width=320,
            target_height=240,
            flip_horizontal=False,
        )
        preprocessor = FramePreprocessor(config)
        result = preprocessor.process(frame)

        assert result.shape == (240, 320, 3)

    def test_normalize(self) -> None:
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        config = PreprocessConfig(normalize=True, flip_horizontal=False)
        preprocessor = FramePreprocessor(config)
        result = preprocessor.process(frame)

        assert result.dtype == np.float32
        np.testing.assert_allclose(result.mean(), 128.0 / 255.0, atol=0.01)

    def test_none_frame_raises(self) -> None:
        preprocessor = FramePreprocessor()
        with pytest.raises(ValueError, match="None"):
            preprocessor.process(None)

    def test_wrong_dims_raises(self) -> None:
        preprocessor = FramePreprocessor()
        with pytest.raises(ValueError, match="BGR"):
            preprocessor.process(np.zeros((100, 100), dtype=np.uint8))

    def test_empty_frame_raises(self) -> None:
        preprocessor = FramePreprocessor()
        with pytest.raises(ValueError, match="empty"):
            preprocessor.process(np.zeros((0, 0, 3), dtype=np.uint8))
