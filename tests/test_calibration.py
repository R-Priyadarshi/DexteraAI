"""Tests for core.calibration — UserCalibrator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.calibration.calibrator import UserCalibrator
from core.types import HandLandmarks, Handedness


class TestUserCalibrator:
    """Tests for per-user gesture calibration."""

    def _make_hand(self, seed: int = 0) -> HandLandmarks:
        rng = np.random.default_rng(seed)
        return HandLandmarks(
            landmarks=rng.random((21, 3)).astype(np.float32),
            handedness=Handedness.RIGHT,
            confidence=0.95,
        )

    def test_calibration_lifecycle(self) -> None:
        cal = UserCalibrator(user_id="test_user")

        assert not cal.is_calibrating

        cal.start_calibration()
        assert cal.is_calibrating

        # Add samples
        for i in range(5):
            count = cal.add_sample("thumbs_up", self._make_hand(seed=i))
        assert count == 5

        for i in range(3):
            cal.add_sample("peace", self._make_hand(seed=100 + i))

        profile = cal.finish_calibration()
        assert not cal.is_calibrating
        assert "thumbs_up" in profile.gesture_references
        assert "peace" in profile.gesture_references
        assert len(profile.gesture_references["thumbs_up"]) == 5

    def test_insufficient_samples_raises(self) -> None:
        cal = UserCalibrator(user_id="test")
        cal.start_calibration()
        cal.add_sample("wave", self._make_hand(0))
        cal.add_sample("wave", self._make_hand(1))

        with pytest.raises(ValueError, match="need at least"):
            cal.finish_calibration()

    def test_add_sample_without_start_raises(self) -> None:
        cal = UserCalibrator(user_id="test")
        with pytest.raises(RuntimeError, match="start_calibration"):
            cal.add_sample("test", self._make_hand())

    def test_similarity(self) -> None:
        cal = UserCalibrator(user_id="test")
        cal.start_calibration()

        hand = self._make_hand(42)
        for _ in range(5):
            cal.add_sample("fist", hand)

        cal.finish_calibration()

        # Same hand should have high similarity
        sim = cal.compute_similarity("fist", hand)
        assert sim > 0.9

        # Unknown gesture should return 0
        sim_unknown = cal.compute_similarity("unknown_gesture", hand)
        assert sim_unknown == 0.0

    def test_save_and_load(self) -> None:
        cal = UserCalibrator(user_id="persist_test")
        cal.start_calibration()
        for i in range(3):
            cal.add_sample("ok", self._make_hand(i))
        cal.finish_calibration()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profile.json"
            cal.save(path)

            assert path.exists()
            data = json.loads(path.read_text())
            assert data["user_id"] == "persist_test"

            # Load into new calibrator
            cal2 = UserCalibrator(user_id="persist_test")
            profile = cal2.load(path)
            assert "ok" in profile.gesture_references
            assert len(profile.gesture_references["ok"]) == 3

    def test_load_nonexistent_raises(self) -> None:
        cal = UserCalibrator(user_id="test")
        with pytest.raises(FileNotFoundError):
            cal.load("/nonexistent/path.json")
