"""Tests for core.inference.pipeline — the end-to-end GesturePipeline.

These exist because three independent, mutually-incompatible API mismatches
(pipeline vs. CLI, pipeline vs. FastAPI routes, engine vs. web dashboard)
shipped unnoticed while this module had zero coverage. The contract tests
below pin the public surface that callers actually use.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from core.inference.pipeline import DEFAULT_GESTURE_LABELS, GesturePipeline, PipelineConfig
from core.types import FrameResult
from tests.conftest import requires_mediapipe_bundle


def _blank_frame(height: int = 240, width: int = 320) -> np.ndarray:
    """A valid BGR frame containing no hands."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestPipelineContract:
    """The API surface other modules call. Breaking these breaks the app."""

    def test_exposes_methods_used_by_callers(self) -> None:
        # dextera.py `demo` and backend/apps/api/routes.py both rely on these.
        for name in ("start", "stop", "process_frame", "enable_privacy", "disable_privacy"):
            assert callable(getattr(GesturePipeline, name, None)), f"missing {name}()"
        assert isinstance(GesturePipeline.is_running, property)

    def test_process_frame_signature_takes_a_frame(self) -> None:
        params = list(inspect.signature(GesturePipeline.process_frame).parameters)
        assert params == ["self", "frame"]

    def test_is_context_manager(self) -> None:
        assert hasattr(GesturePipeline, "__enter__")
        assert hasattr(GesturePipeline, "__exit__")

    def test_config_defaults_are_sane(self) -> None:
        config = PipelineConfig()
        assert config.max_hands >= 1
        assert config.sequence_length > 0
        assert 0.0 <= config.confidence_threshold <= 1.0
        assert config.gesture_labels == DEFAULT_GESTURE_LABELS
        # Mutating one config's labels must not leak into the next.
        config.gesture_labels.append("mutated")
        assert "mutated" not in PipelineConfig().gesture_labels


@requires_mediapipe_bundle
class TestPipelineLifecycle:
    """Start/stop behaviour and guard rails."""

    def test_not_running_before_start(self) -> None:
        pipeline = GesturePipeline(PipelineConfig())
        assert pipeline.is_running is False

    def test_process_frame_before_start_raises(self) -> None:
        pipeline = GesturePipeline(PipelineConfig())
        with pytest.raises(RuntimeError, match="not started"):
            pipeline.process_frame(_blank_frame())

    def test_accepts_no_config(self) -> None:
        # routes.py constructs GesturePipeline(PipelineConfig()); demo passes one too.
        assert GesturePipeline().config.max_hands == PipelineConfig().max_hands

    def test_missing_model_falls_back_to_detection_only(self) -> None:
        pipeline = GesturePipeline(PipelineConfig(model_path="/nonexistent/model.pt"))
        with pipeline:
            assert pipeline.is_running is True
            assert pipeline.has_model is False

    def test_start_stop_cycle(self) -> None:
        pipeline = GesturePipeline(PipelineConfig(max_hands=1))
        pipeline.start()
        assert pipeline.is_running is True
        pipeline.stop()
        assert pipeline.is_running is False

    def test_privacy_toggle(self) -> None:
        pipeline = GesturePipeline(PipelineConfig())
        pipeline.enable_privacy()
        assert pipeline._privacy_mode is True
        pipeline.disable_privacy()
        assert pipeline._privacy_mode is False


@requires_mediapipe_bundle
class TestPipelineInference:
    """Frame processing. Requires the MediaPipe task bundle."""

    def test_blank_frame_returns_empty_result(self) -> None:
        with GesturePipeline(PipelineConfig(max_hands=1)) as pipeline:
            result = pipeline.process_frame(_blank_frame())

        assert isinstance(result, FrameResult)
        assert result.hands == []
        assert result.gestures == []
        assert result.inference_time_ms >= 0.0

    def test_invalid_frame_raises(self) -> None:
        with (
            GesturePipeline(PipelineConfig(max_hands=1)) as pipeline,
            pytest.raises(ValueError),
        ):
            pipeline.process_frame(np.zeros((10, 10), dtype=np.uint8))

    def test_privacy_mode_strips_landmarks(self) -> None:
        config = PipelineConfig(max_hands=1, privacy_mode=True)
        with GesturePipeline(config) as pipeline:
            result = pipeline.process_frame(_blank_frame())
        assert result.hands == []

    def test_callbacks_fire(self) -> None:
        events: list[str] = []

        class RecordingCallback:
            def on_pipeline_start(self, pipeline: GesturePipeline) -> None:
                events.append("start")

            def on_frame(self, pipeline: GesturePipeline, result: FrameResult) -> None:
                events.append("frame")

            def on_pipeline_stop(self, pipeline: GesturePipeline) -> None:
                events.append("stop")

        config = PipelineConfig(max_hands=1, callbacks=[RecordingCallback()])
        with GesturePipeline(config) as pipeline:
            pipeline.process_frame(_blank_frame())

        assert events == ["start", "frame", "stop"]

    def test_partial_callbacks_are_tolerated(self) -> None:
        """A callback implementing only one hook must not break the pipeline."""

        class OnlyFrame:
            def __init__(self) -> None:
                self.frames = 0

            def on_frame(self, pipeline: GesturePipeline, result: FrameResult) -> None:
                self.frames += 1

        cb = OnlyFrame()
        with GesturePipeline(PipelineConfig(max_hands=1, callbacks=[cb])) as pipeline:
            pipeline.process_frame(_blank_frame())
        assert cb.frames == 1
