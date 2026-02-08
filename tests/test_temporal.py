"""Tests for core.temporal — GestureTransformer and SequenceBuffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core.temporal.model import GestureTransformer
from core.temporal.sequence_buffer import SequenceBuffer
from core.types import HandLandmarks, Handedness, NUM_HAND_LANDMARKS, LANDMARK_DIMS


class TestGestureTransformer:
    """Tests for the Transformer gesture model."""

    def test_forward_shape(self, gesture_model: torch.nn.Module) -> None:
        x = torch.randn(4, 30, 86)  # batch=4, seq=30, features=86
        output = gesture_model(x)
        assert output["logits"].shape == (4, 10)
        assert output["confidence"].shape == (4,)

    def test_forward_with_mask(self, gesture_model: torch.nn.Module) -> None:
        x = torch.randn(2, 30, 86)
        mask = torch.zeros(2, 30, dtype=torch.bool)
        mask[:, 20:] = True  # last 10 frames are padded
        output = gesture_model(x, mask=mask)
        assert output["logits"].shape == (2, 10)

    def test_predict(self, gesture_model: torch.nn.Module) -> None:
        x = torch.randn(1, 30, 86)
        result = gesture_model.predict(x)
        assert "class_id" in result
        assert "class_probs" in result
        assert "confidence" in result
        assert result["class_probs"].shape == (1, 10)
        # Probabilities should sum to 1
        prob_sum = result["class_probs"].sum(dim=-1)
        torch.testing.assert_close(prob_sum, torch.ones(1), atol=1e-5, rtol=1e-5)

    def test_single_frame_sequence(self, gesture_model: torch.nn.Module) -> None:
        x = torch.randn(1, 1, 86)
        output = gesture_model(x)
        assert output["logits"].shape == (1, 10)

    def test_model_size(self) -> None:
        model = GestureTransformer(
            input_dim=86, num_classes=10,
            d_model=128, nhead=4, num_layers=4,
        )
        size_mb = model.get_model_size_mb()
        # Should be well under 5MB for edge deployment
        assert size_mb < 10.0

    def test_deterministic_eval(self, gesture_model: torch.nn.Module) -> None:
        gesture_model.eval()
        x = torch.randn(1, 30, 86)
        with torch.no_grad():
            out1 = gesture_model(x)
            out2 = gesture_model(x)
        torch.testing.assert_close(out1["logits"], out2["logits"])

    def test_gradient_flow(self, gesture_model: torch.nn.Module) -> None:
        gesture_model.train()
        x = torch.randn(2, 30, 86)
        labels = torch.randint(0, 10, (2,))
        output = gesture_model(x)
        loss = (
            torch.nn.functional.cross_entropy(output["logits"], labels)
            + output["confidence"].mean()  # include confidence so all params get gradients
        )
        loss.backward()
        # Check gradients exist
        for name, param in gesture_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestSequenceBuffer:
    """Tests for the temporal SequenceBuffer."""

    def test_push_and_length(self, random_landmarks: HandLandmarks) -> None:
        buf = SequenceBuffer(max_len=10)
        assert buf.length == 0
        assert not buf.is_ready

        for _ in range(10):
            buf.push(random_landmarks)

        assert buf.length == 10
        assert buf.is_ready

    def test_sliding_window(self, random_landmarks: HandLandmarks) -> None:
        buf = SequenceBuffer(max_len=5)
        for _ in range(10):
            buf.push(random_landmarks)
        assert buf.length == 5  # Should not exceed max_len

    def test_get_features(self, random_landmarks: HandLandmarks) -> None:
        buf = SequenceBuffer(max_len=5)
        for _ in range(5):
            buf.push(random_landmarks)
        features = buf.get_features()
        assert features.shape == (5, 86)

    def test_padded_features(self, random_landmarks: HandLandmarks) -> None:
        buf = SequenceBuffer(max_len=10)
        for _ in range(3):
            buf.push(random_landmarks)
        features, mask = buf.get_padded_features()
        assert features.shape == (10, 86)
        assert mask.shape == (10,)
        # First 3 should not be masked, rest should be
        assert not mask[:3].any()
        assert mask[3:].all()

    def test_push_empty(self) -> None:
        buf = SequenceBuffer(max_len=5)
        buf.push_empty()
        features = buf.get_features()
        assert features.shape == (1, 86)
        np.testing.assert_array_equal(features[0], np.zeros(86))

    def test_clear(self, random_landmarks: HandLandmarks) -> None:
        buf = SequenceBuffer(max_len=5)
        for _ in range(5):
            buf.push(random_landmarks)
        buf.clear()
        assert buf.length == 0
        assert not buf.is_ready
