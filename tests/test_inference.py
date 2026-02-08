"""Tests for core.inference — ONNX runtime and pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from core.inference.onnx_runtime import ONNXInferenceRuntime
from core.temporal.model import GestureTransformer
from training.export.to_onnx import export_to_onnx


class TestONNXInferenceRuntime:
    """Tests for ONNX Runtime wrapper."""

    def test_not_loaded_raises(self) -> None:
        rt = ONNXInferenceRuntime()
        assert not rt.is_loaded
        with pytest.raises(RuntimeError, match="No model loaded"):
            rt.predict({"input": np.zeros((1, 30, 86), dtype=np.float32)})

    def test_file_not_found_raises(self) -> None:
        rt = ONNXInferenceRuntime()
        with pytest.raises(FileNotFoundError):
            rt.load("/nonexistent/model.onnx")

    def test_context_manager(self) -> None:
        with ONNXInferenceRuntime() as rt:
            assert not rt.is_loaded
        # Should not raise

    @pytest.mark.slow
    def test_load_and_predict(self) -> None:
        """Full round-trip: create model → export ONNX → load → predict."""
        model = GestureTransformer(
            input_dim=86,
            num_classes=10,
            d_model=32,
            nhead=2,
            num_layers=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = export_to_onnx(
                model,
                Path(tmpdir) / "test.onnx",
                seq_len=30,
                validate=True,
            )

            rt = ONNXInferenceRuntime()
            rt.load(onnx_path)

            assert rt.is_loaded
            assert "input" in rt.input_names
            assert "logits" in rt.output_names

            result = rt.predict(
                {
                    "input": np.random.randn(1, 30, 86).astype(np.float32),
                    "mask": np.zeros((1, 30), dtype=bool),
                }
            )

            assert "logits" in result
            assert result["logits"].shape == (1, 10)

            rt.close()


class TestExportToONNX:
    """Tests for ONNX export."""

    @pytest.mark.slow
    def test_export_creates_file(self) -> None:
        model = GestureTransformer(
            input_dim=86,
            num_classes=10,
            d_model=32,
            nhead=2,
            num_layers=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_to_onnx(
                model,
                Path(tmpdir) / "gesture.onnx",
                validate=True,
            )
            assert path.exists()
            assert path.stat().st_size > 0

    @pytest.mark.slow
    def test_export_output_matches_pytorch(self) -> None:
        model = GestureTransformer(
            input_dim=86,
            num_classes=5,
            d_model=32,
            nhead=2,
            num_layers=2,
        )
        model.eval()

        dummy = torch.randn(1, 30, 86)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = export_to_onnx(model, Path(tmpdir) / "test.onnx")

            import onnxruntime as ort

            sess = ort.InferenceSession(str(onnx_path))
            ort_out = sess.run(
                None,
                {
                    "input": dummy.numpy(),
                    "mask": np.zeros((1, 30), dtype=bool),
                },
            )

            with torch.no_grad():
                pt_out = model(dummy)

            np.testing.assert_allclose(
                pt_out["logits"].numpy(),
                ort_out[0],
                atol=1e-4,
            )
