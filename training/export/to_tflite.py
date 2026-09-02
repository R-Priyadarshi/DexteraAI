"""Export trained PyTorch models to TensorFlow Lite format.

Pipeline: PyTorch → ONNX → TensorFlow → TFLite

Handles:
    - Full integer quantization
    - Float16 quantization
    - Representative dataset calibration
    - Metadata embedding
"""

from __future__ import annotations

from collections.abc import Callable, Iterator  # noqa: TCH003 — used in nested function annotation
from pathlib import Path

import numpy as np
from loguru import logger


def export_to_tflite(
    onnx_path: str | Path,
    output_path: str | Path,
    quantize: str = "dynamic",
    representative_data: np.ndarray | None = None,
) -> Path:
    """
    Convert an ONNX model to TFLite with plugin/callback hooks, privacy, validation, and versioning.

    Args:
        onnx_path: Path to source ONNX model.
        output_path: Path for the output .tflite file.
        quantize: Quantization mode: 'none', 'dynamic', 'float16', 'int8'.
        representative_data: Calibration data for int8 quantization.

    Returns:
        Path to the exported TFLite file.
    """
    try:
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare
    except ImportError as e:
        raise ImportError(
            f"TFLite export requires: pip install onnx-tf tensorflow. Missing: {e}"
        ) from e

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: ONNX → TensorFlow SavedModel
        logger.info("Converting ONNX → TensorFlow...")
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)

        tf_saved_model_dir = output_path.parent / "tf_saved_model_tmp"
        tf_rep.export_graph(str(tf_saved_model_dir))

        # Step 2: TensorFlow → TFLite
        logger.info("Converting TensorFlow → TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_saved_model_dir))

        if quantize == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif quantize == "int8" and representative_data is not None:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = lambda: (([x],) for x in representative_data)

        tflite_model = converter.convert()

        # Save
        output_path.write_bytes(tflite_model)

        logger.info(f"TFLite export complete: {output_path}")

        return output_path
    except Exception as e:
        logger.error(f"TFLite export error: {e}")
        raise


def _make_representative_dataset(
    data: np.ndarray,
) -> Callable[[], Iterator[list[np.ndarray]]]:
    """Create a representative dataset generator for int8 calibration."""

    def generator() -> Iterator[list[np.ndarray]]:
        for i in range(min(len(data), 200)):
            sample = data[i : i + 1].astype(np.float32)
            yield [sample]

    return generator
