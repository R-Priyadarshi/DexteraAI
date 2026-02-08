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
    """Convert an ONNX model to TFLite.

    Args:
        onnx_path: Path to source ONNX model.
        output_path: Path for the output .tflite file.
        quantize: Quantization mode: 'none', 'dynamic', 'float16', 'int8'.
        representative_data: Calibration data for int8 quantization.
            Shape: (N, seq_len, feature_dim).

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

    # Step 1: ONNX → TensorFlow SavedModel
    logger.info("Converting ONNX → TensorFlow...")
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)

    tf_saved_model_dir = output_path.parent / "tf_saved_model_tmp"
    tf_rep.export_graph(str(tf_saved_model_dir))

    # Step 2: TensorFlow → TFLite
    logger.info("Converting TensorFlow → TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_saved_model_dir))

    # Quantization
    if quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        if representative_data is None:
            raise ValueError("int8 quantization requires representative_data")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_representative_dataset(representative_data)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    # Save
    output_path.write_bytes(tflite_model)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"TFLite model exported: {output_path} ({file_size_mb:.2f} MB, quantize={quantize})"
    )

    # Cleanup temp dir
    import shutil

    if tf_saved_model_dir.exists():
        shutil.rmtree(tf_saved_model_dir)

    return output_path


def _make_representative_dataset(
    data: np.ndarray,
) -> Callable[[], Iterator[list[np.ndarray]]]:
    """Create a representative dataset generator for int8 calibration."""

    def generator() -> Iterator[list[np.ndarray]]:
        for i in range(min(len(data), 200)):
            sample = data[i : i + 1].astype(np.float32)
            yield [sample]

    return generator
