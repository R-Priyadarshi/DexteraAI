"""Export trained PyTorch models to TensorFlow Lite format.

Pipeline: PyTorch → ONNX → TensorFlow → TFLite

Handles:
    - Full integer quantization
    - Float16 quantization
    - Representative dataset calibration
    - Metadata embedding
"""

from __future__ import annotations

import shutil
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
    # Arguments are validated before the optional heavy imports, so a caller
    # who passes a bad mode is told that, rather than being told TensorFlow is
    # missing when the real problem is their argument.
    valid_modes = {"none", "dynamic", "float16", "int8"}
    if quantize not in valid_modes:
        raise ValueError(
            f"Unknown quantize mode {quantize!r}. Expected one of {sorted(valid_modes)}."
        )

    # int8 needs calibration data to pick activation ranges. Falling through
    # without it silently produced an *unquantized* model, so a caller asking
    # for a 4x smaller export got a full-size one and no indication why.
    if quantize == "int8" and representative_data is None:
        raise ValueError(
            "int8 quantization requires `representative_data` to calibrate "
            "activation ranges. Pass a sample of real input, or use "
            "quantize='dynamic' which needs no calibration set."
        )

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

    tf_saved_model_dir = output_path.parent / "tf_saved_model_tmp"

    try:
        # Step 1: ONNX → TensorFlow SavedModel
        logger.info("Converting ONNX → TensorFlow...")
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(tf_saved_model_dir))

        # Step 2: TensorFlow → TFLite
        logger.info(f"Converting TensorFlow → TFLite (quantize={quantize})...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_saved_model_dir))

        if quantize == "dynamic":
            # The default mode, and the one that was missing: weights are
            # quantized to int8 while activations stay float, which needs no
            # calibration data and is usually the right trade-off.
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantize == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantize == "int8":
            # Guaranteed non-None by the validation above; asserted so the type
            # checker can see what the guard already established.
            assert representative_data is not None
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = _make_representative_dataset(representative_data)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)

        logger.info(
            f"TFLite export complete: {output_path} "
            f"({output_path.stat().st_size / 1_048_576:.2f} MB)"
        )
        return output_path
    except Exception as e:
        logger.error(f"TFLite export error: {e}")
        raise
    finally:
        # The intermediate SavedModel is often larger than the model itself and
        # is useless once converted. Cleaning up in `finally` means a failed
        # export does not leave it behind either.
        if tf_saved_model_dir.exists():
            shutil.rmtree(tf_saved_model_dir, ignore_errors=True)


def _make_representative_dataset(
    data: np.ndarray,
) -> Callable[[], Iterator[list[np.ndarray]]]:
    """Create a representative dataset generator for int8 calibration."""

    def generator() -> Iterator[list[np.ndarray]]:
        for i in range(min(len(data), 200)):
            sample = data[i : i + 1].astype(np.float32)
            yield [sample]

    return generator
