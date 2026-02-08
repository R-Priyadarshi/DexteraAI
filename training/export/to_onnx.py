"""Export trained PyTorch models to ONNX format.

Handles:
    - Dynamic axis configuration (batch + sequence length)
    - ONNX opset selection
    - Post-export validation
    - Model optimization (constant folding, dead code elimination)
    - Quantization (dynamic int8)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from loguru import logger

from core.temporal.model import GestureTransformer


def export_to_onnx(
    model: GestureTransformer,
    output_path: str | Path,
    seq_len: int = 30,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    dynamic_seq: bool = True,
    validate: bool = True,
) -> Path:
    """Export a GestureTransformer to ONNX format.

    Args:
        model: Trained PyTorch model.
        output_path: Path for the output .onnx file.
        seq_len: Default sequence length for dummy input.
        opset_version: ONNX opset version.
        dynamic_batch: Allow dynamic batch dimension.
        dynamic_seq: Allow dynamic sequence length dimension.
        validate: Validate exported model against PyTorch output.

    Returns:
        Path to the exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    # Dummy inputs
    dummy_input = torch.randn(1, seq_len, model.input_dim, device=device)
    dummy_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)

    # Dynamic axes
    dynamic_axes: dict[str, dict[int, str]] = {}
    if dynamic_batch:
        dynamic_axes["input"] = {0: "batch"}
        dynamic_axes["mask"] = {0: "batch"}
        dynamic_axes["logits"] = {0: "batch"}
        dynamic_axes["confidence"] = {0: "batch"}
    if dynamic_seq:
        dynamic_axes.setdefault("input", {})[1] = "seq_len"
        dynamic_axes.setdefault("mask", {})[1] = "seq_len"

    # Export
    logger.info(f"Exporting to ONNX (opset {opset_version})...")

    # Need a wrapper that returns tuple for ONNX export
    class ExportWrapper(torch.nn.Module):
        def __init__(self, base_model: GestureTransformer) -> None:
            super().__init__()
            self.base = base_model

        def forward(
            self, x: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.base(x, mask=mask)
            return out["logits"], out["confidence"]

    wrapper = ExportWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy_input, dummy_mask),
        str(output_path),
        opset_version=opset_version,
        input_names=["input", "mask"],
        output_names=["logits", "confidence"],
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        do_constant_folding=True,
    )

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model exported: {output_path} ({file_size_mb:.2f} MB)")

    if validate:
        _validate_onnx(model, output_path, dummy_input, dummy_mask)

    return output_path


def quantize_onnx(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Apply dynamic int8 quantization to an ONNX model.

    Args:
        input_path: Path to the source ONNX model.
        output_path: Path for quantized model (default: adds '_quant' suffix).

    Returns:
        Path to the quantized ONNX model.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_quant")
    output_path = Path(output_path)

    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )

    orig_size = input_path.stat().st_size / (1024 * 1024)
    quant_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100

    logger.info(
        f"Quantized ONNX model: {output_path} | "
        f"{orig_size:.2f} MB → {quant_size:.2f} MB "
        f"({reduction:.1f}% reduction)"
    )

    return output_path


def _validate_onnx(
    pytorch_model: GestureTransformer,
    onnx_path: Path,
    dummy_input: torch.Tensor,
    dummy_mask: torch.Tensor,
    atol: float = 1e-4,
) -> None:
    """Validate ONNX output matches PyTorch output."""
    # PyTorch reference
    pytorch_model.eval()
    with torch.no_grad():
        pt_output = pytorch_model(dummy_input, mask=dummy_mask)
        pt_logits = pt_output["logits"].cpu().numpy()

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    ort_outputs = session.run(
        None,
        {
            "input": dummy_input.cpu().numpy(),
            "mask": dummy_mask.cpu().numpy(),
        },
    )
    ort_logits = ort_outputs[0]

    # Compare
    max_diff = np.max(np.abs(pt_logits - ort_logits))
    matches = max_diff < atol

    if matches:
        logger.info(
            f"ONNX validation PASSED (max diff: {max_diff:.6f})"
        )
    else:
        logger.warning(
            f"ONNX validation FAILED (max diff: {max_diff:.6f}, "
            f"tolerance: {atol})"
        )
