"""Inference module — cross-platform runtime abstraction."""

from core.inference.onnx_runtime import ONNXInferenceRuntime
from core.inference.pipeline import GesturePipeline

__all__ = ["ONNXInferenceRuntime", "GesturePipeline"]
