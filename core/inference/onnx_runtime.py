"""ONNX Runtime inference backend.

Provides GPU-accelerated (via CUDA/DirectML) or CPU inference
for exported gesture models. Supports dynamic batching and
multiple execution providers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnxruntime as ort
from loguru import logger

if TYPE_CHECKING:
    import numpy as np


class ONNXInferenceRuntime:
    """ONNX Runtime wrapper for cross-platform gesture inference.

    Supports:
        - Automatic provider selection (CUDA → DirectML → CPU)
        - Session options (thread count, graph optimization)
        - Input/output name discovery
        - Performance profiling

    Usage:
        >>> runtime = ONNXInferenceRuntime()
        >>> runtime.load("models/gesture.onnx")
        >>> result = runtime.predict({"input": features_array})
        >>> runtime.close()
    """

    def __init__(
        self,
        providers: list[str] | None = None,
        num_threads: int = 4,
        enable_profiling: bool = False,
    ) -> None:
        """Initialize ONNX Runtime.

        Args:
            providers: Execution providers in priority order.
                       Defaults to auto-detection.
            num_threads: Number of intra-op threads for CPU.
            enable_profiling: Enable ONNX Runtime profiling.
        """
        self._providers = providers or self._detect_providers()
        self._num_threads = num_threads
        self._enable_profiling = enable_profiling
        self._session: ort.InferenceSession | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    @staticmethod
    def _detect_providers() -> list[str]:
        """Auto-detect available execution providers."""
        available = ort.get_available_providers()
        preferred_order = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [p for p in preferred_order if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        return providers

    def load(self, model_path: str | Path) -> None:
        """Load an ONNX model.

        Args:
            model_path: Path to .onnx model file.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model loading fails.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self._num_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self._enable_profiling:
            session_options.enable_profiling = True

        try:
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=self._providers,
            )
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]

            active_provider = self._session.get_providers()[0]
            logger.info(
                f"Loaded ONNX model: {model_path.name} | "
                f"Provider: {active_provider} | "
                f"Inputs: {self._input_names} | "
                f"Outputs: {self._output_names}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference.

        Args:
            inputs: Dict mapping input names to numpy arrays.

        Returns:
            Dict mapping output names to numpy arrays.

        Raises:
            RuntimeError: If session is not loaded.
        """
        if self._session is None:
            raise RuntimeError("No model loaded. Call load() first.")

        outputs = self._session.run(self._output_names, inputs)
        return dict(zip(self._output_names, outputs, strict=False))

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def close(self) -> None:
        """Release ONNX Runtime session."""
        self._session = None

    def __enter__(self) -> ONNXInferenceRuntime:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
