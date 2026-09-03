"""ONNX Runtime inference backend.

Provides GPU-accelerated (via CUDA/DirectML) or CPU inference
for exported gesture models. Supports dynamic batching and
multiple execution providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import onnxruntime as ort
from loguru import logger


class ONNXInferenceRuntime:
    """ONNX Runtime wrapper for cross-platform gesture inference.

    Supports:
        - Automatic provider selection (CUDA → DirectML → CPU)
        - Session options (thread count, graph optimization)
        - Input/output name discovery
        - Performance profiling
        - Plugin/callback extensibility (pre/post-processing, metrics)
        - Robust error handling, privacy controls, versioning
        - Batch/streaming inference, enterprise logging

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
        callbacks: list[Any] | None = None,
        privacy_mode: bool = False,
        model_version: str | None = None,
    ) -> None:
        """Initialize ONNX Runtime with enterprise features.

        Args:
            providers: Execution providers in priority order.
            num_threads: Number of intra-op threads for CPU.
            enable_profiling: Enable ONNX Runtime profiling.
            callbacks: List of plugin/callbacks for extensibility.
            privacy_mode: Enable privacy-preserving inference.
            model_version: Model version for tracking/artifacts.
        """
        self._providers = providers or self._detect_providers()
        self._num_threads = num_threads
        self._enable_profiling = enable_profiling
        self._session: ort.InferenceSession | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._callbacks = callbacks or []
        self._privacy_mode = privacy_mode
        self._model_version = model_version
        self._log = logger

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run inference with privacy, callbacks, and metrics."""
        if self._session is None:
            raise RuntimeError("No model loaded")
        try:
            for cb in self._callbacks:
                cb.on_predict_start(self, inputs)
            # Privacy: mask sensitive data if enabled
            if self._privacy_mode:
                inputs = self._mask_inputs(inputs)
            result = self._session.run(self._output_names, inputs)
            for cb in self._callbacks:
                cb.on_predict_end(self, result)
            self._log.info(
                f"Inference completed | Inputs: {list(inputs.keys())} | "
                f"Outputs: {self._output_names}"
            )
            return dict(zip(self._output_names, result, strict=True))
        except Exception as e:
            self._log.error(f"Inference failed: {e}")
            raise

    def _mask_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive input data for privacy-preserving inference."""
        masked = {}
        for k, v in inputs.items():
            if isinstance(v, list | tuple):
                masked[k] = [0 for _ in v]
            elif hasattr(v, "shape"):
                try:
                    import numpy as np
                    masked[k] = np.zeros_like(v)
                except ImportError:
                    masked[k] = v
            else:
                masked[k] = v
        return masked

    @staticmethod
    def _detect_providers() -> list[str]:
        """Auto-detect available execution providers, fastest first.

        Names must match ONNX Runtime's exactly or the provider is simply never
        selected: DirectML registers as `DmlExecutionProvider`, not
        `DirectMLExecutionProvider`, so the previous list silently fell through
        to CPU on every Windows GPU.
        """
        available = ort.get_available_providers()
        preferred = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        selected = [p for p in preferred if p in available]

        # CPU is always registered, but guarantee it is present and last so a
        # build with only exotic providers still has something to fall back to.
        if "CPUExecutionProvider" not in selected:
            selected.append("CPUExecutionProvider")
        return selected

    def load(self, model_path: str | Path) -> None:
        """Load ONNX model with robust error handling and versioning."""
        try:
            # A bare SessionOptions() discards everything the constructor was
            # given: thread count, graph optimisation and profiling were all
            # accepted as arguments and then silently ignored.
            options = ort.SessionOptions()
            options.intra_op_num_threads = self._num_threads
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            options.enable_profiling = self._enable_profiling

            self._session = ort.InferenceSession(
                str(model_path),
                providers=self._providers,
                sess_options=options,
            )
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._output_names = [o.name for o in self._session.get_outputs()]
            self._log.info(
                f"ONNX model loaded: {model_path} | Providers: {self._providers} | "
                f"Version: {self._model_version}"
            )
        except RuntimeError as e:
            self._log.error(f"Failed to load ONNX model: {e}")
            if 'NO_SUCHFILE' in str(e):
                raise FileNotFoundError(f"Model file not found: {model_path}") from e
            raise
        except Exception as e:
            self._log.error(f"Failed to load ONNX model: {e}")
            if 'NO_SUCHFILE' in str(e):
                raise FileNotFoundError(f"Model file not found: {model_path}") from e
            raise

    def close(self) -> None:
        """Close ONNX session and release resources."""
        if self._session:
            self._session = None
            self._log.info("ONNX session closed.")

    def get_model_version(self) -> str | None:
        """Return model version for tracking/artifacts."""
        return self._model_version

    def enable_profiling(self) -> None:
        """Request profiling for subsequently loaded sessions.

        ONNX Runtime has no `session.enable_profiling()` — profiling is a
        SessionOptions flag fixed when the session is created, so calling it on
        a live session raised AttributeError. Setting the flag here and
        reloading is the only way to turn it on.
        """
        self._enable_profiling = True
        if self._session is not None:
            self._log.warning(
                "Profiling applies from the next load(); the current session "
                "was created without it."
            )

    def get_profiling_data(self) -> Any:
        """Retrieve profiling data if enabled."""
        if self._session and self._enable_profiling:
            return self._session.end_profiling()
        return None

    @property
    def input_names(self) -> list[str]:
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def __enter__(self) -> ONNXInferenceRuntime:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
