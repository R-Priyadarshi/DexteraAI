"""End-to-end gesture recognition pipeline.

Orchestrates the full flow: frame → preprocessing → hand detection →
landmark normalization → feature extraction → temporal buffering →
gesture classification → result output.

This is the main entry point for all applications (web, mobile, desktop).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from core.landmarks.features import LandmarkFeatureExtractor
from core.landmarks.normalizer import LandmarkNormalizer, NormalizationMode
from core.temporal.model import GestureTransformer
from core.temporal.sequence_buffer import SequenceBuffer
from core.types import FrameResult, GestureResult, GestureState, HandLandmarks
from core.vision.detector import MediaPipeHandDetector
from core.vision.preprocessor import FramePreprocessor, PreprocessConfig

if TYPE_CHECKING:
    import numpy as np

DEFAULT_GESTURE_LABELS = [
    "none",
    "open_palm",
    "closed_fist",
    "thumbs_up",
    "thumbs_down",
    "peace",
    "pointing_up",
    "ok_sign",
    "pinch",
    "wave",
]


@dataclass
class PipelineConfig:
    """Configuration for the gesture pipeline.

    Attributes:
        max_hands: Maximum number of hands to track.
        sequence_length: Number of frames in the temporal window.
        confidence_threshold: Minimum confidence to report a gesture.
        model_path: Path to a trained gesture model (.pt checkpoint).
        gesture_labels: Ordered list of gesture class names. Overridden by the
            model bundle's labels.json when one sits next to model_path.
        use_gpu: Whether to use GPU for inference.
        normalization_mode: Landmark normalization strategy.
        preprocess_config: Frame preprocessing settings.
        callbacks: Plugin/callback hooks for extensibility.
        privacy_mode: Drop landmark payloads from results (labels only).
        model_version: Model version string for tracking/artifacts.
    """

    max_hands: int = 2
    sequence_length: int = 30
    confidence_threshold: float = 0.6
    model_path: str | None = None
    gesture_labels: list[str] = field(default_factory=lambda: list(DEFAULT_GESTURE_LABELS))
    use_gpu: bool = False
    normalization_mode: NormalizationMode = NormalizationMode.FULL
    preprocess_config: PreprocessConfig = field(default_factory=PreprocessConfig)
    callbacks: list[Any] = field(default_factory=list)
    privacy_mode: bool = False
    model_version: str | None = None


class GesturePipeline:
    """Production gesture recognition pipeline.

    Combines all core modules into a single interface. Handles multi-hand
    tracking, temporal buffering, and classification.

    Usage:
        >>> pipeline = GesturePipeline(PipelineConfig())
        >>> pipeline.start()
        >>> result = pipeline.process_frame(bgr_frame)
        >>> for gesture in result.gestures:
        ...     print(f"{gesture.gesture_name}: {gesture.confidence:.2f}")
        >>> pipeline.stop()

    Or as a context manager:
        >>> with GesturePipeline(PipelineConfig()) as pipeline:
        ...     result = pipeline.process_frame(bgr_frame)

    Callback hooks (all optional, duck-typed):
        on_pipeline_start(pipeline)
        on_frame(pipeline, result)
        on_pipeline_error(pipeline, exception)
        on_pipeline_stop(pipeline)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._callbacks: list[Any] = list(self._config.callbacks)
        self._preprocessor: FramePreprocessor | None = None
        self._detector: MediaPipeHandDetector | None = None
        self._normalizer: LandmarkNormalizer | None = None
        self._feature_extractor: LandmarkFeatureExtractor | None = None
        self._buffers: dict[int, SequenceBuffer] = {}
        self._model: GestureTransformer | None = None
        self._labels: list[str] = list(self._config.gesture_labels)
        self._device = torch.device(
            "cuda" if self._config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        self._is_running = False
        self._privacy_mode = self._config.privacy_mode
        self._model_version = self._config.model_version
        # Confidence calibration from the model bundle (temperature scaling +
        # an open-set rejection threshold). Populated by _load_labels().
        self._temperature: float = 1.0
        self._rejection_threshold: float | None = None

    # ── Properties ────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    @property
    def has_model(self) -> bool:
        """False means detection-only mode (landmarks but no classification)."""
        return self._model is not None

    def get_model_version(self) -> str | None:
        return self._model_version

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting gesture pipeline...")

        self._preprocessor = FramePreprocessor(self._config.preprocess_config)
        self._detector = MediaPipeHandDetector(max_hands=self._config.max_hands)
        self._normalizer = LandmarkNormalizer(mode=self._config.normalization_mode)
        self._feature_extractor = LandmarkFeatureExtractor()

        for i in range(self._config.max_hands):
            self._buffers[i] = SequenceBuffer(
                max_len=self._config.sequence_length,
                feature_extractor=self._feature_extractor,
            )

        if self._config.model_path:
            self._load_model(self._config.model_path)

        self._is_running = True

        for cb in self._callbacks:
            if hasattr(cb, "on_pipeline_start"):
                cb.on_pipeline_start(self)

        logger.info(
            "Pipeline started | device={} | max_hands={} | seq_len={} | "
            "classes={} | model={} | privacy={}",
            self._device,
            self._config.max_hands,
            self._config.sequence_length,
            len(self._labels),
            "loaded" if self._model else "detection-only",
            self._privacy_mode,
        )

    def stop(self) -> None:
        """Release all resources."""
        if self._detector:
            self._detector.close()
            self._detector = None
        self._buffers.clear()
        self._model = None
        self._is_running = False

        for cb in self._callbacks:
            if hasattr(cb, "on_pipeline_stop"):
                cb.on_pipeline_stop(self)

        logger.info("Pipeline stopped.")

    # ── Inference ─────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Process a single BGR frame through the full pipeline.

        Args:
            frame: BGR image (H, W, 3), dtype uint8.

        Returns:
            FrameResult with detected hands and recognized gestures.

        Raises:
            RuntimeError: If the pipeline has not been started.
        """
        if not self._is_running:
            raise RuntimeError("Pipeline not started. Call start() first.")

        assert self._preprocessor is not None
        assert self._detector is not None
        assert self._normalizer is not None

        t_start = time.perf_counter()

        try:
            processed = self._preprocessor.process(frame)
            hands = self._detector.detect(processed)
            normalized_hands = self._normalizer.normalize_batch(hands)

            gestures: list[GestureResult] = []
            for i, hand in enumerate(normalized_hands[: self._config.max_hands]):
                self._buffers[i].push(hand)

                if self._model and self._buffers[i].is_ready:
                    gesture = self._classify_gesture(i, hand)
                    if gesture:
                        gestures.append(gesture)

            # Keep buffers aligned for hands that dropped out of view.
            for i in range(len(normalized_hands), self._config.max_hands):
                self._buffers[i].push_empty()

            result = FrameResult(
                hands=[] if self._privacy_mode else normalized_hands,
                gestures=gestures,
                timestamp_ms=time.time() * 1000.0,
                inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

            for cb in self._callbacks:
                if hasattr(cb, "on_frame"):
                    cb.on_frame(self, result)

            return result

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            for cb in self._callbacks:
                if hasattr(cb, "on_pipeline_error"):
                    cb.on_pipeline_error(self, e)
            raise

    def process_image(self, frame: np.ndarray) -> FrameResult:
        """Classify a single still image, independently of the streaming buffer.

        `process_frame` accumulates a temporal window across calls, so a one-shot
        request would never reach the sequence length needed to classify. Here the
        detected pose is held across the whole window instead — the same shape the
        model was trained on for static images (see `expand_static` in
        GestureSequenceDataset) — so a single frame yields a real prediction.

        Leaves the streaming buffers untouched, so it is safe to interleave with
        `process_frame`.

        Args:
            frame: BGR image (H, W, 3), dtype uint8.

        Returns:
            FrameResult with detected hands and their gestures.
        """
        if not self._is_running:
            raise RuntimeError("Pipeline not started. Call start() first.")

        assert self._preprocessor is not None
        assert self._detector is not None
        assert self._normalizer is not None
        assert self._feature_extractor is not None

        t_start = time.perf_counter()
        processed = self._preprocessor.process(frame)
        hands = self._detector.detect(processed)
        normalized_hands = self._normalizer.normalize_batch(hands)

        gestures: list[GestureResult] = []
        if self._model is not None:
            for hand in normalized_hands[: self._config.max_hands]:
                gesture = self._classify_static(hand)
                if gesture:
                    gestures.append(gesture)

        result = FrameResult(
            hands=[] if self._privacy_mode else normalized_hands,
            gestures=gestures,
            timestamp_ms=time.time() * 1000.0,
            inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )

        for cb in self._callbacks:
            if hasattr(cb, "on_frame"):
                cb.on_frame(self, result)

        return result

    def _classify_static(self, hand: HandLandmarks) -> GestureResult | None:
        """Classify one hand by holding its pose across the full temporal window."""
        assert self._feature_extractor is not None
        import numpy as _np

        features_1d = self._feature_extractor.extract(hand)
        features = _np.repeat(features_1d[None, :], self._config.sequence_length, axis=0).astype(
            _np.float32
        )
        mask = _np.zeros(self._config.sequence_length, dtype=bool)

        x = torch.from_numpy(features).unsqueeze(0).to(self._device)
        m = torch.from_numpy(mask).unsqueeze(0).to(self._device)
        return self._predict_from_window(x, m, hand)

    def _classify_gesture(self, hand_idx: int, hand: HandLandmarks) -> GestureResult | None:
        """Run gesture classification on the buffered sequence for one hand."""
        assert self._model is not None
        features, mask = self._buffers[hand_idx].get_padded_features()

        x = torch.from_numpy(features).unsqueeze(0).to(self._device)  # (1, S, F)
        m = torch.from_numpy(mask).unsqueeze(0).to(self._device)  # (1, S)

        return self._predict_from_window(x, m, hand)

    def _predict_from_window(
        self, x: torch.Tensor, m: torch.Tensor, hand: HandLandmarks
    ) -> GestureResult | None:
        """Run the model on a prepared (1, S, F) window and apply calibration."""
        assert self._model is not None
        if self._temperature != 1.0:
            # Temperature scaling: recompute probabilities from calibrated logits
            # so the reported confidence means what it says.
            self._model.eval()
            with torch.no_grad():
                logits = self._model(x, mask=m)["logits"] / self._temperature
                probs = torch.softmax(logits, dim=-1)[0]
            class_id = int(torch.argmax(probs).item())
        else:
            result = self._model.predict(x, mask=m)
            class_id = int(result["class_id"].item())
            probs = result["class_probs"][0]

        class_prob = float(probs[class_id].item())

        # A calibrated rejection threshold, when the bundle ships one, is what
        # keeps out-of-vocabulary hand shapes from being reported confidently.
        threshold = (
            self._rejection_threshold
            if self._rejection_threshold is not None
            else self._config.confidence_threshold
        )
        if class_prob < threshold:
            return None

        gesture_name = (
            self._labels[class_id] if class_id < len(self._labels) else f"gesture_{class_id}"
        )

        return GestureResult(
            gesture_id=class_id,
            gesture_name=gesture_name,
            confidence=class_prob,
            state=GestureState.RECOGNIZED,
            landmarks=None if self._privacy_mode else hand,
        )

    # ── Model loading ─────────────────────────────────────────

    def _load_model(self, model_path: str) -> None:
        """Load a trained gesture model and its label bundle."""
        path = Path(model_path)
        if not path.exists():
            logger.warning(f"Model not found: {path}. Running in detection-only mode.")
            return

        if path.suffix != ".pt":
            logger.warning(
                f"Unsupported model format for this pipeline: {path.suffix}. "
                "Use ONNXInferenceRuntime for .onnx models. Running in detection-only mode."
            )
            return

        self._load_labels(path)

        feature_dim = LandmarkFeatureExtractor().feature_dim
        try:
            # Preferred: refuse to unpickle arbitrary objects.
            ckpt = torch.load(str(path), map_location=self._device, weights_only=True)
        except Exception:
            # Legacy checkpoints embed a config dataclass; trust only local files.
            logger.warning(f"Falling back to unsafe load for legacy checkpoint: {path}")
            # The suppression below is deliberate: weights_only=True is tried
            # exists only for checkpoints this project produced itself, which
            # embed a config dataclass. Never point it at a downloaded file.
            ckpt = torch.load(  # nosec B614
                str(path), map_location=self._device, weights_only=False
            )

        # Trainer checkpoints wrap the weights; raw state_dicts are also accepted.
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        # Infer the class count from the checkpoint so the head always matches.
        num_classes = len(self._labels)
        head_weight = state_dict.get("classifier.3.weight")
        if head_weight is not None:
            num_classes = int(head_weight.shape[0])
            if num_classes != len(self._labels):
                logger.warning(
                    f"Checkpoint has {num_classes} classes but {len(self._labels)} labels "
                    "are configured. Falling back to generated label names."
                )
                self._labels = [
                    self._labels[i] if i < len(self._labels) else f"gesture_{i}"
                    for i in range(num_classes)
                ]

        self._model = GestureTransformer(
            input_dim=feature_dim,
            num_classes=num_classes,
            d_model=128,
            nhead=4,
            num_layers=4,
        ).to(self._device)

        self._model.load_state_dict(state_dict)
        self._model.eval()

        if self._model_version is None:
            self._model_version = str(ckpt.get("epoch", "")) if isinstance(ckpt, dict) else None

        logger.info(
            f"Loaded model: {path.name} "
            f"({self._model.get_model_size_mb():.2f} MB, {num_classes} classes)"
        )

    def _load_labels(self, model_path: Path) -> None:
        """Load labels.json from the model bundle directory, if present."""
        labels_path = model_path.parent / "labels.json"
        if not labels_path.exists():
            return
        try:
            data = json.loads(labels_path.read_text())
            labels = data["labels"] if isinstance(data, dict) else data
            if isinstance(labels, list) and labels:
                self._labels = [str(x) for x in labels]
                if isinstance(data, dict) and data.get("version"):
                    self._model_version = str(data["version"])
                logger.info(f"Loaded {len(self._labels)} labels from {labels_path.name}")

            calibration = data.get("calibration") if isinstance(data, dict) else None
            if isinstance(calibration, dict):
                self._temperature = float(calibration.get("temperature", 1.0)) or 1.0
                rejection = calibration.get("rejection_threshold")
                if rejection is not None:
                    self._rejection_threshold = float(rejection)
                logger.info(
                    f"Calibration: temperature={self._temperature:.3f} "
                    f"rejection_threshold={self._rejection_threshold}"
                )
        except Exception as e:
            logger.warning(f"Failed to read {labels_path}: {e}. Using configured labels.")

    # ── Privacy ───────────────────────────────────────────────

    def enable_privacy(self) -> None:
        """Stop emitting landmark payloads in results."""
        self._privacy_mode = True
        logger.info("Privacy mode enabled for pipeline.")

    def disable_privacy(self) -> None:
        """Resume emitting landmark payloads in results."""
        self._privacy_mode = False
        logger.info("Privacy mode disabled for pipeline.")

    # ── Context manager ───────────────────────────────────────

    def __enter__(self) -> GesturePipeline:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
