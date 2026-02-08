"""End-to-end gesture recognition pipeline.

Orchestrates the full flow: frame → preprocessing → hand detection →
landmark normalization → feature extraction → temporal buffering →
gesture classification → result output.

This is the main entry point for all applications (web, mobile, desktop).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from core.landmarks.features import LandmarkFeatureExtractor
from core.landmarks.normalizer import LandmarkNormalizer, NormalizationMode
from core.temporal.model import GestureTransformer
from core.temporal.sequence_buffer import SequenceBuffer
from core.types import FrameResult, GestureResult, GestureState, HandLandmarks
from core.vision.detector import MediaPipeHandDetector
from core.vision.preprocessor import FramePreprocessor, PreprocessConfig


@dataclass
class PipelineConfig:
    """Configuration for the gesture pipeline.

    Attributes:
        max_hands: Maximum number of hands to track.
        sequence_length: Number of frames in temporal window.
        confidence_threshold: Minimum confidence to report a gesture.
        model_path: Path to trained gesture model (ONNX or .pt).
        gesture_labels: Ordered list of gesture class names.
        use_gpu: Whether to use GPU for inference.
        normalization_mode: Landmark normalization strategy.
        preprocess_config: Frame preprocessing settings.
    """
    max_hands: int = 2
    sequence_length: int = 30
    confidence_threshold: float = 0.6
    model_path: str | None = None
    gesture_labels: list[str] = field(default_factory=lambda: [
        "none", "open_palm", "closed_fist", "thumbs_up", "thumbs_down",
        "peace", "pointing_up", "ok_sign", "pinch", "wave",
    ])
    use_gpu: bool = False
    normalization_mode: NormalizationMode = NormalizationMode.FULL
    preprocess_config: PreprocessConfig = field(default_factory=PreprocessConfig)


class GesturePipeline:
    """Production gesture recognition pipeline.

    Combines all core modules into a single, easy-to-use interface.
    Handles multi-hand tracking, temporal buffering, and classification.

    Usage:
        >>> pipeline = GesturePipeline(PipelineConfig())
        >>> pipeline.start()
        >>>
        >>> # In your frame loop:
        >>> result = pipeline.process_frame(bgr_frame)
        >>> for gesture in result.gestures:
        ...     print(f"{gesture.gesture_name}: {gesture.confidence:.2f}")
        >>>
        >>> pipeline.stop()
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._preprocessor: FramePreprocessor | None = None
        self._detector: MediaPipeHandDetector | None = None
        self._normalizer: LandmarkNormalizer | None = None
        self._feature_extractor: LandmarkFeatureExtractor | None = None
        self._buffers: dict[int, SequenceBuffer] = {}  # per-hand buffers
        self._model: GestureTransformer | None = None
        self._device = torch.device("cuda" if self._config.use_gpu and torch.cuda.is_available() else "cpu")
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def start(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting gesture pipeline...")

        self._preprocessor = FramePreprocessor(self._config.preprocess_config)
        self._detector = MediaPipeHandDetector(max_hands=self._config.max_hands)
        self._normalizer = LandmarkNormalizer(mode=self._config.normalization_mode)
        self._feature_extractor = LandmarkFeatureExtractor()

        # Initialize per-hand buffers
        for i in range(self._config.max_hands):
            self._buffers[i] = SequenceBuffer(
                max_len=self._config.sequence_length,
                feature_extractor=self._feature_extractor,
            )

        # Load model if available
        if self._config.model_path:
            self._load_model(self._config.model_path)

        self._is_running = True
        logger.info(
            f"Pipeline started | Device: {self._device} | "
            f"Max hands: {self._config.max_hands} | "
            f"Seq length: {self._config.sequence_length}"
        )

    def stop(self) -> None:
        """Release all resources."""
        if self._detector:
            self._detector.close()
        self._buffers.clear()
        self._model = None
        self._is_running = False
        logger.info("Pipeline stopped.")

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Process a single BGR frame through the full pipeline.

        Args:
            frame: BGR image (H, W, 3), dtype uint8.

        Returns:
            FrameResult with detected hands and recognized gestures.

        Raises:
            RuntimeError: If pipeline is not started.
        """
        if not self._is_running:
            raise RuntimeError("Pipeline not started. Call start() first.")

        t_start = time.perf_counter()

        # 1. Preprocess
        processed = self._preprocessor.process(frame)

        # 2. Detect hands
        hands = self._detector.detect(processed)

        # 3. Normalize landmarks
        normalized_hands = self._normalizer.normalize_batch(hands)

        # 4. Buffer features and classify
        gestures: list[GestureResult] = []
        for i, hand in enumerate(normalized_hands[:self._config.max_hands]):
            self._buffers[i].push(hand)

            if self._model and self._buffers[i].is_ready:
                gesture = self._classify_gesture(i, hand)
                if gesture:
                    gestures.append(gesture)

        # Push empty frames for undetected hands
        for i in range(len(normalized_hands), self._config.max_hands):
            self._buffers[i].push_empty()

        inference_ms = (time.perf_counter() - t_start) * 1000.0

        return FrameResult(
            hands=normalized_hands,
            gestures=gestures,
            timestamp_ms=time.time() * 1000.0,
            inference_time_ms=inference_ms,
        )

    def _classify_gesture(self, hand_idx: int, hand: HandLandmarks) -> GestureResult | None:
        """Run gesture classification on buffered sequence."""
        features, mask = self._buffers[hand_idx].get_padded_features()

        # Convert to tensor
        x = torch.from_numpy(features).unsqueeze(0).to(self._device)  # (1, S, F)
        m = torch.from_numpy(mask).unsqueeze(0).to(self._device)      # (1, S)

        result = self._model.predict(x, mask=m)

        class_id = result["class_id"].item()
        confidence = result["confidence"].item()
        probs = result["class_probs"][0]

        # Check threshold
        class_prob = probs[class_id].item()
        if class_prob < self._config.confidence_threshold:
            return None

        gesture_name = (
            self._config.gesture_labels[class_id]
            if class_id < len(self._config.gesture_labels)
            else f"gesture_{class_id}"
        )

        return GestureResult(
            gesture_id=class_id,
            gesture_name=gesture_name,
            confidence=class_prob,
            state=GestureState.RECOGNIZED,
            landmarks=hand,
        )

    def _load_model(self, model_path: str) -> None:
        """Load a trained gesture model."""
        path = Path(model_path)
        if not path.exists():
            logger.warning(f"Model not found: {path}. Running in detection-only mode.")
            return

        num_classes = len(self._config.gesture_labels)
        feature_dim = LandmarkFeatureExtractor().feature_dim

        self._model = GestureTransformer(
            input_dim=feature_dim,
            num_classes=num_classes,
            d_model=128,
            nhead=4,
            num_layers=4,
        ).to(self._device)

        if path.suffix == ".pt":
            state_dict = torch.load(str(path), map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            logger.info(f"Loaded PyTorch model: {path.name} ({self._model.get_model_size_mb():.2f} MB)")
        else:
            logger.warning(f"Unsupported model format: {path.suffix}. Use ONNX runtime for .onnx files.")

    def __enter__(self) -> GesturePipeline:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
