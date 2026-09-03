"""MediaPipe-based hand landmark detector.

Wraps MediaPipe's Tasks HandLandmarker to provide a clean, typed interface
for hand detection and landmark extraction.

Note: MediaPipe >= 0.10.30 removed the legacy `mp.solutions` API, so this
module targets `mediapipe.tasks.python.vision.HandLandmarker`, which needs a
`.task` model bundle on disk. Fetch it once with:

    make fetch-models

or manually:

    curl -sL -o models/mediapipe/hand_landmarker.task \\
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from core.types import (
    HAND_CONNECTIONS,
    LANDMARK_DIMS,
    NUM_HAND_LANDMARKS,
    Handedness,
    HandLandmarks,
)

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

DEFAULT_MODEL_PATHS = (
    "models/mediapipe/hand_landmarker.task",
    "~/.cache/dextera/hand_landmarker.task",
)


def resolve_hand_landmarker_path(model_path: str | Path | None = None) -> Path:
    """Locate the hand_landmarker.task bundle.

    Resolution order: explicit argument, `DEXTERA_HAND_LANDMARKER` env var,
    then the well-known default locations.

    Raises:
        FileNotFoundError: With download instructions if no bundle is found.
    """
    # An explicit path is authoritative: falling back to a different bundle
    # would silently run a model the caller did not ask for.
    if model_path:
        explicit = Path(model_path).expanduser()
        if explicit.is_file():
            return explicit
        raise FileNotFoundError(f"Model bundle not found at the given path: {explicit}")

    candidates: list[Path] = []
    env_path = os.environ.get("DEXTERA_HAND_LANDMARKER")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(Path(p).expanduser() for p in DEFAULT_MODEL_PATHS)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "MediaPipe hand_landmarker.task not found. Run `make fetch-models`, or:\n"
        f"  mkdir -p models/mediapipe && curl -sL -o {DEFAULT_MODEL_PATHS[0]} \\\n"
        f"    {HAND_LANDMARKER_URL}\n"
        f"Searched: {', '.join(str(c) for c in candidates)}"
    )


class MediaPipeHandDetector:
    """Hand landmark detector using MediaPipe Tasks.

    Features:
        - Multi-hand support (up to max_hands)
        - Configurable confidence thresholds
        - Video mode with cross-frame tracking, or independent image mode
        - Automatic resource cleanup
        - Performance instrumentation

    Usage:
        >>> detector = MediaPipeHandDetector(max_hands=2)
        >>> hands = detector.detect(bgr_frame)
        >>> for hand in hands:
        ...     print(hand.landmarks.shape, hand.handedness, hand.confidence)
        >>> detector.close()
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_path: str | Path | None = None,
    ) -> None:
        """Initialize the hand landmarker.

        Args:
            max_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            static_image_mode: If True, every frame is treated independently
                (slower, no tracking). If False, uses video mode with tracking.
            model_path: Optional explicit path to hand_landmarker.task.
        """
        self._max_hands = max_hands
        self._static_image_mode = static_image_mode
        self._last_inference_ms: float = 0.0
        self._frame_index = 0
        self._closed = False
        self._landmarker: Any = None

        bundle_path = resolve_hand_landmarker_path(model_path)

        running_mode = (
            mp_vision.RunningMode.IMAGE if static_image_mode else mp_vision.RunningMode.VIDEO
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(bundle_path)),
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    @property
    def last_inference_ms(self) -> float:
        """Return last inference time in milliseconds."""
        return self._last_inference_ms

    @property
    def max_hands(self) -> int:
        return self._max_hands

    @staticmethod
    def _to_handedness(label: str) -> Handedness:
        """Map a MediaPipe handedness label to the internal enum."""
        normalized = label.strip().lower()
        if normalized == "left":
            return Handedness.LEFT
        if normalized == "right":
            return Handedness.RIGHT
        return Handedness.UNKNOWN

    def detect(self, frame: np.ndarray) -> list[HandLandmarks]:
        """Detect hand landmarks in a BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3), dtype uint8.

        Returns:
            List of HandLandmarks, one per detected hand.

        Raises:
            ValueError: If frame is not a valid BGR image.
            RuntimeError: If the detector has been closed.
        """
        if self._closed:
            raise RuntimeError("Detector is closed.")
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected BGR frame with shape (H, W, 3), got "
                f"{'None' if frame is None else frame.shape}"
            )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))

        t_start = time.perf_counter()
        if self._static_image_mode:
            result = self._landmarker.detect(mp_image)
        else:
            # Video mode needs monotonically increasing millisecond timestamps.
            timestamp_ms = int(time.perf_counter() * 1000.0) + self._frame_index
            self._frame_index += 1
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        self._last_inference_ms = (time.perf_counter() - t_start) * 1000.0

        if not result.hand_landmarks:
            return []

        hands: list[HandLandmarks] = []
        for idx, hand_lms in enumerate(result.hand_landmarks):
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms],
                dtype=np.float32,
            )
            if landmarks.shape != (NUM_HAND_LANDMARKS, LANDMARK_DIMS):
                continue

            handedness = Handedness.UNKNOWN
            confidence = 0.0
            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                category = result.handedness[idx][0]
                handedness = self._to_handedness(category.category_name)
                confidence = float(category.score)

            hands.append(
                HandLandmarks(
                    landmarks=landmarks,
                    handedness=handedness,
                    confidence=confidence,
                )
            )

        return hands

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hands: list[HandLandmarks] | None = None,
    ) -> np.ndarray:
        """Draw hand landmarks on a BGR frame for debug/visualization.

        If `hands` is None, runs detection internally (slower).

        Args:
            frame: BGR image (H, W, 3).
            hands: Optional pre-detected hands to skip re-detection.

        Returns:
            Annotated BGR image copy.
        """
        annotated = frame.copy()
        detected = self.detect(frame) if hands is None else hands
        height, width = annotated.shape[:2]

        for hand in detected:
            points = [(int(lm[0] * width), int(lm[1] * height)) for lm in hand.landmarks]
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(annotated, points[start_idx], points[end_idx], (0, 255, 0), 2)
            for point in points:
                cv2.circle(annotated, point, 3, (0, 0, 255), -1)

        return annotated

    def close(self) -> None:
        """Release MediaPipe resources. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            with contextlib.suppress(Exception):  # defensive cleanup
                landmarker.close()
            self._landmarker = None

    def __enter__(self) -> MediaPipeHandDetector:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Bare try/except, not contextlib.suppress: during interpreter shutdown
        # module globals (including contextlib) may already be None.
        # Suppressed below: a destructor cannot meaningfully handle a failure and
        # must not raise during interpreter shutdown, when the exception would
        # be printed and ignored anyway.
        try:  # noqa: SIM105 - contextlib may be None during interpreter shutdown
            self.close()
        except Exception:  # nosec B110
            pass
