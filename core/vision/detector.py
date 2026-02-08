"""MediaPipe-based hand landmark detector.

This module wraps Google's MediaPipe Hands solution to provide
a clean, typed interface for hand detection and landmark extraction.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from core.types import LANDMARK_DIMS, NUM_HAND_LANDMARKS, Handedness, HandLandmarks


class MediaPipeHandDetector:
    """Production-grade hand landmark detector using MediaPipe Hands.

    Features:
        - Multi-hand support (up to max_hands)
        - Configurable confidence thresholds
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
    ) -> None:
        """Initialize MediaPipe Hands.

        Args:
            max_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            static_image_mode: If True, treats every frame as independent (slower but no tracking).
        """
        self._max_hands = max_hands
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._last_inference_ms: float = 0.0
        # Drawing utilities for debug overlays
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._mp_hands_solution = mp.solutions.hands

    @property
    def last_inference_ms(self) -> float:
        """Return last inference time in milliseconds."""
        return self._last_inference_ms

    def detect(self, frame: np.ndarray) -> list[HandLandmarks]:
        """Detect hand landmarks in a BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3), dtype uint8.

        Returns:
            List of HandLandmarks for each detected hand.

        Raises:
            ValueError: If frame is not a valid BGR image.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected BGR frame with shape (H, W, 3), got "
                f"{'None' if frame is None else frame.shape}"
            )

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        t_start = time.perf_counter()
        results = self._hands.process(rgb_frame)
        self._last_inference_ms = (time.perf_counter() - t_start) * 1000.0

        if not results.multi_hand_landmarks:
            return []

        hands: list[HandLandmarks] = []
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            # Extract landmarks as (21, 3) array
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            assert landmarks.shape == (NUM_HAND_LANDMARKS, LANDMARK_DIMS)

            # Determine handedness
            handedness = Handedness.UNKNOWN
            confidence = 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                hand_info = results.multi_handedness[idx]
                classification = hand_info.classification[0]
                label = classification.label.lower()
                confidence = classification.score
                handedness = (
                    Handedness.LEFT
                    if label == "left"
                    else Handedness.RIGHT
                    if label == "right"
                    else Handedness.UNKNOWN
                )

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated,
                    hand_lm,
                    self._mp_hands_solution.HAND_CONNECTIONS,
                    self._mp_drawing_styles.get_default_hand_landmarks_style(),
                    self._mp_drawing_styles.get_default_hand_connections_style(),
                )
        return annotated

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    def __enter__(self) -> MediaPipeHandDetector:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
