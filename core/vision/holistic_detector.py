"""Combined hand + body-pose landmark detection for sign language.

Hand shape alone is not enough for sign language. Where a sign is made relative
to the body, and how the hands move through space, carry meaning, so a
hands-only model has a hard accuracy ceiling on anything beyond fingerspelling.

This module pairs the existing `MediaPipeHandDetector` with a MediaPipe Pose
landmarker and returns both in one result, normalized against the signer's
shoulder span so the representation is invariant to distance from the camera
and to body size.

MediaPipe's legacy `Holistic` solution no longer exists in the Tasks API, so this
composes HandLandmarker and PoseLandmarker directly.

Model bundles (fetch once with `make fetch-models`):
    models/mediapipe/hand_landmarker.task
    models/mediapipe/pose_landmarker_lite.task

Scope note: face landmarks are not included. ASL also uses non-manual markers
(eyebrows, mouth morphemes) grammatically, so a complete sign-language system
would add them. See docs/model_card.md for what is and is not implemented.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from core.vision.detector import MediaPipeHandDetector

if TYPE_CHECKING:
    from core.types import HandLandmarks

POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

DEFAULT_POSE_MODEL_PATHS = (
    "models/mediapipe/pose_landmarker_lite.task",
    "~/.cache/dextera/pose_landmarker_lite.task",
)

NUM_POSE_LANDMARKS = 33

# Upper-body landmarks that matter for signing; the legs contribute nothing and
# are dropped to keep the feature vector small.
UPPER_BODY_INDICES = (
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
)

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def resolve_pose_landmarker_path(model_path: str | Path | None = None) -> Path:
    """Locate the pose_landmarker task bundle.

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
    env_path = os.environ.get("DEXTERA_POSE_LANDMARKER")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(Path(p).expanduser() for p in DEFAULT_POSE_MODEL_PATHS)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "MediaPipe pose_landmarker task not found. Run `make fetch-models`, or:\n"
        f"  curl -sL -o {DEFAULT_POSE_MODEL_PATHS[0]} \\\n"
        f"    {POSE_LANDMARKER_URL}\n"
        f"Searched: {', '.join(str(c) for c in candidates)}"
    )


@dataclass(frozen=True, slots=True)
class HolisticResult:
    """Hands plus upper-body pose for one frame.

    Attributes:
        hands: Detected hands (at most two).
        pose: (33, 3) pose landmarks, or None when no body was found.
        shoulder_span: Distance between shoulders in normalized image units,
            used as the scale reference. 0.0 when unavailable.
        inference_time_ms: Combined detection latency.
    """

    hands: list[HandLandmarks] = field(default_factory=list)
    pose: np.ndarray | None = None
    shoulder_span: float = 0.0
    inference_time_ms: float = 0.0

    @property
    def has_pose(self) -> bool:
        return self.pose is not None


class MediaPipeHolisticDetector:
    """Hands + upper-body pose, for sign-language style input.

    Usage:
        >>> detector = MediaPipeHolisticDetector()
        >>> result = detector.detect(bgr_frame)
        >>> len(result.hands), result.has_pose
        >>> detector.close()
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        hand_model_path: str | Path | None = None,
        pose_model_path: str | Path | None = None,
    ) -> None:
        self._static_image_mode = static_image_mode
        self._last_inference_ms = 0.0
        self._frame_index = 0
        self._closed = False
        self._pose_landmarker: Any = None

        self._hand_detector = MediaPipeHandDetector(
            max_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
            model_path=hand_model_path,
        )

        bundle_path = resolve_pose_landmarker_path(pose_model_path)
        running_mode = (
            mp_vision.RunningMode.IMAGE if static_image_mode else mp_vision.RunningMode.VIDEO
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(bundle_path)),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    def detect(self, frame: np.ndarray) -> HolisticResult:
        """Detect hands and upper-body pose in a BGR frame.

        Args:
            frame: BGR image (H, W, 3), dtype uint8.

        Returns:
            HolisticResult with hands, pose, and the shoulder-span scale.
        """
        if self._closed:
            raise RuntimeError("Detector is closed.")
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected BGR frame with shape (H, W, 3), got "
                f"{'None' if frame is None else frame.shape}"
            )

        t_start = time.perf_counter()

        hands = self._hand_detector.detect(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        if self._static_image_mode:
            pose_result = self._pose_landmarker.detect(mp_image)
        else:
            timestamp_ms = int(time.perf_counter() * 1000.0) + self._frame_index
            self._frame_index += 1
            pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        pose: np.ndarray | None = None
        shoulder_span = 0.0
        if pose_result.pose_landmarks:
            pose = np.array(
                [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]],
                dtype=np.float32,
            )
            if pose.shape[0] == NUM_POSE_LANDMARKS:
                shoulder_span = float(
                    np.linalg.norm(pose[LEFT_SHOULDER, :2] - pose[RIGHT_SHOULDER, :2])
                )
            else:
                pose = None

        self._last_inference_ms = (time.perf_counter() - t_start) * 1000.0

        return HolisticResult(
            hands=hands,
            pose=pose,
            shoulder_span=shoulder_span,
            inference_time_ms=self._last_inference_ms,
        )

    def close(self) -> None:
        """Release resources. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._hand_detector.close()
        if self._pose_landmarker is not None:
            with contextlib.suppress(Exception):
                self._pose_landmarker.close()
            self._pose_landmarker = None

    def __enter__(self) -> MediaPipeHolisticDetector:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:  # noqa: SIM105 - contextlib may be None during interpreter shutdown
            self.close()
        except Exception:
            pass
