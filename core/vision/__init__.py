"""Vision module — MediaPipe hand detection and frame preprocessing."""

from core.vision.detector import MediaPipeHandDetector
from core.vision.preprocessor import FramePreprocessor

__all__ = ["MediaPipeHandDetector", "FramePreprocessor"]
