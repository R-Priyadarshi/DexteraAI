"""Feature extraction from hand landmarks.

Converts raw landmark coordinates into richer feature vectors
including distances, angles, velocities, and derived features
that improve gesture classification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.types import HandLandmarks


class LandmarkFeatureExtractor:
    """Extract high-level features from hand landmarks.

    Features include:
        - Flattened landmark coordinates
        - Pairwise finger distances
        - Joint angles
        - Finger curl ratios
        - Palm orientation

    Usage:
        >>> extractor = LandmarkFeatureExtractor()
        >>> features = extractor.extract(hand_landmarks)
        >>> print(features.shape)  # (feature_dim,)
    """

    # Fingertip indices
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    # MCP (knuckle) indices
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17

    # PIP (middle joint) indices
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18

    WRIST = 0

    FINGERTIPS = [4, 8, 12, 16, 20]
    MCPS = [5, 9, 13, 17]  # excluding thumb

    def extract(self, hand: HandLandmarks) -> np.ndarray:
        """Extract full feature vector from landmarks.

        Args:
            hand: Normalized hand landmarks.

        Returns:
            1D feature vector (np.float32).
        """
        lm = hand.landmarks
        features = []

        # 1. Flattened coordinates (21 * 3 = 63)
        features.append(lm.flatten())

        # 2. Fingertip-to-wrist distances (5)
        features.append(self._fingertip_wrist_distances(lm))

        # 3. Fingertip-to-fingertip distances (10 pairs)
        features.append(self._fingertip_pairwise_distances(lm))

        # 4. Finger curl ratios (5)
        features.append(self._finger_curl_ratios(lm))

        # 5. Palm normal / orientation (3)
        features.append(self._palm_normal(lm))

        return np.concatenate(features).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        """Return the total feature dimension."""
        return 63 + 5 + 10 + 5 + 3  # = 86

    def extract_sequence(self, hands: list[HandLandmarks]) -> np.ndarray:
        """Extract features for a sequence of frames.

        Args:
            hands: List of HandLandmarks (temporal sequence).

        Returns:
            Array of shape (seq_len, feature_dim).
        """
        if not hands:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.stack([self.extract(h) for h in hands])

    # ----- Private feature computations -----

    def _fingertip_wrist_distances(self, lm: np.ndarray) -> np.ndarray:
        """Distance from each fingertip to wrist."""
        return np.array(
            [np.linalg.norm(lm[tip] - lm[self.WRIST]) for tip in self.FINGERTIPS], dtype=np.float32
        )

    def _fingertip_pairwise_distances(self, lm: np.ndarray) -> np.ndarray:
        """Pairwise distances between all fingertips (10 pairs)."""
        tips = lm[self.FINGERTIPS]
        dists = []
        for i in range(len(self.FINGERTIPS)):
            for j in range(i + 1, len(self.FINGERTIPS)):
                dists.append(np.linalg.norm(tips[i] - tips[j]))
        return np.array(dists, dtype=np.float32)

    def _finger_curl_ratios(self, lm: np.ndarray) -> np.ndarray:
        """Curl ratio for each finger (0 = extended, 1 = fully curled).

        Computed as: 1 - (tip-to-mcp distance / pip-to-mcp + tip-to-pip distance).
        """
        fingers = [
            (self.THUMB_TIP, 2, 3),  # thumb: tip, IP, MCP≈2
            (self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            (self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP),
            (self.RING_TIP, self.RING_PIP, self.RING_MCP),
            (self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP),
        ]
        curls = []
        for tip, pip_, mcp in fingers:
            tip_to_mcp = np.linalg.norm(lm[tip] - lm[mcp])
            total_length = np.linalg.norm(lm[mcp] - lm[pip_]) + np.linalg.norm(lm[pip_] - lm[tip])
            if total_length < 1e-6:
                curls.append(0.0)
            else:
                curl = 1.0 - (tip_to_mcp / total_length)
                curls.append(max(0.0, min(1.0, curl)))
        return np.array(curls, dtype=np.float32)

    def _palm_normal(self, lm: np.ndarray) -> np.ndarray:
        """Estimate palm normal vector from wrist, index MCP, and pinky MCP."""
        v1 = lm[self.INDEX_MCP] - lm[self.WRIST]
        v2 = lm[self.PINKY_MCP] - lm[self.WRIST]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return np.zeros(3, dtype=np.float32)
        return (normal / norm).astype(np.float32)
