"""Gesture sequence dataset for PyTorch training.

Supports:
    - Loading pre-extracted landmark sequences from disk
    - On-the-fly landmark extraction from video
    - Augmentation integration
    - Variable-length sequences with padding + masks
    - DVC-compatible data paths
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from core.landmarks.augmentor import AugmentationConfig, LandmarkAugmentor
from core.landmarks.features import LandmarkFeatureExtractor
from core.landmarks.normalizer import LandmarkNormalizer, NormalizationMode
from core.types import Handedness, HandLandmarks


class GestureSequenceDataset(Dataset):
    """PyTorch Dataset for gesture recognition training.

    Expected data format on disk:
        data_dir/
        ├── sequences/
        │   ├── 00000.npz   # landmarks: (seq_len, 21, 3), label: int
        │   ├── 00001.npz
        │   └── ...
        └── metadata.json   # {"labels": ["none", "open_palm", ...], "count": N}

    Each .npz file contains:
        - "landmarks": np.ndarray of shape (seq_len, 21, 3)
        - "label": int (gesture class index)
        - "handedness": str ("left" | "right")

    Usage:
        >>> dataset = GestureSequenceDataset("data/hagrid", seq_len=30)
        >>> features, label, mask = dataset[0]
        >>> print(features.shape)  # (30, 86)
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int = 30,
        augment: bool = False,
        augment_config: AugmentationConfig | None = None,
        normalization_mode: NormalizationMode = NormalizationMode.FULL,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Path to dataset directory.
            seq_len: Fixed sequence length (pad/truncate to this).
            augment: Whether to apply data augmentation.
            augment_config: Augmentation parameters.
            normalization_mode: Landmark normalization strategy.
        """
        self._data_dir = Path(data_dir)
        self._seq_len = seq_len

        # Load metadata
        metadata_path = self._data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self._label_names: list[str] = metadata["labels"]
        else:
            self._label_names = [
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

        # Discover sequence files
        seq_dir = self._data_dir / "sequences"
        if seq_dir.exists():
            self._files = sorted(seq_dir.glob("*.npz"))
        else:
            self._files = []

        # Components
        self._normalizer = LandmarkNormalizer(normalization_mode)
        self._extractor = LandmarkFeatureExtractor()
        self._augmentor = (
            LandmarkAugmentor(augment_config or AugmentationConfig()) if augment else None
        )

    @property
    def label_names(self) -> list[str]:
        """Return ordered list of gesture class names."""
        return self._label_names

    @property
    def num_classes(self) -> int:
        return len(self._label_names)

    @property
    def feature_dim(self) -> int:
        return self._extractor.feature_dim

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Returns:
            Tuple of:
                - features: (seq_len, feature_dim) float32 tensor
                - label: scalar int64 tensor
                - mask: (seq_len,) bool tensor (True = padded)
        """
        data = np.load(str(self._files[idx]))
        raw_landmarks = data["landmarks"]  # (T, 21, 3)
        label = int(data["label"])
        handedness_str = str(data.get("handedness", "right"))
        handedness = Handedness.LEFT if handedness_str == "left" else Handedness.RIGHT

        # Convert each frame to HandLandmarks → normalize → (optionally) augment → extract features
        seq_len_actual = raw_landmarks.shape[0]
        features_list: list[np.ndarray] = []

        for t in range(seq_len_actual):
            hand = HandLandmarks(
                landmarks=raw_landmarks[t].astype(np.float32),
                handedness=handedness,
                confidence=1.0,
            )
            hand = self._normalizer.normalize(hand)
            if self._augmentor:
                hand = self._augmentor.augment(hand)
            features_list.append(self._extractor.extract(hand))

        # Pad or truncate to fixed seq_len
        features = np.zeros((self._seq_len, self._extractor.feature_dim), dtype=np.float32)
        mask = np.ones(self._seq_len, dtype=bool)  # True = padded

        actual = min(seq_len_actual, self._seq_len)
        for i in range(actual):
            features[i] = features_list[i]
            mask[i] = False

        return (
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.long),
            torch.from_numpy(mask),
        )

    @staticmethod
    def create_from_landmarks(
        output_dir: str | Path,
        sequences: list[dict[str, Any]],
        label_names: list[str],
    ) -> None:
        """Create a dataset from pre-computed landmark sequences.

        Args:
            output_dir: Directory to write the dataset to.
            sequences: List of dicts, each with:
                - "landmarks": np.ndarray (seq_len, 21, 3)
                - "label": int
                - "handedness": str
            label_names: Ordered list of gesture class names.
        """
        output_dir = Path(output_dir)
        seq_dir = output_dir / "sequences"
        seq_dir.mkdir(parents=True, exist_ok=True)

        for i, seq in enumerate(sequences):
            np.savez_compressed(
                str(seq_dir / f"{i:05d}.npz"),
                landmarks=seq["landmarks"],
                label=seq["label"],
                handedness=seq.get("handedness", "right"),
            )

        metadata = {"labels": label_names, "count": len(sequences)}
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


class SyntheticGestureDataset(Dataset):
    """Synthetic dataset for testing and development.

    Generates random landmark sequences with known labels.
    Useful for verifying the training pipeline before real data.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 30,
        num_classes: int = 10,
        feature_dim: int = 86,
        seed: int = 42,
    ) -> None:
        self._num_samples = num_samples
        self._seq_len = seq_len
        self._num_classes = num_classes
        self._feature_dim = feature_dim
        self._rng = np.random.default_rng(seed)

        # Pre-generate data for speed
        self._features = self._rng.standard_normal((num_samples, seq_len, feature_dim)).astype(
            np.float32
        )
        self._labels = self._rng.integers(0, num_classes, size=num_samples)

    @property
    def label_names(self) -> list[str]:
        return [f"gesture_{i}" for i in range(self._num_classes)]

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.from_numpy(self._features[idx])
        label = torch.tensor(self._labels[idx], dtype=torch.long)
        mask = torch.zeros(self._seq_len, dtype=torch.bool)
        return features, label, mask
