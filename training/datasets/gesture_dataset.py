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


class GestureSequenceDataset(Dataset):  # type: ignore[type-arg]
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
        expand_static: bool = True,
        indices: list[int] | None = None,
        static_jitter: float = 0.005,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Path to dataset directory.
            seq_len: Fixed sequence length (pad/truncate to this).
            augment: Whether to apply data augmentation.
            augment_config: Augmentation parameters.
            normalization_mode: Landmark normalization strategy.
            expand_static: If True, single-frame samples (from static image
                datasets) are repeated across the full temporal window, with
                independent augmentation jitter per frame when augment=True.
                This matches inference, where the buffer holds seq_len frames
                of a held gesture. If False, short sequences are zero-padded
                and masked instead.
            indices: Optional subset of sample indices, for train/val/test
                splits that must not share underlying files.
            static_jitter: Std-dev of per-frame Gaussian noise added in feature
                space when expanding a held static pose with augment=True.
        """
        self._data_dir = Path(data_dir)
        self._seq_len = seq_len
        self._expand_static = expand_static

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

        if indices is not None:
            self._files = [self._files[i] for i in indices]

        # Components
        self._normalizer = LandmarkNormalizer(normalization_mode)
        self._extractor = LandmarkFeatureExtractor()
        self._augmentor = (
            LandmarkAugmentor(augment_config or AugmentationConfig()) if augment else None
        )
        # Per-frame feature jitter for held static poses (see __getitem__).
        self._static_jitter = static_jitter
        self._rng = np.random.default_rng()

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

    def get_labels(self) -> list[int]:
        """Return every sample's label without running the feature pipeline.

        __getitem__ normalizes and extracts 86-dim features per frame, which is
        wasteful when only the label is needed (class weights, split analysis).
        """
        labels: list[int] = []
        for f in self._files:
            with np.load(str(f)) as data:
                labels.append(int(data["label"]))
        return labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Returns:
            Tuple of:
                - features: (seq_len, feature_dim) float32 tensor
                - label: scalar int64 tensor
                - mask: (seq_len,) bool tensor (True = padded)
        """
        with np.load(str(self._files[idx])) as data:
            raw_landmarks = data["landmarks"]  # (T, 21, 3)
            label = int(data["label"])
            handedness_str = str(data.get("handedness", "right"))
        handedness = Handedness.LEFT if handedness_str == "left" else Handedness.RIGHT

        # Convert each frame to HandLandmarks → normalize → (optionally) augment → extract features
        seq_len_actual = raw_landmarks.shape[0]
        feature_dim = self._extractor.feature_dim

        features = np.zeros((self._seq_len, feature_dim), dtype=np.float32)
        mask = np.ones(self._seq_len, dtype=bool)

        if self._expand_static and seq_len_actual == 1:
            # Static sample: the pose is held for the whole window. Extract features
            # once and tile them, rather than repeating an identical normalize +
            # extract 30 times (which dominated epoch time). Tremor is modelled by
            # small per-frame noise in feature space.
            hand = HandLandmarks(
                landmarks=raw_landmarks[0].astype(np.float32),
                handedness=handedness,
                confidence=1.0,
            )
            hand = self._normalizer.normalize(hand)
            if self._augmentor:
                hand = self._augmentor.augment(hand)
            base = self._extractor.extract(hand)

            features[:] = base
            if self._augmentor:
                features += self._rng.normal(0.0, self._static_jitter, size=features.shape).astype(
                    np.float32
                )
            mask[:] = False
        else:
            if self._expand_static and seq_len_actual < self._seq_len:
                # Stretch a short sequence across the window.
                frame_indices = [
                    min(int(t * seq_len_actual / self._seq_len), seq_len_actual - 1)
                    for t in range(self._seq_len)
                ]
            else:
                frame_indices = list(range(min(seq_len_actual, self._seq_len)))

            for i, t in enumerate(frame_indices[: self._seq_len]):
                hand = HandLandmarks(
                    landmarks=raw_landmarks[t].astype(np.float32),
                    handedness=handedness,
                    confidence=1.0,
                )
                hand = self._normalizer.normalize(hand)
                if self._augmentor:
                    hand = self._augmentor.augment(hand)
                features[i] = self._extractor.extract(hand)
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


class SyntheticGestureDataset(Dataset):  # type: ignore[type-arg]
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
