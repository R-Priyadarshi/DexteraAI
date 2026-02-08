"""Tests for training.datasets — GestureSequenceDataset and SyntheticGestureDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from training.datasets.gesture_dataset import (
    GestureSequenceDataset,
    SyntheticGestureDataset,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestSyntheticGestureDataset:
    """Tests for the synthetic dataset."""

    def test_length(self) -> None:
        ds = SyntheticGestureDataset(num_samples=100)
        assert len(ds) == 100

    def test_shapes(self) -> None:
        ds = SyntheticGestureDataset(num_samples=10, seq_len=20, num_classes=5, feature_dim=86)
        features, label, mask = ds[0]
        assert features.shape == (20, 86)
        assert label.shape == ()
        assert mask.shape == (20,)
        assert features.dtype == torch.float32
        assert label.dtype == torch.long
        assert mask.dtype == torch.bool

    def test_labels_in_range(self) -> None:
        ds = SyntheticGestureDataset(num_samples=500, num_classes=10)
        labels = set()
        for i in range(len(ds)):
            _, label, _ = ds[i]
            labels.add(label.item())
        # All labels should be in [0, 10)
        for label in labels:
            assert 0 <= label < 10

    def test_reproducibility(self) -> None:
        ds1 = SyntheticGestureDataset(num_samples=10, seed=42)
        ds2 = SyntheticGestureDataset(num_samples=10, seed=42)
        f1, l1, m1 = ds1[0]
        f2, l2, m2 = ds2[0]
        torch.testing.assert_close(f1, f2)
        assert l1 == l2

    def test_label_names(self) -> None:
        ds = SyntheticGestureDataset(num_classes=5)
        assert len(ds.label_names) == 5
        assert ds.num_classes == 5

    def test_mask_all_false(self) -> None:
        ds = SyntheticGestureDataset(num_samples=10, seq_len=30)
        _, _, mask = ds[0]
        assert not mask.any()  # Synthetic data is fully filled


class TestGestureSequenceDataset:
    """Tests for the real dataset loader."""

    def _create_dummy_dataset(self, tmp_path: Path, n: int = 20) -> Path:
        """Create a minimal dataset on disk."""
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir(parents=True)

        rng = np.random.default_rng(42)
        labels = ["none", "open_palm", "fist"]

        for i in range(n):
            seq_len = rng.integers(15, 40)
            np.savez_compressed(
                str(seq_dir / f"{i:05d}.npz"),
                landmarks=rng.random((seq_len, 21, 3)).astype(np.float32),
                label=rng.integers(0, len(labels)),
                handedness="right",
            )

        import json

        metadata = {"labels": labels, "count": n}
        with open(tmp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return tmp_path

    def test_load_dataset(self, tmp_path: Path) -> None:
        data_dir = self._create_dummy_dataset(tmp_path)
        ds = GestureSequenceDataset(data_dir, seq_len=30)
        assert len(ds) == 20
        assert ds.num_classes == 3

    def test_getitem_shapes(self, tmp_path: Path) -> None:
        data_dir = self._create_dummy_dataset(tmp_path)
        ds = GestureSequenceDataset(data_dir, seq_len=30)
        features, label, mask = ds[0]
        assert features.shape == (30, 86)
        assert label.dtype == torch.long
        assert mask.shape == (30,)

    def test_padding_mask(self, tmp_path: Path) -> None:
        data_dir = self._create_dummy_dataset(tmp_path, n=5)
        ds = GestureSequenceDataset(data_dir, seq_len=100)
        features, _, mask = ds[0]
        # Since we set seq_len=100 and data is 15-40 frames,
        # most of the mask should be True (padded)
        assert mask.any()
        # First frame should not be masked
        assert not mask[0].item()

    def test_augmentation(self, tmp_path: Path) -> None:
        data_dir = self._create_dummy_dataset(tmp_path, n=5)
        ds_no_aug = GestureSequenceDataset(data_dir, seq_len=30, augment=False)
        ds_aug = GestureSequenceDataset(data_dir, seq_len=30, augment=True)

        f1, _, _ = ds_no_aug[0]
        f2, _, _ = ds_aug[0]

        # Augmented should differ (probabilistically)
        assert not torch.allclose(f1, f2)

    def test_empty_dataset(self, tmp_path: Path) -> None:
        ds = GestureSequenceDataset(tmp_path, seq_len=30)
        assert len(ds) == 0
