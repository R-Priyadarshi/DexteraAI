"""Tests for training.datasets — GestureSequenceDataset and SyntheticGestureDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
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
        """Variable-length sequences (e.g. video) are zero-padded and masked."""
        data_dir = self._create_dummy_dataset(tmp_path, n=5)
        ds = GestureSequenceDataset(data_dir, seq_len=100, expand_static=False)
        features, _, mask = ds[0]
        # Since we set seq_len=100 and data is 15-40 frames,
        # most of the mask should be True (padded)
        assert mask.any()
        # First frame should not be masked
        assert not mask[0].item()

    def test_expand_static_fills_window(self, tmp_path: Path) -> None:
        """Static/short samples are stretched across the window, not padded.

        Training data then has the same shape as the live inference buffer,
        which holds seq_len frames of a held gesture.
        """
        data_dir = self._create_dummy_dataset(tmp_path, n=5)
        ds = GestureSequenceDataset(data_dir, seq_len=100, expand_static=True)
        features, _, mask = ds[0]
        assert not mask.any(), "expanded sequences should have no padded positions"
        assert features.shape == (100, 86)
        assert features.abs().sum() > 0

    def test_get_labels_matches_getitem(self, tmp_path: Path) -> None:
        """The fast label path must agree with the full sample pipeline."""
        data_dir = self._create_dummy_dataset(tmp_path, n=8)
        ds = GestureSequenceDataset(data_dir, seq_len=30)

        fast = ds.get_labels()
        slow = [int(ds[i][1]) for i in range(len(ds))]

        assert fast == slow
        assert len(fast) == len(ds)

    def test_get_labels_respects_indices(self, tmp_path: Path) -> None:
        data_dir = self._create_dummy_dataset(tmp_path, n=10)
        subset = GestureSequenceDataset(data_dir, seq_len=30, indices=[1, 3, 5])
        assert len(subset.get_labels()) == 3

    def test_indices_subset(self, tmp_path: Path) -> None:
        """Index subsetting keeps train/val/test splits from sharing files."""
        data_dir = self._create_dummy_dataset(tmp_path, n=10)
        full = GestureSequenceDataset(data_dir, seq_len=30)
        subset = GestureSequenceDataset(data_dir, seq_len=30, indices=[0, 2, 4])
        assert len(full) == 10
        assert len(subset) == 3

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


class TestImageDirExclusion:
    """`--exclude-classes` on the folder-per-class extractor.

    A corpus can carry a class this single-hand pipeline cannot represent — the
    ASL-HG two-handed sign for "0" is the case this was added for. Training on
    one yields a class that can never fire correctly, which is worse than not
    having it, because it also draws probability mass away from the class it
    resembles ("O").
    """

    @staticmethod
    def _corpus(root: Path) -> Path:
        import cv2

        for name in ("A", "B", "0"):
            class_dir = root / name
            class_dir.mkdir(parents=True)
            cv2.imwrite(str(class_dir / "0.png"), np.zeros((32, 32, 3), dtype=np.uint8))
        return root

    def test_excluded_class_is_dropped(self, tmp_path: Path) -> None:
        from training.datasets.extract_landmarks import iter_image_dir

        labels, _ = iter_image_dir(self._corpus(tmp_path), None, frozenset({"0"}))
        assert labels == ["A", "B"]

    def test_labels_stay_contiguous_after_exclusion(self, tmp_path: Path) -> None:
        """Indices must renumber, not leave a hole.

        Labels are positional against the model's logits, so a gap here would
        silently shift every class after it by one.
        """
        from training.datasets.extract_landmarks import iter_image_dir

        labels, stream = iter_image_dir(self._corpus(tmp_path), None, frozenset({"0"}))
        emitted = sorted({idx for _, idx in stream})
        assert emitted == list(range(len(labels)))

    def test_unknown_class_name_is_rejected(self, tmp_path: Path) -> None:
        """A typo must fail loudly rather than silently exclude nothing."""
        from training.datasets.extract_landmarks import iter_image_dir

        with pytest.raises(ValueError, match="absent from"):
            iter_image_dir(self._corpus(tmp_path), None, frozenset({"zero"}))
