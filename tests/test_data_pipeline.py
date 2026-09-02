"""Tests for the dataset extraction, merge, and calibration tooling."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - used at runtime in helpers

import numpy as np
import pytest
import torch

from core.temporal.model import GestureTransformer
from training.datasets.extract_landmarks import _to_bgr
from training.datasets.merge_datasets import load_labels, merge
from training.evaluation.calibrate_confidence import (
    expected_calibration_error,
    fit_temperature,
    sweep_thresholds,
)


def _write_dataset(
    directory: Path, labels: list[str], per_class: int = 4, seed: int = 0
) -> Path:
    """Create a minimal extracted-dataset directory."""
    seq_dir = directory / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    index = 0
    for label_idx in range(len(labels)):
        for _ in range(per_class):
            np.savez_compressed(
                str(seq_dir / f"{index:06d}.npz"),
                landmarks=rng.random((1, 21, 3)).astype(np.float32),
                label=label_idx,
                handedness="right",
            )
            index += 1
    (directory / "metadata.json").write_text(
        json.dumps({"labels": labels, "count": index})
    )
    return directory


class TestImageConversion:
    """_to_bgr must accept every shape the extractor can encounter."""

    def test_accepts_encoded_bytes_dict(self) -> None:
        import cv2

        original = np.full((16, 16, 3), 127, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", original)
        assert ok
        result = _to_bgr({"bytes": buf.tobytes(), "path": None})
        assert result is not None
        assert result.shape == (16, 16, 3)

    def test_accepts_raw_bytes(self) -> None:
        import cv2

        ok, buf = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
        assert ok
        assert _to_bgr(buf.tobytes()) is not None

    def test_accepts_pil_image(self) -> None:
        from PIL import Image

        result = _to_bgr(Image.new("RGB", (12, 10), (255, 0, 0)))
        assert result is not None
        assert result.shape == (10, 12, 3)
        # PIL RGB red becomes BGR (0, 0, 255)
        assert result[0, 0].tolist() == [0, 0, 255]

    def test_rejects_empty_and_unknown(self) -> None:
        assert _to_bgr({"bytes": None}) is None
        assert _to_bgr(None) is None
        assert _to_bgr(np.zeros((5, 5), dtype=np.uint8)) is None


class TestMergeDatasets:
    def test_merges_disjoint_label_sets(self, tmp_path: Path) -> None:
        a = _write_dataset(tmp_path / "a", ["palm", "fist"], per_class=3)
        b = _write_dataset(tmp_path / "b", ["peace"], per_class=2, seed=1)

        metadata = merge([a, b], tmp_path / "out")

        assert metadata["count"] == 8
        assert metadata["labels"] == ["palm", "fist", "peace"]
        assert metadata["per_class_counts"] == {"fist": 3, "palm": 3, "peace": 2}

    def test_shared_labels_are_not_duplicated(self, tmp_path: Path) -> None:
        a = _write_dataset(tmp_path / "a", ["palm", "fist"], per_class=2)
        b = _write_dataset(tmp_path / "b", ["fist", "ok"], per_class=2, seed=2)

        metadata = merge([a, b], tmp_path / "out")

        assert metadata["labels"] == ["palm", "fist", "ok"]
        assert metadata["per_class_counts"]["fist"] == 4

    def test_relabels_indices_consistently(self, tmp_path: Path) -> None:
        """A label at a different local index must land on the merged index."""
        a = _write_dataset(tmp_path / "a", ["palm", "fist"], per_class=1)
        b = _write_dataset(tmp_path / "b", ["fist"], per_class=1, seed=3)

        merge([a, b], tmp_path / "out")
        labels = json.loads((tmp_path / "out" / "metadata.json").read_text())["labels"]
        fist_index = labels.index("fist")

        found = [
            int(np.load(str(f))["label"])
            for f in sorted((tmp_path / "out" / "sequences").glob("*.npz"))
        ]
        assert found.count(fist_index) == 2

    def test_missing_metadata_is_recovered(self, tmp_path: Path) -> None:
        """An interrupted extraction leaves sequences with no metadata.json."""
        d = _write_dataset(tmp_path / "partial", ["a", "b"], per_class=2)
        (d / "metadata.json").unlink()

        files = sorted((d / "sequences").glob("*.npz"))
        labels = load_labels(d, files)

        assert labels == ["class_0", "class_1"]

    def test_label_override(self, tmp_path: Path) -> None:
        d = _write_dataset(tmp_path / "partial", ["x", "y"], per_class=2)
        (d / "metadata.json").unlink()

        metadata = merge([d], tmp_path / "out", label_override=["palm", "fist"])
        assert metadata["labels"] == ["palm", "fist"]


class TestConfidenceCalibration:
    def test_ece_is_zero_for_perfect_calibration(self) -> None:
        # Always predicts class 0 with probability 1.0, and is always right.
        probs = np.tile(np.array([1.0, 0.0]), (50, 1))
        labels = np.zeros(50, dtype=int)
        assert expected_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-6)

    def test_ece_detects_overconfidence(self) -> None:
        # Claims 100% confidence but is right only half the time.
        probs = np.tile(np.array([1.0, 0.0]), (50, 1))
        labels = np.array([0, 1] * 25)
        assert expected_calibration_error(probs, labels) == pytest.approx(0.5, abs=1e-6)

    def test_temperature_softens_overconfident_logits(self) -> None:
        rng = np.random.default_rng(0)
        labels_np = rng.integers(0, 3, size=300)
        # Large logits on a frequently-wrong class => overconfident model.
        logits_np = np.zeros((300, 3), dtype=np.float32)
        for i, label in enumerate(labels_np):
            wrong = (label + 1) % 3 if i % 3 == 0 else label
            logits_np[i, wrong] = 8.0

        temperature = fit_temperature(
            torch.from_numpy(logits_np), torch.from_numpy(labels_np).long()
        )
        assert temperature > 1.0, "overconfident logits should be softened (T > 1)"

    def test_threshold_sweep_trades_coverage_for_accuracy(self) -> None:
        probs = np.array(
            [[0.99, 0.01], [0.95, 0.05], [0.55, 0.45], [0.51, 0.49]],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 1, 1])  # low-confidence rows are the wrong ones

        sweep = sweep_thresholds(probs, labels, thresholds=(0.5, 0.9))
        low, high = sweep[0], sweep[1]

        assert low["coverage"] == 1.0
        assert high["coverage"] == 0.5
        assert high["accuracy"] > low["accuracy"]

    def test_calibration_runs_on_a_real_model(self) -> None:
        from torch.utils.data import TensorDataset

        from training.evaluation.calibrate_confidence import calibrate

        model = GestureTransformer(
            input_dim=86, num_classes=4, d_model=32, nhead=2, num_layers=1
        )
        features = torch.randn(24, 10, 86)
        labels = torch.randint(0, 4, (24,))
        masks = torch.zeros(24, 10, dtype=torch.bool)

        result = calibrate(model, TensorDataset(features, labels, masks), device="cpu")

        assert result.temperature > 0
        assert 0.0 <= result.rejection_threshold <= 1.0
        assert result.ece_after >= 0.0
        assert len(result.threshold_sweep) > 0
        assert set(result.to_dict()) >= {"temperature", "rejection_threshold"}
