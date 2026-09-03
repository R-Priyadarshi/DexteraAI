"""Export one canonical hand pose per gesture class, for the web UI to draw.

The site renders its gesture vocabulary from real captured hands rather than
from icons or emoji. For each class this picks the *medoid* — the real sample
closest to the class centroid — instead of averaging, because an average of
hand poses is not itself an anatomically valid hand.

Output: apps/web/public/data/gesture-atlas.json

    {
      "classes": [
        {"label": "palm", "index": 8, "handedness": "right",
         "landmarks": [[x, y, z], ... 21 points, wrist-centred, unit-scaled]}
      ],
      "connections": [[0, 1], [1, 2], ...]
    }

Usage:
    python scripts/export_gesture_atlas.py \\
        --dataset data/sequences/hagrid --output apps/web/public/data/gesture-atlas.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

from core.types import HAND_CONNECTIONS


def normalise_for_display(landmarks: np.ndarray) -> np.ndarray:
    """Centre a hand on its wrist and scale it to a consistent size.

    MediaPipe returns image-space coordinates, so raw poses carry the position
    and distance of the hand in frame. For a specimen drawing we want only the
    shape: translation and scale removed, orientation kept.
    """
    centred = landmarks - landmarks[0]  # landmark 0 is the wrist
    # Scale by the largest joint distance from the wrist so every plate fills
    # the same box regardless of how close the hand was to the camera.
    extent = float(np.max(np.linalg.norm(centred[:, :2], axis=1)))
    if extent > 1e-6:
        centred = centred / extent
    return centred.astype(np.float32)


def load_class_samples(dataset_dir: Path) -> tuple[list[str], dict[int, list[np.ndarray]]]:
    """Load landmark samples grouped by class index."""
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    labels: list[str] = metadata["labels"]

    by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    handedness: dict[int, list[str]] = defaultdict(list)

    files = sorted((dataset_dir / "sequences").glob("*.npz"))
    logger.info(f"Reading {len(files)} samples from {dataset_dir}")

    for path in files:
        data = np.load(str(path))
        # Static datasets store a single frame; take the first either way.
        frame = data["landmarks"][0]
        label = int(data["label"])
        by_class[label].append(normalise_for_display(frame))
        handedness[label].append(str(data.get("handedness", "right")))

    return labels, by_class, handedness  # type: ignore[return-value]


def frontality(pose: np.ndarray) -> float:
    """How square-on the hand is to the camera. Lower is more frontal.

    A hand rotated away from the lens projects to a tangle of overlapping
    bones — anatomically real, but useless as a specimen drawing. Depth spread
    relative to in-plane size separates the two cases.
    """
    xy_extent = float(np.max(np.linalg.norm(pose[:, :2], axis=1)))
    if xy_extent < 1e-6:
        return float("inf")
    return float(np.std(pose[:, 2]) / xy_extent)


def pick_medoid(
    samples: list[np.ndarray], frontal_quantile: float = 0.45
) -> tuple[np.ndarray, float, int]:
    """Return a canonical pose: the medoid among the most front-facing samples.

    Restricting to front-facing candidates first, then taking the medoid of
    that subset, keeps the result a real observed hand while ensuring the
    plate actually reads as the gesture it names.
    """
    stack = np.stack(samples)  # (N, 21, 3)

    scores = np.array([frontality(p) for p in stack])
    cutoff = float(np.quantile(scores, frontal_quantile))
    keep = np.flatnonzero(scores <= cutoff)
    if len(keep) < 8:  # too few to be meaningful; fall back to everything
        keep = np.arange(len(stack))

    subset = stack[keep]
    centroid = subset.mean(axis=0)
    distances = np.linalg.norm(subset.reshape(len(subset), -1) - centroid.ravel(), axis=1)
    best_local = int(np.argmin(distances))
    return subset[best_local], float(distances[best_local]), len(keep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Extracted landmark dataset directory")
    parser.add_argument("--output", required=True, help="Destination JSON path")
    parser.add_argument(
        "--precision", type=int, default=4, help="Decimal places to keep (keeps the file small)"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    labels, by_class, handedness = load_class_samples(dataset_dir)

    classes = []
    for index, label in enumerate(labels):
        samples = by_class.get(index, [])
        if not samples:
            logger.warning(f"No samples for class '{label}' — skipping")
            continue

        pose, distance, candidates = pick_medoid(samples)
        # Majority handedness, so the drawing is labelled correctly.
        hands = handedness[index]
        dominant = max(set(hands), key=hands.count) if hands else "right"

        classes.append(
            {
                "label": label,
                "index": index,
                "handedness": dominant,
                "sampleCount": len(samples),
                "landmarks": np.round(pose, args.precision).tolist(),
            }
        )
        logger.info(
            f"  {label:<18} n={len(samples):<6} frontal={candidates:<6} "
            f"medoid_dist={distance:.3f}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "source": str(dataset_dir),
                "note": "Medoid pose per class: a real captured hand, not an average.",
                "connections": [list(c) for c in HAND_CONNECTIONS],
                "classes": classes,
            },
            separators=(",", ":"),
        )
    )
    size_kb = output.stat().st_size / 1024
    logger.info(f"Wrote {len(classes)} gesture plates to {output} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
