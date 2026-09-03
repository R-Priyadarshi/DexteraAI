"""Convert recorded motion packs into a trainable landmark-sequence dataset.

Both shipped models were trained with `frames_per_sequence: 1` — every sample a
still image replicated across the temporal window. The architecture is a
30-frame Transformer, so it can represent motion; it has simply never been
shown any. Genuinely dynamic gestures (swipes, waves, circles, push/pull) have
no class to fall under, and what motion the product exposes today comes from
hand-written velocity thresholds rather than the model.

The obvious public corpus for this is Jester, distributed under CC BY-NC-SA —
a real gate on shipping commercially, not a formality. Recording the data
directly avoids that entirely, at the cost of having to record it. This script
is the bridge: it takes the JSON packs written by the browser's Motion Studio
and emits the `.npz` + `metadata.json` layout `GestureSequenceDataset` reads.

Because the browser records exactly the window the model consumes at inference,
no resampling is needed for same-length clips; clips of a different length are
resampled onto the target window rather than being dropped.

Usage:
    python training/datasets/import_recordings.py \
        --pack ~/Downloads/dextera-motion-*.json \
        --out data/sequences/motion
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

MOTION_PACK_FORMAT = "dextera.motion-pack"
MOTION_PACK_VERSION = 1
LANDMARKS_PER_FRAME = 21


def load_pack(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read one pack, returning its clips and declared frame count."""
    data = json.loads(path.read_text())

    if data.get("format") != MOTION_PACK_FORMAT:
        raise ValueError(f"{path.name}: not a Dextera motion pack")
    if data.get("version") != MOTION_PACK_VERSION:
        raise ValueError(f"{path.name}: unsupported pack version {data.get('version')}")

    clips = data.get("clips")
    if not isinstance(clips, list):
        raise ValueError(f"{path.name}: pack contains no clips")

    return clips, int(data.get("frameCount", 30))


def clip_to_array(clip: dict[str, Any]) -> np.ndarray | None:
    """Convert one clip's frames to (seq_len, 21, 3), or None if malformed.

    Recordings come from a browser and may have been edited or merged by hand,
    so nothing here trusts the file's shape. A malformed clip that reached
    training would not error — it would quietly teach the model noise.
    """
    frames = clip.get("frames")
    if not isinstance(frames, list) or not frames:
        return None

    out = np.zeros((len(frames), LANDMARKS_PER_FRAME, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        if not isinstance(frame, list) or len(frame) != LANDMARKS_PER_FRAME:
            return None
        for j, point in enumerate(frame):
            try:
                out[i, j] = (point["x"], point["y"], point["z"])
            except (KeyError, TypeError, ValueError):
                return None

    if not np.isfinite(out).all():
        return None
    return out


def resample(clip: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a clip onto `target_len` frames by linear interpolation.

    Clips recorded against a different window length are still perfectly good
    demonstrations of the gesture; dropping them would throw away data over a
    detail that costs one interpolation to fix. Interpolating in landmark space
    is safe here because consecutive frames of a tracked hand are close
    together, so the path between them is close to linear.
    """
    n = clip.shape[0]
    if n == target_len:
        return clip
    if n == 1:
        return np.repeat(clip, target_len, axis=0)

    src = np.linspace(0.0, n - 1, num=n)
    dst = np.linspace(0.0, n - 1, num=target_len)
    flat = clip.reshape(n, -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for d in range(flat.shape[1]):
        out[:, d] = np.interp(dst, src, flat[:, d])
    return out.reshape(target_len, LANDMARKS_PER_FRAME, 3)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", nargs="+", required=True, help="Motion pack JSON file(s)")
    p.add_argument("--out", required=True, help="Output dataset directory")
    p.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Frames per sample. Defaults to the frameCount declared in the first pack.",
    )
    p.add_argument(
        "--min-clips",
        type=int,
        default=5,
        help=(
            "Drop labels with fewer clips than this. A class with a handful of "
            "examples inflates the class count without being learnable, and "
            "costs accuracy on the classes that are."
        ),
    )
    p.add_argument(
        "--mirror",
        action="store_true",
        help=(
            "Also emit a left/right mirrored copy of every clip. Doubles the "
            "dataset and makes the model handedness-agnostic — worth it when "
            "recordings come from one person using one hand."
        ),
    )
    args = p.parse_args()

    # Shells expand globs, but an unmatched pattern is passed through literally.
    paths: list[Path] = []
    for pattern in args.pack:
        path = Path(pattern).expanduser()
        if path.exists():
            paths.append(path)
        else:
            matched = sorted(Path(path.parent).glob(path.name))
            if not matched:
                logger.error(f"No pack matched: {pattern}")
                return 1
            paths.extend(matched)

    all_clips: list[dict[str, Any]] = []
    declared_len: int | None = None
    for path in paths:
        clips, frame_count = load_pack(path)
        if declared_len is None:
            declared_len = frame_count
        all_clips.extend(clips)
        logger.info(f"{path.name}: {len(clips)} clips (frameCount={frame_count})")

    seq_len = args.seq_len or declared_len or 30

    # ── Validate and group ────────────────────────────────────
    by_label: dict[str, list[np.ndarray]] = {}
    malformed = 0
    for clip in all_clips:
        label = str(clip.get("label", "")).strip()
        array = clip_to_array(clip)
        if not label or array is None:
            malformed += 1
            continue
        by_label.setdefault(label, []).append(resample(array, seq_len))

    if malformed:
        logger.warning(f"Skipped {malformed} malformed clips")

    # ── Drop under-represented labels ─────────────────────────
    dropped = {
        label: len(clips) for label, clips in by_label.items() if len(clips) < args.min_clips
    }
    for label in dropped:
        del by_label[label]
    if dropped:
        logger.warning(
            f"Dropped {len(dropped)} labels below --min-clips={args.min_clips}: {dropped}"
        )

    if not by_label:
        logger.error("No labels have enough clips to train on.")
        return 1

    # ── Write the dataset ─────────────────────────────────────
    labels = sorted(by_label)
    out_dir = Path(args.out)
    seq_dir = out_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Stale files from an earlier import would be read as part of this dataset
    # while metadata.json describes only the new one.
    for stale in seq_dir.glob("*.npz"):
        stale.unlink()

    index = 0
    written = Counter[str]()
    for label_id, label in enumerate(labels):
        for sequence in by_label[label]:
            variants = [sequence]
            if args.mirror:
                # x is normalised to [0, 1], so mirroring is 1 - x. Only the
                # coordinate is flipped: doing anything to y or z would change
                # the gesture rather than reflect it.
                mirrored = sequence.copy()
                mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]
                variants.append(mirrored)

            for variant in variants:
                np.savez_compressed(
                    seq_dir / f"{index:05d}.npz",
                    landmarks=variant.astype(np.float32),
                    label=np.int64(label_id),
                    handedness="unknown",
                )
                index += 1
                written[label] += 1

    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "count": index,
                "frames_per_sequence": seq_len,
                "source": "motion-pack",
                "mirrored": bool(args.mirror),
                "per_label": dict(written),
            },
            indent=2,
        )
        + "\n"
    )

    logger.success(f"Wrote {index} samples across {len(labels)} classes to {out_dir}")
    for label in labels:
        logger.info(f"  {label}: {written[label]}")

    if min(written.values()) < 20:
        logger.warning(
            "Some classes have under 20 samples. Expect the model to overfit "
            "them; record more clips before trusting the accuracy figure."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
