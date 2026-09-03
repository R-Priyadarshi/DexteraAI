"""Extract MediaPipe hand landmarks from image datasets into trainable sequences.

This is the offline preprocessing stage of the data pipeline: raw images are run
through MediaPipe *once*, and only the resulting 21x3 landmark arrays are stored.
Training then reads landmarks directly, which is both far faster and consistent
with the project's privacy stance (no images retained).

Output layout (consumed by `GestureSequenceDataset`):

    <output_dir>/
    ├── sequences/
    │   ├── 00000.npz   # landmarks: (T, 21, 3), label: int, handedness: str
    │   └── ...
    └── metadata.json   # {"labels": [...], "count": N, "source": ...}

Usage:
    # HuggingFace image-classification dataset (streamed, no full download)
    python -m training.datasets.extract_landmarks \\
        --hf-dataset Jayabalambika/hagrid-classification-512p-dataset \\
        --output data/sequences/hagrid --max-per-class 3000

    # Local folder-per-class layout
    python -m training.datasets.extract_landmarks \\
        --image-dir data/raw/my_gestures --output data/sequences/custom
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp_proc
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from tqdm import tqdm

# Keep worker processes from oversubscribing CPU threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GLOG_minloglevel", "2")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Per-process detector, created lazily so it is never pickled across processes.
_WORKER_DETECTOR: Any = None


def _get_worker_detector() -> Any:
    """Return this process's MediaPipe detector, creating it on first use."""
    global _WORKER_DETECTOR
    if _WORKER_DETECTOR is None:
        from core.vision.detector import MediaPipeHandDetector

        _WORKER_DETECTOR = MediaPipeHandDetector(
            max_hands=1,
            min_detection_confidence=0.3,
            static_image_mode=True,
        )
    return _WORKER_DETECTOR


def _extract_one(task: tuple[int, np.ndarray, int]) -> tuple[int, np.ndarray, int, str] | None:
    """Worker: run MediaPipe on one BGR image.

    Args:
        task: (index, bgr_image, label)

    Returns:
        (index, landmarks (1, 21, 3), label, handedness) or None if no hand found.
    """
    index, bgr, label = task
    try:
        detector = _get_worker_detector()
        hands = detector.detect(bgr)
        if not hands:
            return None
        hand = hands[0]
        return index, hand.landmarks[None, ...], label, hand.handedness.value
    except Exception as e:  # pragma: no cover - defensive, keeps the pool alive
        logger.debug(f"Extraction failed for sample {index}: {e}")
        return None


def _to_bgr(image: Any) -> np.ndarray | None:
    """Convert a PIL image, raw encoded bytes, or array to BGR uint8."""
    import cv2
    from PIL import Image

    # Undecoded HuggingFace image column: {"bytes": ..., "path": ...}
    if isinstance(image, dict):
        raw = image.get("bytes")
        if not raw:
            return None
        buf = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if isinstance(image, (bytes, bytearray)):
        return cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)

    rgb: np.ndarray | None
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        rgb = image if image.ndim == 3 else None
    else:
        return None

    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)


def iter_hf_dataset(
    dataset_id: str,
    split: str,
    max_per_class: int | None,
    image_key: str,
    label_key: str,
    revision: str = "main",
) -> tuple[list[str], Any]:
    """Stream a HuggingFace image-classification dataset.

    Returns:
        (label_names, generator of (bgr_image, label_index))
    """
    from datasets import load_dataset

    logger.info(f"Streaming HuggingFace dataset: {dataset_id} [{split}]")
    # Pinning the revision matters for reproducibility as much as for supply
    # chain: an unpinned dataset id resolves to whatever the hub serves today,
    # so a re-run months later can silently train on different data.
    ds = load_dataset(dataset_id, split=split, streaming=True, revision=revision)

    features = ds.features
    if label_key not in features:
        raise KeyError(f"Label column '{label_key}' not in dataset features: {list(features)}")
    label_feature = features[label_key]
    label_names: list[str] = list(getattr(label_feature, "names", []))
    if not label_names:
        raise ValueError(f"Dataset '{dataset_id}' has no ClassLabel names on '{label_key}'.")

    def generator() -> Any:
        seen: Counter[int] = Counter()
        target_total = max_per_class * len(label_names) if max_per_class else None
        for row in ds:
            label = int(row[label_key])
            if max_per_class is not None and seen[label] >= max_per_class:
                # Stop early once every class has hit the cap.
                if target_total is not None and sum(seen.values()) >= target_total:
                    break
                continue
            bgr = _to_bgr(row[image_key])
            if bgr is None:
                continue
            seen[label] += 1
            yield bgr, label

    return label_names, generator()


def iter_image_dir(
    image_dir: Path,
    max_per_class: int | None,
    exclude: frozenset[str] = frozenset(),
) -> tuple[list[str], Any]:
    """Iterate a folder-per-class image directory.

    `exclude` drops class directories by name. Corpora sometimes carry a class
    this pipeline cannot represent - a two-handed sign, say, against an 86-dim
    single-hand feature vector - and training on one produces a class that can
    never fire correctly rather than an error.
    """
    import cv2

    class_dirs = sorted(d for d in image_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise ValueError(f"No class subdirectories found under {image_dir}")

    if exclude:
        present = {d.name for d in class_dirs}
        missing = exclude - present
        if missing:
            raise ValueError(
                f"--exclude-classes names {sorted(missing)}, absent from {image_dir}. "
                f"Available: {sorted(present)}"
            )
        class_dirs = [d for d in class_dirs if d.name not in exclude]
        logger.info(f"Excluding {len(exclude)} class(es): {sorted(exclude)}")

    label_names = [d.name for d in class_dirs]
    logger.info(f"Found {len(label_names)} classes in {image_dir}")

    def generator() -> Any:
        for label_idx, class_dir in enumerate(class_dirs):
            files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
            if max_per_class is not None:
                files = files[:max_per_class]
            for f in files:
                bgr = cv2.imread(str(f))
                if bgr is not None:
                    yield bgr, label_idx

    return label_names, generator()


def iter_parquet(
    parquet_paths: list[Path],
    max_per_class: int | None,
    image_key: str,
    label_key: str,
    label_names: list[str] | None = None,
) -> tuple[list[str], Any]:
    """Iterate local parquet shards produced by a HuggingFace dataset export.

    Shards are read one at a time so memory stays flat, and each shard is
    capped independently. Datasets whose rows are grouped by class (HaGRID is)
    therefore still yield a balanced sample without reading everything.
    """
    from datasets import load_dataset

    if label_names is None:
        # Suppressed below: reads a local file through the parquet loader; there is
        # no hub download here and so nothing to pin.
        ds_probe = load_dataset(  # nosec B615
            "parquet", data_files=str(parquet_paths[0]), split="train", streaming=True
        )
        label_feature = ds_probe.features.get(label_key)
        label_names = list(getattr(label_feature, "names", []))
        if not label_names:
            raise ValueError(
                f"Parquet files have no ClassLabel names on '{label_key}'. "
                "Pass --label-names explicitly."
            )

    def generator() -> Any:
        seen: Counter[int] = Counter()
        from datasets import Image as HFImage

        for shard in parquet_paths:
            # Suppressed below: local file, as above.
            ds = load_dataset(  # nosec B615
                "parquet", data_files=str(shard), split="train", streaming=True
            )
            # Skip PIL decoding: rows over the per-class cap are discarded without
            # ever paying for a decode, which dominates runtime on large datasets.
            with contextlib.suppress(Exception):
                ds = ds.cast_column(image_key, HFImage(decode=False))
            for row in ds:
                label = int(row[label_key])
                if max_per_class is not None and seen[label] >= max_per_class:
                    continue
                bgr = _to_bgr(row[image_key])
                if bgr is None:
                    continue
                seen[label] += 1
                yield bgr, label

    return label_names, generator()


def extract(
    source_iter: Any,
    label_names: list[str],
    output_dir: Path,
    num_workers: int,
    source_name: str,
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Run extraction over a stream of (bgr, label) pairs and write the dataset."""
    seq_dir = output_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    attempted = 0
    per_class: Counter[int] = Counter()
    t_start = time.time()

    def tasks() -> Any:
        for i, (bgr, label) in enumerate(source_iter):
            yield i, bgr, label

    # Only animate when attached to a terminal; keeps CI/log output readable.
    progress = tqdm(
        desc="Extracting landmarks",
        unit="img",
        disable=not sys.stderr.isatty(),
        mininterval=2.0,
    )

    if num_workers <= 1:
        results_iter: Any = map(_extract_one, tasks())
        pool = None
    else:
        ctx = mp_proc.get_context("spawn")
        pool = ctx.Pool(processes=num_workers)
        results_iter = pool.imap_unordered(_extract_one, tasks(), chunksize=chunk_size)

    try:
        for result in results_iter:
            attempted += 1
            progress.update(1)
            if result is None:
                continue
            _, landmarks, label, handedness = result
            np.savez_compressed(
                str(seq_dir / f"{written:06d}.npz"),
                landmarks=landmarks.astype(np.float32),
                label=label,
                handedness=handedness,
            )
            written += 1
            per_class[label] += 1
            progress.set_postfix(kept=written, detect_rate=f"{written / max(attempted, 1):.0%}")
            if not sys.stderr.isatty() and written % 2000 == 0:
                logger.info(
                    f"  ... {written} kept / {attempted} seen "
                    f"({written / max(attempted, 1):.0%} detected)"
                )
    finally:
        progress.close()
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - t_start
    metadata = {
        "labels": label_names,
        "count": written,
        "source": source_name,
        "attempted": attempted,
        "detection_rate": round(written / max(attempted, 1), 4),
        "per_class_counts": {label_names[k]: v for k, v in sorted(per_class.items())},
        "extraction_seconds": round(elapsed, 1),
        "frames_per_sequence": 1,
        "note": (
            "Single-frame landmark samples. GestureSequenceDataset expands these to the "
            "model's temporal window at load time (expand_static=True)."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info(
        f"Wrote {written}/{attempted} samples ({metadata['detection_rate']:.1%} detection rate) "
        f"to {output_dir} in {elapsed:.1f}s"
    )
    if per_class:
        missing = [n for i, n in enumerate(label_names) if per_class.get(i, 0) == 0]
        if missing:
            logger.warning(f"No samples extracted for classes: {missing}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks into a trainable dataset."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-dataset", type=str, help="HuggingFace dataset id")
    source.add_argument("--image-dir", type=str, help="Local folder-per-class image directory")
    source.add_argument(
        "--parquet-dir",
        type=str,
        help="Directory of local .parquet shards (HuggingFace export format)",
    )

    parser.add_argument("--output", type=str, required=True, help="Output dataset directory")
    parser.add_argument("--split", type=str, default="train", help="HF split name")
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help=(
            "HF dataset revision (branch, tag or commit). Pin a commit for a "
            "reproducible extraction: 'main' resolves to whatever the hub serves "
            "today, so a re-run later can silently use different data."
        ),
    )
    parser.add_argument("--image-key", type=str, default="image", help="HF image column")
    parser.add_argument("--label-key", type=str, default="label", help="HF label column")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap samples per class (keeps runs fast and classes balanced)",
    )
    parser.add_argument(
        "--exclude-classes",
        type=str,
        default="",
        help=(
            "Comma-separated class directory names to skip (--image-dir only). "
            "Use for classes this single-hand pipeline cannot represent."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="Parallel extraction workers",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_dataset:
        label_names, stream = iter_hf_dataset(
            args.hf_dataset,
            args.split,
            args.max_per_class,
            args.image_key,
            args.label_key,
            args.revision,
        )
        source_name = f"hf:{args.hf_dataset}:{args.split}@{args.revision}"
    elif args.parquet_dir:
        shards = sorted(Path(args.parquet_dir).glob("*.parquet"))
        if not shards:
            raise SystemExit(f"No .parquet files found in {args.parquet_dir}")
        logger.info(f"Reading {len(shards)} parquet shard(s) from {args.parquet_dir}")
        label_names, stream = iter_parquet(
            shards, args.max_per_class, args.image_key, args.label_key
        )
        source_name = f"parquet:{args.parquet_dir}"
    else:
        excluded = frozenset(c.strip() for c in args.exclude_classes.split(",") if c.strip())
        label_names, stream = iter_image_dir(Path(args.image_dir), args.max_per_class, excluded)
        source_name = f"dir:{args.image_dir}"
        if excluded:
            source_name += f" (excluding {','.join(sorted(excluded))})"

    logger.info(f"Classes ({len(label_names)}): {label_names}")
    extract(stream, label_names, output_dir, args.workers, source_name)


if __name__ == "__main__":
    main()
