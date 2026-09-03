"""Merge landmark datasets produced by separate extraction runs.

Extraction is often done in several passes: a first pass over part of a corpus,
a later pass over the remaining shards, or an additional dataset that shares a
label space. This merges those into one directory that `GestureSequenceDataset`
can read, remapping label indices so the merged label list is consistent.

Usage:
    python -m training.datasets.merge_datasets \\
        --inputs data/sequences/hagrid data/sequences/hagrid_tail \\
        --output data/sequences/hagrid_merged
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger


def load_labels(
    dataset_dir: Path,
    sample_files: list[Path],
    fallback: list[str] | None = None,
) -> list[str]:
    """Read a dataset's label list, tolerating a missing metadata.json.

    An interrupted extraction leaves sequences behind without metadata. In that
    case `fallback` names them positionally if supplied, otherwise placeholder
    names are generated so the data is not lost, and the caller is warned.
    """
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        labels = json.loads(metadata_path.read_text()).get("labels", [])
        if labels:
            return [str(label) for label in labels]

    max_label = -1
    for f in sample_files:
        max_label = max(max_label, int(np.load(f)["label"]))
    needed = max_label + 1

    if fallback:
        if len(fallback) < needed:
            raise ValueError(
                f"{dataset_dir} contains {needed} label indices but only "
                f"{len(fallback)} names were supplied via --labels."
            )
        logger.info(f"{dataset_dir} has no metadata.json; using supplied --labels names.")
        return [str(name) for name in fallback[:needed]]

    logger.warning(
        f"{dataset_dir} has no metadata.json; inferring {needed} placeholder labels. "
        "Pass --labels to name them correctly."
    )
    return [f"class_{i}" for i in range(needed)]


def merge(
    input_dirs: list[Path],
    output_dir: Path,
    label_override: list[str] | None = None,
) -> dict[str, object]:
    """Merge several extracted datasets into one directory."""
    out_sequences = output_dir / "sequences"
    out_sequences.mkdir(parents=True, exist_ok=True)

    # label_override names datasets that lack metadata; it does not pre-seed the
    # merged label space, or classes present in no input would appear in output.
    merged_labels: list[str] = []
    label_index: dict[str, int] = {}
    per_class: Counter[str] = Counter()
    written = 0
    sources: list[dict[str, object]] = []

    for dataset_dir in input_dirs:
        files = sorted((dataset_dir / "sequences").glob("*.npz"))
        if not files:
            logger.warning(f"No sequences found in {dataset_dir}, skipping.")
            continue

        labels = load_labels(dataset_dir, files, fallback=label_override)

        # Map this dataset's local label indices onto the merged label space.
        local_to_merged: dict[int, int] = {}
        for local_idx, name in enumerate(labels):
            if name not in label_index:
                label_index[name] = len(merged_labels)
                merged_labels.append(name)
            local_to_merged[local_idx] = label_index[name]

        count = 0
        for f in files:
            data = np.load(f)
            local_label = int(data["label"])
            if local_label not in local_to_merged:
                continue
            merged_label = local_to_merged[local_label]
            np.savez_compressed(
                str(out_sequences / f"{written:06d}.npz"),
                landmarks=data["landmarks"],
                label=merged_label,
                handedness=str(data.get("handedness", "right")),
            )
            per_class[merged_labels[merged_label]] += 1
            written += 1
            count += 1

        sources.append({"path": str(dataset_dir), "samples": count})
        logger.info(f"Merged {count} samples from {dataset_dir}")

    metadata = {
        "labels": merged_labels,
        "count": written,
        "source": "merged",
        "sources": sources,
        "per_class_counts": dict(sorted(per_class.items())),
        "frames_per_sequence": 1,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    empty = [name for name in merged_labels if per_class.get(name, 0) == 0]
    if empty:
        logger.warning(f"Classes with no samples after merge: {empty}")

    logger.info(f"Merged dataset: {written} samples, {len(merged_labels)} classes -> {output_dir}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge extracted landmark datasets.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Dataset directories to merge")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Explicit ordered label names (use when an input lacks metadata.json)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Delete the output directory first"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    merge([Path(p) for p in args.inputs], output_dir, args.labels)


if __name__ == "__main__":
    main()
