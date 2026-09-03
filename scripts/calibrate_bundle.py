"""Fit confidence calibration onto an already-trained model bundle.

Calibration normally happens as the last stage of `dextera.py train --calibrate`.
That is fine for a model you are about to train, but useless for one that is
already on disk: the only way to add calibration would be to retrain, which
throws away hours of compute to fit a single scalar.

This script closes that gap. It reconstructs the exact deterministic validation
split the training run used (same seed, same ratios, same `_split_indices`
logic), loads the checkpoint, fits temperature scaling plus an open-set
rejection threshold, and patches the `calibration` block into the bundle's
`labels.json` in place.

Without calibration a bundle falls back to a generic fixed threshold, and the
softmax is typically overconfident — so near-identical classes (ASL M/N/S/T/A,
for instance) get reported with confidence they have not earned.

Usage:
    python scripts/calibrate_bundle.py \
        --dataset data/sequences/asl_alphabet \
        --checkpoint checkpoints/asl/best.pt \
        --bundle models/asl_alphabet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.temporal.model import GestureTransformer  # noqa: E402
from training.datasets.gesture_dataset import GestureSequenceDataset  # noqa: E402
from training.evaluation.calibrate_confidence import calibrate  # noqa: E402


def split_indices(
    total: int, val_split: float, test_split: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Reproduce `dextera.py::_split_indices` exactly.

    This must stay byte-identical in behaviour to the training-time split, or
    calibration would be fitted on data the model was trained on and the
    resulting temperature would be meaningless.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total).tolist()
    n_test = int(total * test_split)
    n_val = int(total * val_split)
    return perm[n_test + n_val :], perm[n_test : n_test + n_val], perm[:n_test]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="Landmark sequence directory")
    p.add_argument("--checkpoint", required=True, help="Trained .pt checkpoint")
    p.add_argument("--bundle", required=True, help="Model bundle dir holding labels.json")
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--test-split", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fit and report, but leave labels.json untouched.",
    )
    args = p.parse_args()

    bundle = Path(args.bundle)
    labels_path = bundle / "labels.json"
    if not labels_path.exists():
        logger.error(f"No labels.json in {bundle}")
        return 1
    meta = json.loads(labels_path.read_text())
    seq_len = int(meta.get("seq_len", 30))

    # Rebuild the validation split the training run held out.
    probe = GestureSequenceDataset(args.dataset, seq_len=seq_len)
    _, val_idx, _ = split_indices(
        len(probe), args.val_split, args.test_split, args.seed
    )
    val_ds = GestureSequenceDataset(args.dataset, seq_len=seq_len, indices=val_idx)
    logger.info(f"Validation split: {len(val_ds)} samples of {len(probe)} total")

    feature_dim = int(meta.get("feature_dim", 86))
    num_classes = len(meta["labels"])
    model = GestureTransformer(
        input_dim=feature_dim,
        num_classes=num_classes,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        max_seq_len=max(seq_len, 60),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded {args.checkpoint} ({num_classes} classes)")

    result = calibrate(model, val_ds, device=args.device)
    payload = result.to_dict()

    logger.info(
        f"temperature={payload['temperature']} "
        f"threshold={payload['rejection_threshold']} "
        f"ECE {payload['ece_before']} -> {payload['ece_after']}"
    )

    if args.dry_run:
        logger.info("Dry run — labels.json not modified.")
        return 0

    meta["calibration"] = payload
    labels_path.write_text(json.dumps(meta, indent=2) + "\n")
    logger.success(f"Patched calibration into {labels_path}")

    # Keep the checkpoint-side copy in step, so whichever one a consumer reads
    # they see the same numbers.
    ckpt_labels = Path(args.checkpoint).parent / "labels.json"
    if ckpt_labels.exists() and ckpt_labels.resolve() != labels_path.resolve():
        ck = json.loads(ckpt_labels.read_text())
        ck["calibration"] = payload
        ckpt_labels.write_text(json.dumps(ck, indent=2) + "\n")
        logger.success(f"Patched calibration into {ckpt_labels}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
