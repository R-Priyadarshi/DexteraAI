"""Confidence calibration and open-set rejection thresholds.

A softmax classifier is closed-set: it always picks one of its trained classes, and
it is usually overconfident while doing it. For a gesture product that matters,
because a user will constantly make hand shapes that are not in the vocabulary, and
reporting a confident wrong label is worse than reporting nothing.

This module provides:

1. **Temperature scaling** — a single scalar fitted on a held-out split that makes
   predicted probabilities match observed accuracy (Guo et al., 2017). It never
   changes which class wins, only how confident the model claims to be.
2. **Threshold selection** — the confidence cut-off below which a prediction is
   reported as "unknown" rather than a label.

This is model confidence calibration. It is unrelated to
`core/calibration/calibrator.py`, which personalizes recognition to an individual
user's hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import Dataset


@dataclass
class CalibrationResult:
    """Outcome of fitting confidence calibration."""

    temperature: float = 1.0
    rejection_threshold: float = 0.5
    ece_before: float = 0.0
    ece_after: float = 0.0
    accuracy: float = 0.0
    coverage_at_threshold: float = 0.0
    accuracy_at_threshold: float = 0.0
    threshold_sweep: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "temperature": round(self.temperature, 4),
            "rejection_threshold": round(self.rejection_threshold, 4),
            "ece_before": round(self.ece_before, 4),
            "ece_after": round(self.ece_after, 4),
            "accuracy": round(self.accuracy, 4),
            "coverage_at_threshold": round(self.coverage_at_threshold, 4),
            "accuracy_at_threshold": round(self.accuracy_at_threshold, 4),
        }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error: mean gap between confidence and accuracy.

    Args:
        probs: (N, C) predicted probabilities.
        labels: (N,) true class indices.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE in [0, 1]. Lower is better; 0 means confidence matches accuracy exactly.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    ece = 0.0
    total = len(labels)
    if total == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_accuracy = float(correct[in_bin].mean())
        bin_confidence = float(confidences[in_bin].mean())
        ece += (count / total) * abs(bin_accuracy - bin_confidence)

    return float(ece)


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    dataset: Dataset[object],
    device: str = "auto",
    batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a dataset and return (logits, labels)."""
    torch_device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    model = model.to(torch_device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        if len(batch) == 3:
            features, labels, masks = batch
            output = model(features.to(torch_device), mask=masks.to(torch_device))
        else:
            features, labels = batch
            output = model(features.to(torch_device))
        logits = output["logits"] if isinstance(output, dict) else output
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.01,
) -> float:
    """Fit a single temperature scalar by minimizing NLL on held-out logits.

    Args:
        logits: (N, C) uncalibrated logits.
        labels: (N,) true class indices.
        max_iter: LBFGS iterations.
        lr: LBFGS learning rate.

    Returns:
        The fitted temperature. > 1 softens overconfident predictions.
    """
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        # Optimize in log-space so temperature stays positive.
        scaled = logits / torch.exp(log_temperature)
        loss = F.cross_entropy(scaled, labels)
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    return float(torch.exp(log_temperature.detach()).item())


def sweep_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> list[dict[str, float]]:
    """Trade coverage against accuracy across candidate rejection thresholds.

    Coverage is the fraction of inputs the model is willing to answer for;
    accuracy is measured only over those accepted inputs.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == labels

    sweep: list[dict[str, float]] = []
    for threshold in thresholds:
        accepted = confidences >= threshold
        n_accepted = int(accepted.sum())
        sweep.append(
            {
                "threshold": float(threshold),
                "coverage": float(n_accepted / max(len(labels), 1)),
                "accuracy": float(correct[accepted].mean()) if n_accepted else 0.0,
            }
        )
    return sweep


def calibrate(
    model: nn.Module,
    val_dataset: Dataset[object],
    device: str = "auto",
    target_accuracy: float = 0.95,
) -> CalibrationResult:
    """Fit temperature scaling and pick a rejection threshold.

    The threshold is the lowest one whose accuracy-on-accepted meets
    `target_accuracy`, which keeps as much coverage as possible while holding
    the promised precision.

    Args:
        model: Trained classifier.
        val_dataset: Held-out data, not used for training.
        device: Torch device or "auto".
        target_accuracy: Accuracy to hold among accepted predictions.

    Returns:
        CalibrationResult with the temperature, threshold, and diagnostics.
    """
    logits, labels = collect_logits(model, val_dataset, device=device)

    probs_before = F.softmax(logits, dim=-1).numpy()
    labels_np = labels.numpy()
    ece_before = expected_calibration_error(probs_before, labels_np)

    temperature = fit_temperature(logits, labels)
    probs_after = F.softmax(logits / temperature, dim=-1).numpy()
    ece_after = expected_calibration_error(probs_after, labels_np)

    sweep = sweep_thresholds(probs_after, labels_np)
    accuracy = float((probs_after.argmax(axis=1) == labels_np).mean())

    # Prefer the most permissive threshold that still meets the accuracy target.
    chosen = next(
        (row for row in sweep if row["accuracy"] >= target_accuracy and row["coverage"] > 0.0),
        None,
    )
    if chosen is None:
        chosen = max(sweep, key=lambda row: row["accuracy"])
        logger.warning(
            f"No threshold reached {target_accuracy:.0%} accuracy; "
            f"falling back to best available ({chosen['accuracy']:.1%} "
            f"at {chosen['threshold']:.2f})."
        )

    result = CalibrationResult(
        temperature=temperature,
        rejection_threshold=chosen["threshold"],
        ece_before=ece_before,
        ece_after=ece_after,
        accuracy=accuracy,
        coverage_at_threshold=chosen["coverage"],
        accuracy_at_threshold=chosen["accuracy"],
        threshold_sweep=sweep,
    )

    logger.info(
        f"Calibration | T={temperature:.3f} | ECE {ece_before:.4f} -> {ece_after:.4f} | "
        f"threshold={result.rejection_threshold:.2f} "
        f"(coverage {result.coverage_at_threshold:.1%}, "
        f"accuracy {result.accuracy_at_threshold:.1%})"
    )
    return result
