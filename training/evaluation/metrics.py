"""Evaluation metrics for gesture recognition models.

Provides comprehensive evaluation including:
    - Per-class precision, recall, F1
    - Confusion matrix
    - Top-k accuracy
    - Latency benchmarking
    - Model card generation
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from core.temporal.model import GestureTransformer
    from training.datasets.gesture_dataset import GestureSequenceDataset


@dataclass
class EvalResult:
    """Comprehensive evaluation results.

    Attributes:
        accuracy: Overall accuracy.
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        f1_macro: Macro-averaged F1 score.
        per_class_report: Dict with per-class metrics.
        confusion_matrix: (num_classes, num_classes) array.
        top_k_accuracy: Dict mapping k → top-k accuracy.
        avg_latency_ms: Average per-sample inference latency.
        p95_latency_ms: 95th percentile latency.
        p99_latency_ms: 99th percentile latency.
        total_samples: Number of evaluation samples.
    """

    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    per_class_report: dict = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    top_k_accuracy: dict[int, float] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_samples: int = 0


class GestureEvaluator:
    """Evaluate gesture recognition models.

    Usage:
        >>> evaluator = GestureEvaluator(model, test_dataset, device="cuda")
        >>> result = evaluator.evaluate()
        >>> evaluator.print_report(result)
        >>> evaluator.save_report(result, "reports/eval.json")
    """

    def __init__(
        self,
        model: GestureTransformer,
        dataset: GestureSequenceDataset,
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._model = model.to(self._device)
        self._model.eval()
        self._dataset = dataset
        self._label_names = dataset.label_names

        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    @torch.no_grad()
    def evaluate(self, top_k: tuple[int, ...] = (1, 3, 5)) -> EvalResult:
        """Run full evaluation.

        Args:
            top_k: Tuple of k values for top-k accuracy.

        Returns:
            EvalResult with all metrics.
        """
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[np.ndarray] = []
        latencies: list[float] = []

        for features, labels, masks in self._loader:
            features = features.to(self._device, non_blocking=True)
            masks = masks.to(self._device, non_blocking=True)

            t_start = time.perf_counter()
            output = self._model(features, mask=masks)
            t_end = time.perf_counter()

            latency_per_sample = (t_end - t_start) * 1000.0 / features.size(0)
            latencies.extend([latency_per_sample] * features.size(0))

            probs = torch.softmax(output["logits"], dim=-1).cpu().numpy()
            preds = output["logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
            all_probs.extend(probs)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)
        latencies_arr = np.array(latencies)

        # Core metrics
        result = EvalResult(
            accuracy=accuracy_score(y_true, y_pred),
            precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
            f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
            per_class_report=classification_report(
                y_true,
                y_pred,
                target_names=self._label_names,
                output_dict=True,
                zero_division=0,
            ),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            total_samples=len(y_true),
            avg_latency_ms=float(np.mean(latencies_arr)),
            p95_latency_ms=float(np.percentile(latencies_arr, 95)),
            p99_latency_ms=float(np.percentile(latencies_arr, 99)),
        )

        # Top-k accuracy
        for k in top_k:
            if k <= y_probs.shape[1]:
                top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
                top_k_correct = sum(1 for i, label in enumerate(y_true) if label in top_k_preds[i])
                result.top_k_accuracy[k] = top_k_correct / len(y_true)

        return result

    def print_report(self, result: EvalResult) -> None:
        """Print a formatted evaluation report."""
        logger.info("=" * 60)
        logger.info("GESTURE MODEL EVALUATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total samples:      {result.total_samples}")
        logger.info(f"Accuracy:           {result.accuracy:.4f}")
        logger.info(f"Precision (macro):  {result.precision_macro:.4f}")
        logger.info(f"Recall (macro):     {result.recall_macro:.4f}")
        logger.info(f"F1 (macro):         {result.f1_macro:.4f}")
        logger.info("-" * 60)
        for k, acc in sorted(result.top_k_accuracy.items()):
            logger.info(f"Top-{k} Accuracy:    {acc:.4f}")
        logger.info("-" * 60)
        logger.info(f"Avg latency:        {result.avg_latency_ms:.2f} ms")
        logger.info(f"P95 latency:        {result.p95_latency_ms:.2f} ms")
        logger.info(f"P99 latency:        {result.p99_latency_ms:.2f} ms")
        logger.info("=" * 60)

    def save_report(self, result: EvalResult, path: str | Path) -> None:
        """Save evaluation report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "accuracy": result.accuracy,
            "precision_macro": result.precision_macro,
            "recall_macro": result.recall_macro,
            "f1_macro": result.f1_macro,
            "top_k_accuracy": {str(k): v for k, v in result.top_k_accuracy.items()},
            "latency": {
                "avg_ms": result.avg_latency_ms,
                "p95_ms": result.p95_latency_ms,
                "p99_ms": result.p99_latency_ms,
            },
            "total_samples": result.total_samples,
            "per_class": result.per_class_report,
            "confusion_matrix": result.confusion_matrix.tolist(),
        }

        path.write_text(json.dumps(report, indent=2))
        logger.info(f"Evaluation report saved: {path}")


def benchmark_latency(
    model: GestureTransformer,
    seq_len: int = 30,
    feature_dim: int = 86,
    num_iterations: int = 1000,
    device: str = "cpu",
    warmup: int = 50,
) -> dict[str, float]:
    """Benchmark raw model inference latency.

    Args:
        model: Trained gesture model.
        seq_len: Input sequence length.
        feature_dim: Feature dimension.
        num_iterations: Number of benchmark iterations.
        device: Device to benchmark on.
        warmup: Number of warmup iterations.

    Returns:
        Dict with avg_ms, p50_ms, p95_ms, p99_ms, min_ms, max_ms.
    """
    device_t = torch.device(device)
    model = model.to(device_t).eval()
    dummy_input = torch.randn(1, seq_len, feature_dim, device=device_t)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(dummy_input)

    if device_t.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(num_iterations):
        if device_t.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy_input)
        if device_t.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    results = {
        "avg_ms": float(np.mean(arr)),
        "p50_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }

    logger.info(
        f"Latency benchmark ({device}, {num_iterations} iters): "
        f"avg={results['avg_ms']:.2f}ms, "
        f"p95={results['p95_ms']:.2f}ms, "
        f"p99={results['p99_ms']:.2f}ms"
    )

    return results
