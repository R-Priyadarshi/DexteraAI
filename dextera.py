"""DexteraAI CLI — command-line interface for training, evaluation, export, and demo.

Usage:
    python -m dextera train --dataset data/hagrid --epochs 50
    python -m dextera eval --checkpoint checkpoints/best.pt --dataset data/hagrid_test
    python -m dextera export --checkpoint checkpoints/best.pt --format onnx
    python -m dextera demo --model models/gesture.onnx
    python -m dextera benchmark --model models/gesture.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

if TYPE_CHECKING:
    import numpy as np

    from core.inference.onnx_runtime import ONNXInferenceRuntime
    from core.types import FrameResult
    from training.datasets.gesture_dataset import GestureSequenceDataset

TORCH_NOT_INSTALLED = "not installed"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dextera",
        description="DexteraAI — Gesture Intelligence Platform CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    train_parser = subparsers.add_parser("train", help="Train a gesture model")
    train_parser.add_argument("--dataset", type=str, default=None, help="Path to dataset directory")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    train_parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing"
    )
    train_parser.add_argument(
        "--seq-len", type=int, default=30, help="Temporal window length in frames"
    )
    train_parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Fraction held out for the final test report (never trained or validated on)",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: min(8, cpu_count-2)). Feature extraction "
             "runs in these workers, so more helps on large datasets.",
    )
    train_parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Weight the loss by inverse class frequency (for imbalanced data)",
    )
    train_parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit temperature scaling and an open-set rejection threshold on the "
             "validation split, and record them in the exported bundle",
    )
    train_parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="After training, export an ONNX model bundle to this directory",
    )

    # ---- eval ----
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model or ONNX export")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint or ONNX model"
    )
    eval_parser.add_argument("--dataset", type=str, required=True, help="Path to test dataset")
    eval_parser.add_argument(
        "--output",
        type=str,
        default="reports/eval.json",
        help="Output report path"
    )
    eval_parser.add_argument("--device", type=str, default="auto", help="Device")
    eval_parser.add_argument(
        "--onnx",
        action="store_true",
        help="Use ONNX model for evaluation (auto-detected by .onnx extension)"
    )
    eval_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset for evaluation (pipeline test)"
    )

    # ---- export ----
    export_parser = subparsers.add_parser("export", help="Export model to ONNX/TFLite")
    export_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    export_parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "tflite", "both"],
        default="onnx"
    )
    export_parser.add_argument("--output", type=str, default="models/", help="Output directory")
    export_parser.add_argument("--quantize", action="store_true", help="Apply quantization")

    # ---- demo ----
    demo_parser = subparsers.add_parser("demo", help="Run real-time webcam demo")
    demo_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to ONNX model (optional)"
    )
    demo_parser.add_argument("--camera", type=int, default=0, help="Camera device index")

    # ---- benchmark ----
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark model latency (PyTorch or ONNX)"
    )
    bench_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint or ONNX model"
    )
    bench_parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    bench_parser.add_argument("--device", type=str, default="cpu", help="Device")
    bench_parser.add_argument(
        "--onnx",
        action="store_true",
        help="Use ONNX model for benchmarking (auto-detected by .onnx extension)"
    )

    # ---- serve ----
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI API server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port")
    serve_parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # ---- info ----
    subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "info":
        cmd_info()


def _split_indices(
    total: int, val_split: float, test_split: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic disjoint train/val/test index split."""
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total).tolist()
    n_test = int(total * test_split)
    n_val = int(total * val_split)
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return train_idx, val_idx, test_idx


def cmd_train(args: argparse.Namespace) -> None:
    """Train a gesture model on landmark sequences."""
    import json
    import os

    from core.temporal.model import GestureTransformer
    from training.datasets.gesture_dataset import (
        GestureSequenceDataset,
        SyntheticGestureDataset,
    )
    from training.trainers.train_gesture import GestureTrainer, TrainConfig

    default_workers = min(8, max(1, (os.cpu_count() or 4) - 2))
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        use_class_weights=args.class_weights,
        num_workers=args.num_workers if args.num_workers is not None else default_workers,
    )

    seq_len = args.seq_len

    if args.synthetic:
        logger.info("Using synthetic dataset for pipeline testing")
        train_ds: object = SyntheticGestureDataset(num_samples=2000, seq_len=seq_len)
        val_ds = None
        test_ds = None
        label_names = list(train_ds.label_names)  # type: ignore[attr-defined]
        feature_dim = train_ds.feature_dim  # type: ignore[attr-defined]
        num_classes = train_ds.num_classes  # type: ignore[attr-defined]
    elif args.dataset:
        probe = GestureSequenceDataset(args.dataset, seq_len=seq_len)
        total = len(probe)
        if total == 0:
            logger.error(f"Dataset is empty: {args.dataset}")
            logger.info("Run `python -m training.datasets.extract_landmarks --help` to build one.")
            sys.exit(1)

        label_names = list(probe.label_names)
        feature_dim = probe.feature_dim
        num_classes = probe.num_classes

        train_idx, val_idx, test_idx = _split_indices(
            total, config.val_split, args.test_split, config.seed
        )
        # Augmentation on train only; val/test stay clean for honest metrics.
        train_ds = GestureSequenceDataset(
            args.dataset, seq_len=seq_len, augment=True, indices=train_idx
        )
        val_ds = GestureSequenceDataset(args.dataset, seq_len=seq_len, indices=val_idx)
        test_ds = (
            GestureSequenceDataset(args.dataset, seq_len=seq_len, indices=test_idx)
            if test_idx
            else None
        )
        logger.info(
            f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
            f"| classes={num_classes} | feature_dim={feature_dim}"
        )
    else:
        logger.error("Provide --dataset <path> or use --synthetic for testing.")
        sys.exit(1)

    model = GestureTransformer(
        input_dim=feature_dim,
        num_classes=num_classes,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        max_seq_len=max(seq_len, 60),
    )
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    trainer = GestureTrainer(
        model,
        train_ds,  # type: ignore[arg-type]
        config,
        device=args.device,
        val_dataset=val_ds,  # type: ignore[arg-type]
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    result = trainer.train()
    logger.info(f"Training complete! Best val accuracy: {result.best_val_accuracy:.4f}")
    test_accuracy: float | None = None

    # ── Held-out test evaluation ──────────────────────────────
    if test_ds is not None and len(test_ds) > 0:
        from training.evaluation.metrics import GestureEvaluator

        best_ckpt = Path(config.checkpoint_dir) / "best.pt"
        if best_ckpt.exists():
            ckpt = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

        evaluator = GestureEvaluator(model, test_ds, device=args.device)
        eval_result = evaluator.evaluate()
        evaluator.print_report(eval_result)

        # Keep a per-model report alongside the canonical one, so training a
        # second model does not silently overwrite the first model's metrics.
        dataset_name = Path(args.dataset).name if args.dataset else "synthetic"
        evaluator.save_report(eval_result, f"reports/eval_{dataset_name}.json")
        evaluator.save_report(eval_result, "reports/eval.json")
        test_accuracy = eval_result.accuracy

    # ── Optional confidence calibration ───────────────────────
    calibration: dict[str, object] | None = None
    if args.calibrate and val_ds is not None and len(val_ds) > 0:
        from training.evaluation.calibrate_confidence import calibrate as fit_calibration

        best_ckpt = Path(config.checkpoint_dir) / "best.pt"
        if best_ckpt.exists():
            ckpt = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        calibration = fit_calibration(model, val_ds, device=args.device).to_dict()

    # ── Optional export ───────────────────────────────────────
    if args.export:
        from training.export.to_onnx import export_to_onnx

        export_dir = Path(args.export)
        export_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = export_to_onnx(model.cpu(), export_dir / "gesture.onnx", seq_len=seq_len)
        (export_dir / "labels.json").write_text(
            json.dumps(
                {
                    "labels": label_names,
                    "seq_len": seq_len,
                    "feature_dim": feature_dim,
                    "val_accuracy": round(result.best_val_accuracy, 4),
                    "test_accuracy": round(test_accuracy, 4) if test_accuracy else None,
                    "calibration": calibration,
                },
                indent=2,
            )
        )
        # Also place labels next to the checkpoint, so loading best.pt directly
        # (CLI demo, GesturePipeline) resolves the same vocabulary.
        checkpoint_labels = Path(config.checkpoint_dir) / "labels.json"
        checkpoint_labels.write_text(json.dumps({"labels": label_names}, indent=2))

        logger.info(f"Exported model bundle: {onnx_path} + labels.json")


def _onnx_eval(
    runtime: ONNXInferenceRuntime,
    dataset: GestureSequenceDataset,
    label_names: list[str],
    output_path: str
) -> None:
    import time

    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    from training.evaluation.metrics import EvalResult, GestureEvaluator

    all_preds, all_labels, all_probs, latencies = [], [], [], []
    for features, labels, _ in dataset:
        features_np = np.expand_dims(features, axis=0).astype(np.float32)
        mask_np = np.ones((1, features_np.shape[1]), dtype=bool)
        input_feed = {
            name: mask_np if "mask" in name else features_np
            for name in runtime.input_names
        }
        t_start = time.perf_counter()
        outputs = runtime.predict(input_feed)
        t_end = time.perf_counter()
        logits = outputs[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        pred = np.argmax(logits)
        all_preds.append(pred)
        all_labels.append(labels)
        all_probs.append(probs)
        latencies.append((t_end - t_start) * 1000.0)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    latencies_arr = np.array(latencies)

    result = EvalResult(
        accuracy=accuracy_score(y_true, y_pred),
        precision_macro=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        per_class_report=classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        confusion_matrix=confusion_matrix(y_true, y_pred),
        total_samples=len(y_true),
        avg_latency_ms=float(np.mean(latencies_arr)),
        p95_latency_ms=float(np.percentile(latencies_arr, 95)),
        p99_latency_ms=float(np.percentile(latencies_arr, 99)),
    )
    for k in (1, 3, 5):
        if k <= y_probs.shape[1]:
            top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
            top_k_correct = sum(
                1 for i, label in enumerate(y_true) if label in top_k_preds[i]
            )
            result.top_k_accuracy[k] = top_k_correct / len(y_true)
    GestureEvaluator.print_report(GestureEvaluator, result)
    GestureEvaluator.save_report(GestureEvaluator, result, output_path)
    runtime.close()


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a trained model or ONNX export."""

    from core.inference.onnx_runtime import ONNXInferenceRuntime
    from training.datasets.gesture_dataset import GestureSequenceDataset
    from training.evaluation.metrics import GestureEvaluator

    dataset = GestureSequenceDataset(args.dataset, seq_len=30)

    is_onnx = args.onnx or args.checkpoint.endswith(".onnx")

    if is_onnx:
        # ONNX evaluation
        runtime = ONNXInferenceRuntime()
        runtime.load(args.checkpoint)

        # Support synthetic dataset for ONNX eval
        if getattr(args, "synthetic", False):
            from training.datasets.gesture_dataset import SyntheticGestureDataset
            dataset = SyntheticGestureDataset(num_samples=200, seq_len=30)

        _onnx_eval(runtime, dataset, dataset.label_names, args.output)
    else:
        # PyTorch evaluation (unchanged)
        from core.temporal.model import GestureTransformer
        dataset = GestureSequenceDataset(args.dataset, seq_len=30)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = GestureTransformer(
            input_dim=dataset.feature_dim,
            num_classes=dataset.num_classes,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        evaluator = GestureEvaluator(model, dataset, device=args.device)
        result = evaluator.evaluate()
        evaluator.print_report(result)
        evaluator.save_report(result, args.output)


def cmd_export(args: argparse.Namespace) -> None:
    """Export model to ONNX/TFLite."""
    from core.temporal.model import GestureTransformer
    from training.export.to_onnx import export_to_onnx, quantize_onnx

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model = GestureTransformer()
    model.load_state_dict(ckpt["model_state_dict"])

    # ONNX export
    if args.format in ("onnx", "both"):
        onnx_path = export_to_onnx(model, output_dir / "gesture.onnx")
        if args.quantize:
            quantize_onnx(onnx_path)

    # TFLite export
    if args.format in ("tflite", "both"):
        from training.export.to_tflite import export_to_tflite
        onnx_path = output_dir / "gesture.onnx"
        if not onnx_path.exists():
            onnx_path = export_to_onnx(model, onnx_path)
        export_to_tflite(onnx_path, output_dir / "gesture.tflite")


def _draw_overlay(frame: np.ndarray, result: FrameResult) -> None:
    y_offset = 30
    import cv2
    cv2.putText(
        frame,
        f"Hands: {len(result.hands)} | Latency: {result.inference_time_ms:.1f}ms",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    for gesture in result.gestures:
        y_offset += 35
        color = (0, 255, 0) if gesture.confidence > 0.8 else (0, 255, 255)
        cv2.putText(
            frame,
            f"{gesture.gesture_name}: {gesture.confidence:.1%}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


def cmd_demo(args: argparse.Namespace) -> None:
    """Run real-time webcam demo."""
    import cv2

    from core.inference.pipeline import GesturePipeline, PipelineConfig

    config = PipelineConfig(model_path=args.model)

    with GesturePipeline(config) as pipeline:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {args.camera}")
            sys.exit(1)

        logger.info("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = pipeline.process_frame(frame)

            _draw_overlay(frame, result)

            cv2.imshow("DexteraAI Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def _benchmark_onnx(
    runtime: ONNXInferenceRuntime,
    input_feed: dict[str, np.ndarray],
    iterations: int
) -> dict[str, float]:
    import time

    import numpy as np
    # Warmup
    for _ in range(50):
        runtime.predict(input_feed)
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        runtime.predict(input_feed)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(latencies)
    return {
        "avg_ms": float(np.mean(arr)),
        "p50_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def _print_benchmark_results(results: dict, label: str = "LATENCY BENCHMARK") -> None:
    print(f"\n=== {label} ===")
    for key, val in results.items():
        print(f"  {key}: {val:.3f} ms")
    if results["p95_ms"] < 20.0:
        print("\n✅ PASS: P95 latency within 20ms budget")
    else:
        print(f"\n❌ FAIL: P95 latency {results['p95_ms']:.1f}ms exceeds 20ms budget")
        sys.exit(1)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark model latency (PyTorch or ONNX)."""
    import numpy as np

    from core.inference.onnx_runtime import ONNXInferenceRuntime

    is_onnx = args.onnx or args.checkpoint.endswith(".onnx")

    seq_len = 30
    feature_dim = 86

    if is_onnx:
        rng = np.random.default_rng(seed=42)
        runtime = ONNXInferenceRuntime()
        runtime.load(args.checkpoint)
        input_names = runtime.input_names
        dummy_input = rng.standard_normal((1, seq_len, feature_dim), dtype=np.float32)
        dummy_mask = np.ones((1, seq_len), dtype=bool)
        input_feed = {name: dummy_mask if "mask" in name else dummy_input for name in input_names}
        results = _benchmark_onnx(runtime, input_feed, args.iterations)
        _print_benchmark_results(results, label="ONNX LATENCY BENCHMARK")
        runtime.close()
    else:
        from core.temporal.model import GestureTransformer
        from training.evaluation.metrics import benchmark_latency
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = GestureTransformer()
        model.load_state_dict(ckpt["model_state_dict"])
        results = benchmark_latency(
            model,
            num_iterations=args.iterations,
            device=args.device,
        )
        _print_benchmark_results(results)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI API server (optional server mode)."""
    import uvicorn

    logger.info("Starting DexteraAI API server...")
    uvicorn.run(
        "backend.apps.api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


def cmd_info() -> None:
    """Show system information."""
    import platform

    try:
        cuda = torch.cuda.is_available()
        torch_ver = torch.__version__
    except Exception:
        cuda = False
        torch_ver = TORCH_NOT_INSTALLED

    try:
        import mediapipe as mp
        mp_ver = mp.__version__
    except ImportError:
        mp_ver = TORCH_NOT_INSTALLED

    try:
        import onnxruntime as ort
        ort_ver = ort.__version__
        providers = ort.get_available_providers()
    except ImportError:
        ort_ver = TORCH_NOT_INSTALLED
        providers = []

    print(f"""
🤟 DexteraAI — Gesture Intelligence Platform
══════════════════════════════════════════════
  Python:       {platform.python_version()}
  Platform:     {platform.system()} {platform.machine()}
  PyTorch:      {torch_ver}
  CUDA:         {"✅" if cuda else "❌"}
  MediaPipe:    {mp_ver}
  ONNX Runtime: {ort_ver}
  ORT Providers:{providers}
""")


if __name__ == "__main__":
    main()
