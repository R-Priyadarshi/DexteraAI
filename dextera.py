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

import torch
from loguru import logger


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
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    train_parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")

    # ---- eval ----
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--dataset", type=str, required=True, help="Path to test dataset")
    eval_parser.add_argument("--output", type=str, default="reports/eval.json", help="Output report path")
    eval_parser.add_argument("--device", type=str, default="auto", help="Device")

    # ---- export ----
    export_parser = subparsers.add_parser("export", help="Export model to ONNX/TFLite")
    export_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    export_parser.add_argument("--format", type=str, choices=["onnx", "tflite", "both"], default="onnx")
    export_parser.add_argument("--output", type=str, default="models/", help="Output directory")
    export_parser.add_argument("--quantize", action="store_true", help="Apply quantization")

    # ---- demo ----
    demo_parser = subparsers.add_parser("demo", help="Run real-time webcam demo")
    demo_parser.add_argument("--model", type=str, default=None, help="Path to ONNX model (optional)")
    demo_parser.add_argument("--camera", type=int, default=0, help="Camera device index")

    # ---- benchmark ----
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark model latency")
    bench_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    bench_parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    bench_parser.add_argument("--device", type=str, default="cpu", help="Device")

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


def cmd_train(args: argparse.Namespace) -> None:
    """Train a gesture model."""
    from core.temporal.model import GestureTransformer
    from training.datasets.gesture_dataset import (
        GestureSequenceDataset,
        SyntheticGestureDataset,
    )
    from training.trainers.train_gesture import GestureTrainer, TrainConfig

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Dataset
    if args.synthetic:
        logger.info("Using synthetic dataset for pipeline testing")
        dataset = SyntheticGestureDataset(num_samples=2000, seq_len=30)
    elif args.dataset:
        dataset = GestureSequenceDataset(args.dataset, seq_len=30, augment=True)
    else:
        logger.error("Provide --dataset <path> or use --synthetic for testing.")
        sys.exit(1)

    if len(dataset) == 0:
        logger.error(f"Dataset is empty: {args.dataset}")
        logger.info("Use --synthetic for testing, or prepare data in the expected format.")
        sys.exit(1)

    # Model
    model = GestureTransformer(
        input_dim=dataset.feature_dim,
        num_classes=dataset.num_classes,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
    )

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    trainer = GestureTrainer(model, dataset, config, device=args.device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    result = trainer.train()

    logger.info(f"Training complete! Best val accuracy: {result.best_val_accuracy:.4f}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a trained model."""
    from core.temporal.model import GestureTransformer
    from training.datasets.gesture_dataset import GestureSequenceDataset
    from training.evaluation.metrics import GestureEvaluator

    dataset = GestureSequenceDataset(args.dataset, seq_len=30)

    # Load checkpoint
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

            # Draw info overlay
            y_offset = 30
            cv2.putText(
                frame,
                f"Hands: {len(result.hands)} | "
                f"Latency: {result.inference_time_ms:.1f}ms",
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
                    0.8,
                    color,
                    2,
                )

            # Draw landmarks
            h, w = frame.shape[:2]
            for hand in result.hands:
                for lm in hand.landmarks:
                    x_px = int(lm[0] * w)
                    y_px = int(lm[1] * h)
                    cv2.circle(frame, (x_px, y_px), 4, (99, 102, 241), -1)

            cv2.imshow("DexteraAI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark model latency."""
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

    print("\n=== LATENCY BENCHMARK ===")
    for key, val in results.items():
        print(f"  {key}: {val:.3f} ms")

    # Pass/fail against 20ms budget
    if results["p95_ms"] < 20.0:
        print("\n✅ PASS: P95 latency within 20ms budget")
    else:
        print(f"\n❌ FAIL: P95 latency {results['p95_ms']:.1f}ms exceeds 20ms budget")
        sys.exit(1)


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
        torch_ver = "not installed"

    try:
        import mediapipe as mp
        mp_ver = mp.__version__
    except ImportError:
        mp_ver = "not installed"

    try:
        import onnxruntime as ort
        ort_ver = ort.__version__
        providers = ort.get_available_providers()
    except ImportError:
        ort_ver = "not installed"
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
