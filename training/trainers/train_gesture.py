"""Production PyTorch training loop for gesture recognition.

Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Linear warmup → cosine annealing LR schedule
    - Class-weighted loss for imbalanced datasets
    - Early stopping on validation loss
    - MLflow experiment tracking (optional)
    - Checkpoint management (best + last)
    - Reproducible seeding
    - Callback hooks for extensibility
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, random_split

if TYPE_CHECKING:
    from collections.abc import Callable


def _init_worker(worker_id: int) -> None:
    """Keep DataLoader workers single-threaded to avoid CPU oversubscription."""
    torch.set_num_threads(1)


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        epochs: Maximum training epochs.
        batch_size: Batch size.
        learning_rate: Peak learning rate (after warmup).
        weight_decay: AdamW weight decay.
        warmup_epochs: Number of linear warmup epochs.
        min_lr: Floor learning rate for the cosine schedule.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        max_grad_norm: Gradient clipping norm.
        use_amp: Use mixed precision training.
        early_stopping_patience: Stop after N epochs without val-loss improvement.
        val_split: Fraction of data for validation (ignored if val_dataset is passed).
        num_workers: DataLoader workers.
        seed: Random seed for reproducibility.
        checkpoint_dir: Directory to save checkpoints.
        experiment_name: MLflow experiment name.
        label_smoothing: Cross-entropy label smoothing.
        use_class_weights: Weight the loss by inverse class frequency.
    """

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    early_stopping_patience: int = 15
    val_split: float = 0.15
    num_workers: int = 4
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "gesture_training"
    label_smoothing: float = 0.1
    use_class_weights: bool = False


@dataclass
class TrainResult:
    """Result of a training run."""

    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    best_epoch: int = 0
    total_epochs: int = 0
    train_history: list[dict[str, float]] = field(default_factory=list)
    val_history: list[dict[str, float]] = field(default_factory=list)
    training_time_sec: float = 0.0


class Callback:
    """Extension hook. Subclass and override the hooks you need."""

    def on_train_start(self, trainer: GestureTrainer) -> None:
        """Called once before the first epoch."""

    def on_epoch_start(self, epoch: int, trainer: GestureTrainer) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: GestureTrainer) -> None:
        """Called at the end of each epoch with validation metrics."""

    def on_train_end(self, trainer: GestureTrainer, result: TrainResult) -> None:
        """Called once after the final epoch."""


class GestureTrainer:
    """Training loop for GestureTransformer / StaticGestureClassifier.

    Usage:
        >>> config = TrainConfig(epochs=50, batch_size=32)
        >>> trainer = GestureTrainer(model, dataset, config, device="cuda")
        >>> result = trainer.train()
        >>> print(f"Best val accuracy: {result.best_val_accuracy:.4f}")

    Args:
        model: The model to train. Must return a dict containing "logits".
        dataset: Training dataset yielding (features, label, mask) or (features, label).
        config: Training hyperparameters.
        device: "auto", "cpu", "cuda", or an explicit device string.
        callbacks: Optional extension hooks.
        val_dataset: Optional explicit validation set. When given, `dataset` is used
            entirely for training and `config.val_split` is ignored. Prefer this for
            real datasets so the split can be subject-disjoint.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset[Any],
        config: TrainConfig | None = None,
        device: str = "auto",
        callbacks: list[Callback] | None = None,
        val_dataset: Dataset[Any] | None = None,
    ) -> None:
        self._config = config or TrainConfig()
        self._set_seed(self._config.seed)

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._model = model.to(self._device)
        self._callbacks = callbacks or []

        # ── Dataset splits ────────────────────────────────────
        if val_dataset is not None:
            self._train_dataset: Dataset[Any] = dataset
            self._val_dataset: Dataset[Any] = val_dataset
        else:
            total = len(dataset)  # type: ignore[arg-type]
            val_size = int(total * self._config.val_split)
            train_size = total - val_size
            self._train_dataset, self._val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self._config.seed),
            )

        train_len = len(self._train_dataset)  # type: ignore[arg-type]
        val_len = len(self._val_dataset)  # type: ignore[arg-type]
        if train_len == 0:
            raise ValueError("Training split is empty — check dataset path and val_split.")

        loader_kwargs: dict[str, Any] = {
            "batch_size": self._config.batch_size,
            "num_workers": self._config.num_workers,
            "pin_memory": self._device.type == "cuda",
        }
        if self._config.num_workers > 0:
            # Each worker does numpy/torch work that would otherwise spawn its own
            # thread pool; N workers x M threads oversubscribes the CPU badly and
            # collapses throughput. Pin every worker to a single thread.
            loader_kwargs["worker_init_fn"] = _init_worker
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4

        self._train_loader = DataLoader(
            self._train_dataset,
            shuffle=True,
            drop_last=train_len > self._config.batch_size,
            **loader_kwargs,
        )
        self._val_loader = DataLoader(
            self._val_dataset,
            shuffle=False,
            **loader_kwargs,
        )

        # ── Optimizer ─────────────────────────────────────────
        self._optimizer = AdamW(
            self._model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        # ── LR schedule: linear warmup → cosine annealing ─────
        self._scheduler = LambdaLR(self._optimizer, lr_lambda=self._lr_lambda)

        # ── Loss ──────────────────────────────────────────────
        weights = self._compute_class_weights(dataset) if self._config.use_class_weights else None
        self._criterion = nn.CrossEntropyLoss(
            label_smoothing=self._config.label_smoothing,
            weight=weights,
        )

        # ── AMP ───────────────────────────────────────────────
        self._amp_enabled = self._config.use_amp and self._device.type == "cuda"
        self._scaler = GradScaler(device=self._device.type, enabled=self._amp_enabled)

        self._ckpt_dir = Path(self._config.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Trainer ready | device={} | train={} | val={} | params={:,} | amp={}",
            self._device,
            train_len,
            val_len,
            sum(p.numel() for p in self._model.parameters()),
            self._amp_enabled,
        )

    # ── Properties ────────────────────────────────────────────

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def config(self) -> TrainConfig:
        return self._config

    @property
    def device(self) -> torch.device:
        return self._device

    # ── LR schedule ───────────────────────────────────────────

    def _lr_lambda(self, epoch: int) -> float:
        """Linear warmup for `warmup_epochs`, then cosine decay to `min_lr`."""
        warmup = self._config.warmup_epochs
        peak = self._config.learning_rate
        floor_ratio = self._config.min_lr / peak if peak > 0 else 0.0

        if warmup > 0 and epoch < warmup:
            # Ramp from 1/warmup up to 1.0 so the first epoch is never lr=0.
            return float(epoch + 1) / float(warmup)

        total_decay_epochs = max(1, self._config.epochs - warmup)
        progress = min(1.0, float(epoch - warmup) / float(total_decay_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(floor_ratio + (1.0 - floor_ratio) * cosine)

    # ── Class weights ─────────────────────────────────────────

    def _compute_class_weights(self, dataset: Dataset[Any]) -> torch.Tensor | None:
        """Inverse-frequency class weights, for imbalanced real-world datasets."""
        num_classes = getattr(dataset, "num_classes", None)
        if num_classes is None:
            logger.warning("Dataset exposes no num_classes; skipping class weights.")
            return None

        labels: list[int] = []
        try:
            # Fast path: read labels straight off disk instead of materializing
            # every sample through the normalize + feature-extract pipeline.
            fast = getattr(dataset, "get_labels", None)
            if callable(fast):
                labels = [int(x) for x in fast()]
            else:
                for i in range(len(dataset)):  # type: ignore[arg-type]
                    labels.append(int(dataset[i][1]))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not compute class weights: {e}")
            return None

        counts = Counter(labels)
        total = len(labels)
        weights = torch.tensor(
            [total / (num_classes * max(counts.get(c, 0), 1)) for c in range(num_classes)],
            dtype=torch.float32,
            device=self._device,
        )
        logger.info(f"Class weights enabled | min={weights.min():.3f} max={weights.max():.3f}")
        return weights

    # ── Training ──────────────────────────────────────────────

    def train(self) -> TrainResult:
        """Run the full training loop.

        Returns:
            TrainResult with metrics history and best performance.
        """
        result = TrainResult()
        best_val_loss = float("inf")
        patience_counter = 0
        t_start = time.time()
        epoch = 0

        mlflow_active = self._init_mlflow()

        for cb in self._callbacks:
            cb.on_train_start(self)

        for epoch in range(1, self._config.epochs + 1):
            for cb in self._callbacks:
                cb.on_epoch_start(epoch, self)

            train_metrics = self._train_epoch()
            val_metrics = self._validate_epoch()

            result.train_history.append(train_metrics)
            result.val_history.append(val_metrics)

            # Step the epoch-wise schedule *after* the epoch completes.
            self._scheduler.step()
            current_lr = self._optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch {}/{} | train_loss={:.4f} train_acc={:.4f} | "
                "val_loss={:.4f} val_acc={:.4f} | lr={:.2e}",
                epoch,
                self._config.epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                current_lr,
            )

            if mlflow_active:
                self._log_mlflow(epoch, train_metrics, val_metrics, current_lr)

            self._save_checkpoint(epoch, val_metrics["loss"], "last.pt")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                result.best_val_loss = best_val_loss
                result.best_val_accuracy = val_metrics["accuracy"]
                result.best_epoch = epoch
                self._save_checkpoint(epoch, val_metrics["loss"], "best.pt")
                patience_counter = 0
                logger.info("  ✓ New best model (val_loss={:.4f})", best_val_loss)
            else:
                patience_counter += 1

            for cb in self._callbacks:
                cb.on_epoch_end(epoch, val_metrics, self)

            if patience_counter >= self._config.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch {} (no improvement for {} epochs)",
                    epoch,
                    patience_counter,
                )
                break

        result.total_epochs = epoch
        result.training_time_sec = time.time() - t_start

        for cb in self._callbacks:
            cb.on_train_end(self, result)

        self._save_artifacts(result)

        if mlflow_active:
            with contextlib.suppress(Exception):
                import mlflow

                mlflow.end_run()

        logger.info(
            "Training complete | best_epoch={} | best_val_loss={:.4f} | "
            "best_val_acc={:.4f} | time={:.1f}s",
            result.best_epoch,
            result.best_val_loss,
            result.best_val_accuracy,
            result.training_time_sec,
        )
        return result

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Support both (features, label, mask) and (features, label) datasets."""
        if len(batch) == 3:
            features, labels, masks = batch
            return (
                features.to(self._device, non_blocking=True),
                labels.to(self._device, non_blocking=True),
                masks.to(self._device, non_blocking=True),
            )
        features, labels = batch
        return (
            features.to(self._device, non_blocking=True),
            labels.to(self._device, non_blocking=True),
            None,
        )

    def _forward(self, features: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
        """Run the model, tolerating both temporal (mask-aware) and static models."""
        output = self._model(features) if masks is None else self._model(features, mask=masks)
        logits: torch.Tensor = output["logits"] if isinstance(output, dict) else output
        return logits

    def _train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self._model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        self._optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self._train_loader):
            features, labels, masks = self._unpack_batch(batch)

            with autocast(device_type=self._device.type, enabled=self._amp_enabled):
                logits = self._forward(features, masks)
                loss = self._criterion(logits, labels)
                loss = loss / self._config.gradient_accumulation_steps

            self._scaler.scale(loss).backward()

            if (batch_idx + 1) % self._config.gradient_accumulation_steps == 0:
                self._scaler.unscale_(self._optimizer)
                nn.utils.clip_grad_norm_(self._model.parameters(), self._config.max_grad_norm)
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad(set_to_none=True)

            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
            total_loss += float(loss.item()) * self._config.gradient_accumulation_steps

        return {
            "loss": total_loss / max(len(self._train_loader), 1),
            "accuracy": correct / max(total, 1),
        }

    @torch.no_grad()
    def _validate_epoch(self) -> dict[str, float]:
        """Run one validation epoch."""
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self._val_loader:
            features, labels, masks = self._unpack_batch(batch)
            logits = self._forward(features, masks)
            loss = self._criterion(logits, labels)

            total_loss += float(loss.item())
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

        return {
            "loss": total_loss / max(len(self._val_loader), 1),
            "accuracy": correct / max(total, 1),
        }

    # ── Checkpointing ─────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, filename: str) -> None:
        """Save a training checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": self._scheduler.state_dict(),
                "val_loss": val_loss,
                "config": asdict(self._config),
            },
            self._ckpt_dir / filename,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint and return its epoch number.

        Unpickling arbitrary objects is attempted only as a fallback, matching
        `GesturePipeline.load_model`: `torch.load` without `weights_only` can
        execute code embedded in the file, which matters the moment anyone
        resumes from a checkpoint they did not train themselves.
        """
        try:
            ckpt = torch.load(str(path), map_location=self._device, weights_only=True)
        except Exception:
            logger.warning(f"Falling back to unsafe load for legacy checkpoint: {path}")
            # nosec B614 — see above; the safe path is tried first and this is
            # reached only for this project's own older checkpoints.
            ckpt = torch.load(  # nosec B614
                str(path), map_location=self._device, weights_only=False
            )
        self._model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self._scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        epoch = int(ckpt.get("epoch", 0))
        logger.info(f"Loaded checkpoint: {path} (epoch {epoch})")
        return epoch

    def _save_artifacts(self, result: TrainResult) -> None:
        """Write metrics JSON next to the checkpoints (and to MLflow if active)."""
        try:
            metrics_path = self._ckpt_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "train_history": result.train_history,
                        "val_history": result.val_history,
                        "best_val_loss": result.best_val_loss,
                        "best_val_accuracy": result.best_val_accuracy,
                        "best_epoch": result.best_epoch,
                        "total_epochs": result.total_epochs,
                        "training_time_sec": result.training_time_sec,
                    },
                    indent=2,
                )
            )
            with contextlib.suppress(Exception):
                import mlflow

                mlflow.log_artifact(str(metrics_path))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Failed to save training artifacts: {e}")

    # ── MLflow ────────────────────────────────────────────────

    def _init_mlflow(self) -> bool:
        """Start an MLflow run if MLflow is installed."""
        try:
            import mlflow
        except ImportError:
            logger.info("MLflow not installed — skipping experiment tracking.")
            return False

        try:
            mlflow.set_experiment(self._config.experiment_name)
            mlflow.start_run()
            params = {k: v for k, v in asdict(self._config).items() if not is_dataclass(v)}
            params["model_params"] = sum(p.numel() for p in self._model.parameters())
            params["device"] = str(self._device)
            mlflow.log_params(params)
            logger.info("MLflow tracking enabled.")
            return True
        except Exception as e:
            logger.warning(f"MLflow init failed, continuing without tracking: {e}")
            return False

    def _log_mlflow(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        lr: float,
    ) -> None:
        """Log epoch metrics to MLflow."""
        with contextlib.suppress(Exception):
            import mlflow

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "learning_rate": lr,
                },
                step=epoch,
            )

    # ── Utilities ─────────────────────────────────────────────

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set all random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def tune(
        self,
        objective_fn: Callable[[TrainConfig], float],
        n_trials: int = 20,
    ) -> TrainConfig:
        """Hyperparameter search with Optuna. Returns the best config found."""
        import optuna

        def optuna_objective(trial: optuna.Trial) -> float:
            trial_config = TrainConfig(
                epochs=self._config.epochs,
                checkpoint_dir=self._config.checkpoint_dir,
                learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
                warmup_epochs=trial.suggest_int("warmup_epochs", 1, 10),
            )
            return objective_fn(trial_config)

        study = optuna.create_study(direction="maximize")
        study.optimize(optuna_objective, n_trials=n_trials)
        logger.info(f"Optuna best params: {study.best_params}")

        best = TrainConfig(
            epochs=self._config.epochs,
            checkpoint_dir=self._config.checkpoint_dir,
            **study.best_params,
        )
        return best
