"""Production-grade PyTorch training loop for gesture recognition.

Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate scheduling (cosine annealing + warmup)
    - Early stopping
    - MLflow experiment tracking
    - Checkpoint management (best + last)
    - Reproducible seeding
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split

if TYPE_CHECKING:
    from core.temporal.model import GestureTransformer
    from training.datasets.gesture_dataset import GestureSequenceDataset


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        epochs: Maximum training epochs.
        batch_size: Batch size.
        learning_rate: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_epochs: Number of warmup epochs.
        min_lr: Minimum learning rate for cosine schedule.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        max_grad_norm: Gradient clipping norm.
        use_amp: Use mixed precision training.
        early_stopping_patience: Stop after N epochs without improvement.
        val_split: Fraction of data for validation.
        num_workers: DataLoader workers.
        seed: Random seed for reproducibility.
        checkpoint_dir: Directory to save checkpoints.
        experiment_name: MLflow experiment name.
        log_every_n_steps: Log metrics every N training steps.
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
    log_every_n_steps: int = 10


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


class GestureTrainer:
    """Production training loop for GestureTransformer.

    Usage:
        >>> config = TrainConfig(epochs=50, batch_size=32)
        >>> trainer = GestureTrainer(model, dataset, config, device="cuda")
        >>> result = trainer.train()
        >>> print(f"Best val accuracy: {result.best_val_accuracy:.4f}")
    """

    def __init__(
        self,
        model: GestureTransformer,
        dataset: GestureSequenceDataset,
        config: TrainConfig | None = None,
        device: str = "auto",
    ) -> None:
        self._config = config or TrainConfig()
        self._set_seed(self._config.seed)

        # Device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._model = model.to(self._device)

        # Split dataset
        val_size = int(len(dataset) * self._config.val_split)
        train_size = len(dataset) - val_size
        self._train_dataset, self._val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self._config.seed),
        )

        # DataLoaders
        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=self._config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self._val_loader = DataLoader(
            self._val_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=self._config.num_workers,
            pin_memory=True,
        )

        # Optimizer
        self._optimizer = AdamW(
            self._model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        # LR Scheduler: linear warmup → cosine annealing
        # Clamp warmup to be strictly less than total epochs
        effective_warmup = min(self._config.warmup_epochs, max(self._config.epochs - 1, 0))
        cosine_epochs = max(self._config.epochs - effective_warmup, 1)

        warmup_scheduler = LinearLR(
            self._optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(effective_warmup, 1),
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self._optimizer,
            T_0=cosine_epochs,
            eta_min=self._config.min_lr,
        )
        self._scheduler = SequentialLR(
            self._optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[effective_warmup],
        )

        # Loss
        self._criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # AMP
        self._scaler = GradScaler(
            device=self._device.type,
            enabled=self._config.use_amp,
        )

        # Checkpoint dir
        self._ckpt_dir = Path(self._config.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Trainer initialized | Device: {self._device} | "
            f"Train: {train_size} | Val: {val_size} | "
            f"Params: {sum(p.numel() for p in model.parameters()):,}"
        )

    def train(self) -> TrainResult:
        """Run the full training loop.

        Returns:
            TrainResult with metrics history and best performance.
        """
        result = TrainResult()
        best_val_loss = float("inf")
        patience_counter = 0
        t_start = time.time()

        # Optional: MLflow tracking
        mlflow_active = self._init_mlflow()

        for epoch in range(1, self._config.epochs + 1):
            # Train
            train_metrics = self._train_epoch(epoch)
            result.train_history.append(train_metrics)

            # Validate
            val_metrics = self._validate_epoch(epoch)
            result.val_history.append(val_metrics)

            # LR step
            self._scheduler.step()
            current_lr = self._optimizer.param_groups[0]["lr"]

            # Logging
            logger.info(
                f"Epoch {epoch}/{self._config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # MLflow logging
            if mlflow_active:
                self._log_mlflow(epoch, train_metrics, val_metrics, current_lr)

            # Checkpoint: save last
            self._save_checkpoint(epoch, val_metrics["loss"], "last.pt")

            # Best model check
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                result.best_val_loss = best_val_loss
                result.best_val_accuracy = val_metrics["accuracy"]
                result.best_epoch = epoch
                self._save_checkpoint(epoch, val_metrics["loss"], "best.pt")
                patience_counter = 0
                logger.info(f"  ✓ New best model (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self._config.early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {patience_counter} epochs)"
                )
                break

        result.total_epochs = epoch
        result.training_time_sec = time.time() - t_start

        logger.info(
            f"Training complete | Best epoch: {result.best_epoch} | "
            f"Best val loss: {result.best_val_loss:.4f} | "
            f"Best val acc: {result.best_val_accuracy:.4f} | "
            f"Time: {result.training_time_sec:.1f}s"
        )

        return result

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self._model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        step = 0

        self._optimizer.zero_grad()

        for batch_idx, (features, labels, masks) in enumerate(self._train_loader):
            features = features.to(self._device, non_blocking=True)
            labels = labels.to(self._device, non_blocking=True)
            masks = masks.to(self._device, non_blocking=True)

            with autocast(
                device_type=self._device.type,
                enabled=self._config.use_amp,
            ):
                output = self._model(features, mask=masks)
                loss = self._criterion(output["logits"], labels)
                loss = loss / self._config.gradient_accumulation_steps

            self._scaler.scale(loss).backward()

            if (batch_idx + 1) % self._config.gradient_accumulation_steps == 0:
                self._scaler.unscale_(self._optimizer)
                nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    self._config.max_grad_norm,
                )
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
                step += 1

            total_loss += loss.item() * self._config.gradient_accumulation_steps
            preds = output["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(len(self._train_loader), 1)
        accuracy = correct / max(total, 1)

        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> dict[str, float]:
        """Run one validation epoch."""
        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels, masks in self._val_loader:
            features = features.to(self._device, non_blocking=True)
            labels = labels.to(self._device, non_blocking=True)
            masks = masks.to(self._device, non_blocking=True)

            output = self._model(features, mask=masks)
            loss = self._criterion(output["logits"], labels)

            total_loss += loss.item()
            preds = output["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(len(self._val_loader), 1)
        accuracy = correct / max(total, 1)

        return {"loss": avg_loss, "accuracy": accuracy}

    def _save_checkpoint(self, epoch: int, val_loss: float, filename: str) -> None:
        """Save model checkpoint."""
        path = self._ckpt_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": self._scheduler.state_dict(),
                "val_loss": val_loss,
                "config": self._config,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint and return the epoch number."""
        ckpt = torch.load(str(path), map_location=self._device, weights_only=False)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        epoch: int = int(ckpt["epoch"])
        logger.info(f"Loaded checkpoint: {path} (epoch {epoch})")
        return epoch

    def _init_mlflow(self) -> bool:
        """Try to initialize MLflow tracking."""
        try:
            import mlflow

            mlflow.set_experiment(self._config.experiment_name)
            mlflow.start_run()
            mlflow.log_params(
                {
                    "epochs": self._config.epochs,
                    "batch_size": self._config.batch_size,
                    "learning_rate": self._config.learning_rate,
                    "weight_decay": self._config.weight_decay,
                    "model_params": sum(p.numel() for p in self._model.parameters()),
                    "device": str(self._device),
                }
            )
            logger.info("MLflow tracking enabled.")
            return True
        except ImportError:
            logger.info("MLflow not installed. Skipping experiment tracking.")
            return False

    def _log_mlflow(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        lr: float,
    ) -> None:
        """Log metrics to MLflow."""
        try:
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
        except Exception:
            pass

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set all random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
