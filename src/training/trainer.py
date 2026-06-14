
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.evaluation.metrics import compute_metrics

if TYPE_CHECKING:
    from src.training.callbacks import EarlyStopping, ModelCheckpoint

_METRIC_KEYS = {"loss", "accuracy", "precision", "recall", "f1", "roc_auc", "eer"}
_LOG_METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "eer")


def _batch_size(targets: torch.Tensor) -> int:
    return targets.shape[0]


class Trainer:
  

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        max_epochs: int = 50,
        grad_clip_norm: float | None = 5.0,
        use_amp: bool = True,
        early_stopping: "EarlyStopping | None" = None,
        checkpoint: "ModelCheckpoint | None" = None,
        checkpoint_metric: str = "loss",
        log_dir: Path | str | None = None,
        input_key: str = "spectrogram",
        threshold: float = 0.5,
        verbose: bool = True,
    ) -> None:
        if checkpoint_metric not in _METRIC_KEYS:
            raise ValueError(f"checkpoint_metric must be one of {sorted(_METRIC_KEYS)}, got {checkpoint_metric!r}")

        if device is None:
            from src.data.dataloaders import get_device

            device = get_device()
        self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint
        self.checkpoint_metric = checkpoint_metric
        self.input_key = input_key
        self.threshold = threshold
        self.verbose = verbose

        self.use_amp = use_amp and self.device.type in ("cuda", "mps")
        self.scaler = torch.amp.GradScaler(device=self.device.type) if self.use_amp else None

        self.writer = SummaryWriter(log_dir=str(log_dir)) if log_dir is not None else None

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        for key in _LOG_METRIC_KEYS:
            self.history[f"val_{key}"] = []

  
    def _move_batch(self, batch: dict) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor]:
        inputs = batch[self.input_key]
        if isinstance(inputs, dict):
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        else:
            inputs = inputs.to(self.device)
        targets = batch["label"].to(self.device)
        return inputs, targets

    def _autocast_ctx(self):
        if self.use_amp:
            return torch.autocast(device_type=self.device.type, dtype=torch.float16)
        return contextlib.nullcontext()

    def train_one_epoch(self) -> float:
        """Run one training epoch; returns the sample-weighted average loss."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in self.train_loader:
            inputs, targets = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast_ctx():
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            bs = _batch_size(targets)
            total_loss += loss.item() * bs
            n_samples += bs

        return total_loss / n_samples

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        y_true_chunks: list[np.ndarray] = []
        y_prob_chunks: list[np.ndarray] = []

        for batch in loader:
            inputs, targets = self._move_batch(batch)
            with self._autocast_ctx():
                logits = self.model(inputs)
                loss = self.loss_fn(logits, targets)

            bs = _batch_size(targets)
            total_loss += loss.item() * bs
            n_samples += bs

            probs = torch.sigmoid(logits.float()).view(-1)
            y_true_chunks.append(targets.detach().cpu().numpy())
            y_prob_chunks.append(probs.detach().cpu().numpy())

        y_true = np.concatenate(y_true_chunks)
        y_prob = np.concatenate(y_prob_chunks)
        metrics = compute_metrics(y_true, y_prob, threshold=self.threshold)
        return total_loss / n_samples, metrics

    def _build_checkpoint_state(self, epoch: int) -> dict:
        return {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "early_stopping_state": self.early_stopping.state_dict() if self.early_stopping is not None else None,
            "checkpoint_state": self.checkpoint.state_dict() if self.checkpoint is not None else None,
            "history": self.history,
        }

    def save_checkpoint(self, path: Path | str) -> None:
        torch.save(self._build_checkpoint_state(epoch=-1), path)

    def load_checkpoint(self, path: Path | str) -> int:

        state = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scheduler is not None and state["scheduler_state"] is not None:
            self.scheduler.load_state_dict(state["scheduler_state"])
        if self.scaler is not None and state["scaler_state"] is not None:
            self.scaler.load_state_dict(state["scaler_state"])
        if self.early_stopping is not None and state["early_stopping_state"] is not None:
            self.early_stopping.load_state_dict(state["early_stopping_state"])
        if self.checkpoint is not None and state["checkpoint_state"] is not None:
            self.checkpoint.load_state_dict(state["checkpoint_state"])

        self.history = state.get("history", self.history)
        return state["epoch"] + 1

   
    def fit(self, resume_from: Path | str | None = None) -> dict[str, list[float]]:
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)
            if self.verbose:
                print(f"Resumed from {resume_from} at epoch {start_epoch + 1}")

        for epoch in range(start_epoch, self.max_epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader)

            if self.scheduler is not None:
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            for key in _LOG_METRIC_KEYS:
                self.history[f"val_{key}"].append(val_metrics[key])

            if self.writer is not None:
                self.writer.add_scalar("train/loss", train_loss, epoch)
                self.writer.add_scalar("val/loss", val_loss, epoch)
                for key in _LOG_METRIC_KEYS:
                    self.writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
                self.writer.add_scalar("lr", lr, epoch)

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{self.max_epochs} | "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f} val_eer={val_metrics['eer']:.4f} "
                    f"val_auc={val_metrics['roc_auc']:.4f} lr={lr:.2e}"
                )

            score = val_loss if self.checkpoint_metric == "loss" else val_metrics[self.checkpoint_metric]

            if self.checkpoint is not None:
                state = self._build_checkpoint_state(epoch)
                is_best = self.checkpoint.step(score, state)
                if self.verbose and is_best:
                    print(f"  -> new best {self.checkpoint_metric}={score:.4f}, saved to {self.checkpoint.best_path}")

            if self.early_stopping is not None:
                if self.early_stopping.step(score, epoch):
                    if self.verbose:
                        print(
                            f"Early stopping at epoch {epoch + 1} "
                            f"(best {self.checkpoint_metric}={self.early_stopping.best_score:.4f} "
                            f"at epoch {self.early_stopping.best_epoch + 1})"
                        )
                    break

        if self.writer is not None:
            self.writer.close()

        return self.history
