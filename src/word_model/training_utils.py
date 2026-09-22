"""Training helpers: early stopping, history, reproducibility."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.utils.device import seed_everything


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        self.best_val_accuracy = float("-inf")
        self.best_epoch = 0
        self.stale_epochs = 0

    def step(self, val_accuracy: float, epoch: int) -> bool:
        """Return True when validation accuracy improved."""
        if val_accuracy > self.best_val_accuracy + self.min_delta:
            self.best_val_accuracy = val_accuracy
            self.best_epoch = epoch
            self.stale_epochs = 0
            return True
        self.stale_epochs += 1
        return False

    def should_stop(self) -> bool:
        return self.stale_epochs >= self.patience


def seed_word_experiment(seed: int) -> None:
    seed_everything(seed)
    # Keep benchmark enabled on CUDA for practical training speed.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def gpu_memory_snapshot() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"gpu_memory_allocated_mb": None, "gpu_memory_reserved_mb": None}
    return {
        "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
        "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
    }


def build_history_entry(
    *,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
    learning_rate: float,
    epoch_duration_seconds: float,
) -> dict[str, Any]:
    entry = {
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
        "train_accuracy": round(train_accuracy, 4),
        "val_loss": round(val_loss, 6),
        "val_accuracy": round(val_accuracy, 4),
        "learning_rate": learning_rate,
        "epoch_duration_seconds": round(epoch_duration_seconds, 3),
        "train_val_accuracy_gap": round(train_accuracy - val_accuracy, 4),
    }
    entry.update(gpu_memory_snapshot())
    return entry


def resolve_training_history_path(
    default_path: Path,
    *,
    experiment_name: str | None = None,
    replace_history: bool = False,
) -> Path:
    if experiment_name:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in experiment_name)
        return default_path.with_name(f"training_history_{safe}.json")
    if default_path.exists() and not replace_history:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return default_path.with_name(f"training_history_{stamp}.json")
    return default_path


def write_training_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
