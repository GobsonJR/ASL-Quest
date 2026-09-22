"""Checkpoint build/load helpers for the WLASL100 word model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.word_model.config import WordModelConfig, default_config

REQUIRED_CHECKPOINT_KEYS = (
    "model_state_dict",
    "class_names",
    "num_classes",
    "architecture",
    "backbone",
    "rnn_type",
    "hidden_size",
    "num_layers",
    "dropout",
    "frame_count",
    "image_size",
    "freeze_backbone",
    "seed",
    "full_config",
)


def config_to_dict(cfg: WordModelConfig) -> dict[str, Any]:
    return {
        "dataset_root": str(cfg.dataset_root),
        "manifest_path": str(cfg.manifest_path),
        "classes_path": str(cfg.classes_path),
        "video_root": str(cfg.video_root),
        "outputs_dir": str(cfg.outputs_dir),
        "models_dir": str(cfg.models_dir),
        "num_classes": cfg.num_classes,
        "frame_count": cfg.frame_count,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "learning_rate": cfg.learning_rate,
        "num_epochs": cfg.num_epochs,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "backbone": cfg.backbone,
        "rnn_type": cfg.rnn_type,
        "freeze_backbone": cfg.freeze_backbone,
        "weight_decay": cfg.weight_decay,
        "seed": cfg.seed,
        "imagenet_mean": list(cfg.imagenet_mean),
        "imagenet_std": list(cfg.imagenet_std),
    }


def config_from_dict(data: dict[str, Any]) -> WordModelConfig:
    payload = dict(data)
    for key in ("dataset_root", "manifest_path", "classes_path", "video_root", "outputs_dir", "models_dir"):
        if key in payload and payload[key] is not None:
            payload[key] = Path(payload[key])
    if "imagenet_mean" in payload:
        payload["imagenet_mean"] = tuple(payload["imagenet_mean"])
    if "imagenet_std" in payload:
        payload["imagenet_std"] = tuple(payload["imagenet_std"])
    allowed = {field for field in WordModelConfig.__dataclass_fields__}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return WordModelConfig(**filtered)


def validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise ValueError(
            "Checkpoint is missing required configuration fields: "
            f"{missing}. Retrain or upgrade the checkpoint before evaluation/inference."
        )
    if not payload.get("full_config"):
        raise ValueError("Checkpoint is missing full_config. Retrain or upgrade the checkpoint.")


def build_word_checkpoint(
    model: nn.Module,
    class_names: list[str],
    cfg: WordModelConfig,
    *,
    epoch: int,
    best_epoch: int,
    val_accuracy: float,
    best_val_accuracy: float,
    optimizer: Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    full_config = config_to_dict(cfg)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "architecture": f"{cfg.backbone}_{cfg.rnn_type}",
        "backbone": cfg.backbone,
        "rnn_type": cfg.rnn_type,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "frame_count": cfg.frame_count,
        "image_size": cfg.image_size,
        "freeze_backbone": cfg.freeze_backbone,
        "seed": cfg.seed,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "val_accuracy": val_accuracy,
        "best_val_accuracy": best_val_accuracy,
        "full_config": full_config,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    return payload


def save_word_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_word_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> tuple[dict[str, Any], WordModelConfig, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Word-model checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid word-model checkpoint format: {path}")
    validate_checkpoint_payload(payload)
    class_names = list(payload["class_names"])
    cfg = config_from_dict(payload["full_config"])
    return payload, cfg, class_names


def load_word_checkpoint_if_present(path: Path, map_location: str | torch.device = "cpu") -> tuple[dict[str, Any], WordModelConfig, list[str]] | None:
    if not path.exists():
        return None
    return load_word_checkpoint(path, map_location=map_location)
