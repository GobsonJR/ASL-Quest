"""Checkpoint save/load for ASL Citizen models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.asl_citizen.config import ASLCitizenConfig, default_config

REQUIRED_KEYS = (
    "model_state_dict",
    "class_to_idx",
    "idx_to_class",
    "num_classes",
    "num_frames",
    "image_size",
    "imagenet_mean",
    "imagenet_std",
    "architecture",
    "epoch",
    "val_accuracy",
    "full_config",
)


def config_to_dict(cfg: ASLCitizenConfig) -> dict[str, Any]:
    return {
        "dataset_root": str(cfg.dataset_root),
        "video_root": str(cfg.video_root),
        "data_dir": str(cfg.data_dir),
        "train_manifest": str(cfg.train_manifest),
        "val_manifest": str(cfg.val_manifest),
        "test_manifest": str(cfg.test_manifest),
        "num_classes": cfg.num_classes,
        "num_frames": cfg.num_frames,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "num_epochs": cfg.num_epochs,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
        "seed": cfg.seed,
        "freeze_backbone": cfg.freeze_backbone,
        "backbone_train_mode": getattr(cfg, "backbone_train_mode", "frozen"),
        "head_learning_rate": getattr(cfg, "head_learning_rate", cfg.learning_rate),
        "backbone_learning_rate": getattr(cfg, "backbone_learning_rate", 1e-4),
        "imagenet_mean": list(cfg.imagenet_mean),
        "imagenet_std": list(cfg.imagenet_std),
    }


def config_from_dict(data: dict[str, Any]) -> ASLCitizenConfig:
    payload = dict(data)
    for key in (
        "dataset_root",
        "video_root",
        "data_dir",
        "train_manifest",
        "val_manifest",
        "test_manifest",
        "class_to_idx_path",
        "idx_to_class_path",
        "outputs_dir",
        "checkpoints_dir",
    ):
        if key in payload and payload[key] is not None:
            payload[key] = Path(payload[key])
    if "imagenet_mean" in payload:
        payload["imagenet_mean"] = tuple(payload["imagenet_mean"])
    if "imagenet_std" in payload:
        payload["imagenet_std"] = tuple(payload["imagenet_std"])
    allowed = {field for field in ASLCitizenConfig.__dataclass_fields__}
    filtered = {k: v for k, v in payload.items() if k in allowed}
    return ASLCitizenConfig(**filtered)


def build_checkpoint(
    model: nn.Module,
    class_to_idx: dict[str, int],
    idx_to_class: dict[str, str],
    cfg: ASLCitizenConfig,
    *,
    epoch: int,
    val_accuracy: float,
    val_top5: float | None = None,
    optimizer: Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": dict(class_to_idx),
        "idx_to_class": dict(idx_to_class),
        "num_classes": cfg.num_classes,
        "num_frames": cfg.num_frames,
        "image_size": cfg.image_size,
        "imagenet_mean": list(cfg.imagenet_mean),
        "imagenet_std": list(cfg.imagenet_std),
        "architecture": "resnet18_gru",
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "val_top5_accuracy": val_top5,
        "full_config": config_to_dict(cfg),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    return payload


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def validate_checkpoint(payload: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Checkpoint missing keys: {missing}")


def load_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], ASLCitizenConfig, dict[str, int], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint format: {path}")
    validate_checkpoint(payload)
    cfg = config_from_dict(payload["full_config"])
    class_to_idx = {str(k): int(v) for k, v in payload["class_to_idx"].items()}
    idx_to_class = {str(k): str(v) for k, v in payload["idx_to_class"].items()}
    return payload, cfg, class_to_idx, idx_to_class
