from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.config import (
    ARCHITECTURE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def build_checkpoint(
    model: torch.nn.Module,
    class_names: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "class_names": list(class_names),
        "classes": list(class_names),
        "num_classes": len(class_names),
        "architecture": ARCHITECTURE,
        "image_size": IMAGE_SIZE,
        "normalize_mean": list(IMAGENET_MEAN),
        "normalize_std": list(IMAGENET_STD),
    }
    if extra:
        payload.update(extra)
    return payload


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint format: {path}")
    class_names = payload.get("class_names") or payload.get("classes")
    if not class_names:
        raise ValueError(f"Checkpoint is missing class mapping: {path}")
    payload["class_names"] = list(class_names)
    payload["classes"] = list(class_names)
    payload.setdefault("num_classes", len(class_names))
    payload.setdefault("architecture", ARCHITECTURE)
    payload.setdefault("image_size", IMAGE_SIZE)
    payload.setdefault("normalize_mean", list(IMAGENET_MEAN))
    payload.setdefault("normalize_std", list(IMAGENET_STD))
    return payload
