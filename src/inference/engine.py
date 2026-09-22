from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.config import BEST_MODEL_PATH, CONFIDENCE_THRESHOLD, LEGACY_MODEL_PATH
from src.data.preprocessing import eval_transforms
from src.models.resnet18 import build_resnet18
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device
from src.utils.smoothing import softmax_topk


class Predictor:
    def __init__(self, model_path: Path | None = None) -> None:
        self.device = get_device()
        path = model_path or _resolve_model_path()
        payload = load_checkpoint(path, map_location=self.device)
        self.class_names: list[str] = payload["class_names"]
        self.image_size = int(payload["image_size"])
        self.architecture = payload.get("architecture", "resnet18")
        self.model_path = path
        self.model = build_resnet18(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = eval_transforms(self.image_size)

    def predict_pil(self, image: Image.Image, top_k: int = 3) -> dict[str, Any]:
        rgb = image.convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        top = softmax_topk(logits, self.class_names, k=top_k)
        best = top[0]
        return {
            "prediction": best["label"],
            "confidence": best["confidence"],
            "top_predictions": top,
            "low_confidence": float(best["confidence"]) < CONFIDENCE_THRESHOLD,
        }

    def predict_path(self, image_path: Path, top_k: int = 3) -> dict[str, Any]:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.load()
                return self.predict_pil(image, top_k=top_k)
        except Exception as exc:
            raise ValueError(f"Could not read image: {image_path}") from exc


def _resolve_model_path() -> Path:
    if BEST_MODEL_PATH.exists():
        return BEST_MODEL_PATH
    if LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    raise FileNotFoundError(
        f"No trained model found. Expected {BEST_MODEL_PATH} or {LEGACY_MODEL_PATH}."
    )


_PREDICTOR: Predictor | None = None


def get_predictor(model_path: Path | None = None) -> Predictor:
    global _PREDICTOR
    if _PREDICTOR is None or (model_path and model_path != _PREDICTOR.model_path):
        _PREDICTOR = Predictor(model_path=model_path)
    return _PREDICTOR
