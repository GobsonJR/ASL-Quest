"""Upgrade the legacy checkpoint with inference metadata without retraining."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.config import BEST_MODEL_PATH, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, LEGACY_MODEL_PATH
from src.utils.checkpoint import save_checkpoint


def main() -> None:
    source = LEGACY_MODEL_PATH
    if not source.exists():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    class_names = payload.get("class_names") or payload.get("classes")
    if not class_names:
        raise ValueError("Legacy checkpoint has no class mapping.")
    payload["class_names"] = list(class_names)
    payload["classes"] = list(class_names)
    payload["num_classes"] = len(class_names)
    payload["architecture"] = "resnet18"
    payload["image_size"] = IMAGE_SIZE
    payload["normalize_mean"] = list(IMAGENET_MEAN)
    payload["normalize_std"] = list(IMAGENET_STD)
    payload["source"] = "legacy_asl_resnet18_best.pth"
    if "accuracy" in payload:
        payload["val_accuracy"] = payload["accuracy"]
        payload["note"] = (
            "Legacy validation accuracy from the original 80/20 random split. "
            "This is not a held-out test metric."
        )
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not BEST_MODEL_PATH.exists():
        save_checkpoint(payload, BEST_MODEL_PATH)
        print(f"Wrote {BEST_MODEL_PATH}")
    else:
        print(f"{BEST_MODEL_PATH} already exists; left unchanged.")
    print(f"Classes: {payload['class_names']}")
    print(f"Legacy accuracy field: {payload.get('accuracy')}")


if __name__ == "__main__":
    main()
