"""Evaluate a trained ASL Citizen checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.checkpoint import load_checkpoint
from src.asl_citizen.config import BEST_CHECKPOINT_PATH, METRICS_DIR, default_config
from src.asl_citizen.dataset import build_dataloader
from src.asl_citizen.metrics import batch_accuracy, top_k_accuracy
from src.asl_citizen.model import build_model
from src.asl_citizen.utils import choose_batch_size
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASL Citizen 100 checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=BEST_CHECKPOINT_PATH)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    return parser.parse_args()


@torch.no_grad()
def evaluate(checkpoint_path: Path, split: str) -> dict:
    device = get_device()
    payload, cfg, class_to_idx, idx_to_class = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    loader, dataset = build_dataloader(
        split,
        cfg,
        batch_size=choose_batch_size(cfg.batch_size, device),
        shuffle=False,
    )

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch in loader:
        clips = batch["video"].to(device)
        labels = batch["label"].to(device)
        logits = model(clips)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    results = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "samples": len(dataset),
        "top1_accuracy": batch_accuracy(logits_cat, labels_cat),
        "top5_accuracy": top_k_accuracy(logits_cat, labels_cat, k=5),
        "num_classes": len(class_to_idx),
    }
    return results


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"No checkpoint at {args.checkpoint}")
        return
    results = evaluate(args.checkpoint, args.split)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / f"evaluation_{args.split}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
