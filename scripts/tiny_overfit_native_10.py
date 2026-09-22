"""Tiny-overfit diagnostic for the asl_citizen_native_10 experiment.

Mirrors src/asl_citizen/tiny_overfit.py's memorization gate, but points at the new
data/asl_citizen_native_10 manifest instead of data/asl_citizen_100. Does not modify
src/asl_citizen/tiny_overfit.py or any asl_citizen_100 artifact; reuses the shared,
experiment-agnostic dataset/model/preprocessing/checkpoint infrastructure as-is.

Goal: verify the 10-class temporal model (ResNet18 encoder + GRU) can memorize a small,
controlled subset of native_10 training clips before any longer training run is
considered.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_asl_citizen_native_10_manifest import native_10_config
from src.asl_citizen.dataset import ASLCitizenDataset
from src.asl_citizen.model import (
    ASLCitizenResNet18GRU,
    build_optimizer,
    encoder_trainability_report,
    gradient_flow_report,
)
from src.asl_citizen.preprocessing import eval_spatial_transforms
from src.asl_citizen.utils import load_class_mappings, load_manifest_rows, seed_pipeline
from src.utils.device import get_device, print_device_info

DIAG_DIR = ROOT / "outputs" / "asl_citizen_native_10" / "diagnostics"
SUCCESS_THRESHOLD = 0.95
STRONG_THRESHOLD = 0.98
NUM_FRAMES = 16
VIDEOS_PER_CLASS = 5  # matches src/asl_citizen/tiny_overfit.py's tiny-subset size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="native_10 tiny-overfit memorization gate")
    parser.add_argument("--backbone-mode", choices=("frozen", "layer4", "full"), default="frozen")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0, help="0.0 for memorization; production default is 0.3")
    parser.add_argument("--videos-per-class", type=int, default=VIDEOS_PER_CLASS)
    return parser.parse_args()


def select_tiny_rows(cfg, videos_per_class: int) -> tuple[list[dict], list[str]]:
    class_to_idx, _ = load_class_mappings(cfg.class_to_idx_path, cfg.idx_to_class_path)
    rows = load_manifest_rows(cfg.train_manifest)
    by_gloss: dict[str, list[dict]] = {}
    for row in rows:
        by_gloss.setdefault(row["gloss"], []).append(row)

    glosses = sorted(class_to_idx.keys())
    if set(glosses) != set(by_gloss.keys()):
        raise RuntimeError(
            f"Train manifest glosses {sorted(by_gloss.keys())} do not match "
            f"class_to_idx glosses {glosses}"
        )

    tiny: list[dict] = []
    for new_idx, gloss in enumerate(glosses):
        available = by_gloss[gloss]
        if len(available) < videos_per_class:
            raise RuntimeError(
                f"Gloss {gloss} has only {len(available)} train videos, "
                f"need {videos_per_class}"
            )
        for source in available[:videos_per_class]:
            item = dict(source)
            item["class_idx"] = new_idx
            item["gloss"] = gloss
            tiny.append(item)
    if len(tiny) != len(glosses) * videos_per_class:
        raise RuntimeError(f"Expected {len(glosses) * videos_per_class} samples, got {len(tiny)}")
    return tiny, glosses


def cache_clips(rows: list[dict], glosses: list[str], cfg) -> TensorDataset:
    class_to_idx = {gloss: index for index, gloss in enumerate(glosses)}
    transform = eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    dataset = ASLCitizenDataset(
        rows,
        video_root=cfg.video_root,
        class_to_idx=class_to_idx,
        num_frames=NUM_FRAMES,
        transform=transform,
        training=False,
        jitter_frames=0,
        seed=cfg.seed,
    )
    clips = []
    labels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        clips.append(sample["video"])
        labels.append(int(sample["label"]))
    return TensorDataset(torch.stack(clips, dim=0), torch.tensor(labels, dtype=torch.long))


def unique_pred_report(preds: list[int], num_classes: int) -> dict:
    counts = Counter(preds)
    return {
        "unique_predicted_classes": len(counts),
        "distribution": {str(k): v for k, v in sorted(counts.items())},
        "collapse_warning": len(counts) == 1 and num_classes > 1,
    }


def run_overfit(
    *,
    backbone_mode: str,
    epochs: int,
    batch_size: int,
    head_lr: float,
    backbone_lr: float,
    dropout: float,
    videos_per_class: int,
) -> dict:
    seed_pipeline(42)
    device = print_device_info()
    cfg = native_10_config()
    rows, glosses = select_tiny_rows(cfg, videos_per_class)
    num_classes = len(glosses)
    cached = cache_clips(rows, glosses, cfg)
    loader = DataLoader(cached, batch_size=batch_size, shuffle=True)

    model = ASLCitizenResNet18GRU(
        num_classes=num_classes,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=dropout,
        freeze_backbone=backbone_mode == "frozen",
        pretrained=True,
        backbone_train_mode=backbone_mode,  # type: ignore[arg-type]
    ).to(device)

    trainability = encoder_trainability_report(model)
    print(json.dumps(trainability, indent=2))
    if model.num_classes != num_classes:
        raise RuntimeError(f"Tiny experiment must use {num_classes} classes, got {model.num_classes}")

    optimizer = build_optimizer(model, head_lr=head_lr, backbone_lr=backbone_lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    gradient_check: dict | None = None
    best_acc = 0.0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        preds: list[int] = []
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(clips)
            if logits.shape[1] != num_classes:
                raise RuntimeError(f"Expected logits [B,{num_classes}], got {tuple(logits.shape)}")
            loss = criterion(logits, labels)
            loss.backward()
            if gradient_check is None:
                gradient_check = gradient_flow_report(model)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            batch_preds = logits.argmax(1)
            preds.extend(batch_preds.detach().cpu().tolist())
            correct += (batch_preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / max(total, 1)
        pred_info = unique_pred_report(preds, num_classes)
        entry = {
            "epoch": epoch,
            "loss": running_loss / max(total, 1),
            "accuracy": acc,
            "unique_predicted_classes": pred_info["unique_predicted_classes"],
            "prediction_distribution": pred_info["distribution"],
            "collapse_warning": pred_info["collapse_warning"],
        }
        history.append(entry)
        best_acc = max(best_acc, acc)
        print(
            f"[{backbone_mode}] epoch {epoch}/{epochs} loss={entry['loss']:.4f} "
            f"acc={acc:.2%} unique_preds={pred_info['unique_predicted_classes']}"
            + (" COLLAPSE" if pred_info["collapse_warning"] else "")
        )
        if acc >= SUCCESS_THRESHOLD:
            stopped_early = True
            break

    checkpoint_path = DIAG_DIR / f"tiny_overfit_{backbone_mode}_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": {gloss: idx for idx, gloss in enumerate(glosses)},
            "idx_to_class": {str(idx): gloss for idx, gloss in enumerate(glosses)},
            "num_classes": num_classes,
            "backbone_mode": backbone_mode,
        },
        checkpoint_path,
    )

    result = {
        "experiment_name": "asl_citizen_native_10",
        "backbone_mode": backbone_mode,
        "num_classes": num_classes,
        "glosses": glosses,
        "num_samples": len(rows),
        "videos_per_class": videos_per_class,
        "epochs_requested": epochs,
        "epochs_ran": len(history),
        "stopped_early_at_95": stopped_early,
        "batch_size": batch_size,
        "head_lr": head_lr,
        "backbone_lr": backbone_lr,
        "dropout": dropout,
        "deterministic_eval_transforms": True,
        "temporal_jitter": False,
        "num_frames": NUM_FRAMES,
        "device": str(device),
        "trainability": trainability,
        "gradient_flow": gradient_check,
        "history": history,
        "final_train_accuracy": history[-1]["accuracy"] if history else 0.0,
        "best_train_accuracy": best_acc,
        "final_loss": history[-1]["loss"] if history else None,
        "gate_passed": best_acc >= SUCCESS_THRESHOLD,
        "strong_success": best_acc >= STRONG_THRESHOLD,
        "checkpoint_path": str(checkpoint_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result


def main() -> None:
    args = parse_args()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    result = run_overfit(
        backbone_mode=args.backbone_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        dropout=args.dropout,
        videos_per_class=args.videos_per_class,
    )
    out = DIAG_DIR / f"tiny_overfit_{args.backbone_mode}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("backbone_mode", "best_train_accuracy", "final_train_accuracy", "gate_passed", "epochs_ran")}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
