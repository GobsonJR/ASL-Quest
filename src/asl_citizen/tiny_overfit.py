"""Tiny-overfit diagnostic for ASL Citizen (NOT full 100-class training)."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.config import default_config
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

DIAG_DIR = ROOT / "outputs" / "asl_citizen_100" / "diagnostics"
SUCCESS_THRESHOLD = 0.95
STRONG_THRESHOLD = 0.98
TINY_CLASSES = 10
VIDEOS_PER_CLASS = 5
NUM_FRAMES = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny 10-class overfit gate (no 100-class training)")
    parser.add_argument("--backbone-mode", choices=("frozen", "layer4", "full"), required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0, help="0.0 for memorization; production default is 0.3")
    return parser.parse_args()


def select_tiny_rows(num_classes: int = TINY_CLASSES, videos_per_class: int = VIDEOS_PER_CLASS) -> tuple[list[dict], list[str]]:
    cfg = default_config()
    class_to_idx, _ = load_class_mappings(cfg.class_to_idx_path, cfg.idx_to_class_path)
    rows = load_manifest_rows(cfg.train_manifest)
    by_gloss: dict[str, list[dict]] = {}
    for row in rows:
        by_gloss.setdefault(row["gloss"], []).append(row)
    glosses = sorted(by_gloss.keys())[:num_classes]
    tiny: list[dict] = []
    for new_idx, gloss in enumerate(glosses):
        for source in by_gloss[gloss][:videos_per_class]:
            item = dict(source)
            item["class_idx"] = new_idx
            item["gloss"] = gloss
            tiny.append(item)
            _ = class_to_idx[gloss]
    if len(tiny) != num_classes * videos_per_class:
        raise RuntimeError(f"Expected {num_classes * videos_per_class} samples, got {len(tiny)}")
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
) -> dict:
    seed_pipeline(42)
    device = print_device_info()
    cfg = default_config()
    rows, glosses = select_tiny_rows()
    cached = cache_clips(rows, glosses, cfg)
    loader = DataLoader(cached, batch_size=batch_size, shuffle=True)

    model = ASLCitizenResNet18GRU(
        num_classes=len(glosses),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=dropout,
        freeze_backbone=backbone_mode == "frozen",
        pretrained=True,
        backbone_train_mode=backbone_mode,  # type: ignore[arg-type]
    ).to(device)

    trainability = encoder_trainability_report(model)
    print(json.dumps(trainability, indent=2))
    if model.num_classes != 10:
        raise RuntimeError(f"Tiny experiment must use 10 classes, got {model.num_classes}")

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
            if logits.shape[1] != 10:
                raise RuntimeError(f"Expected logits [B,10], got {tuple(logits.shape)}")
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
        pred_info = unique_pred_report(preds, 10)
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

    result = {
        "backbone_mode": backbone_mode,
        "num_classes": 10,
        "num_samples": len(rows),
        "glosses": glosses,
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
    )
    out = DIAG_DIR / f"tiny_overfit_{args.backbone_mode}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("backbone_mode", "best_train_accuracy", "final_train_accuracy", "gate_passed", "epochs_ran")}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
