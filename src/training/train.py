from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    BATCH_SIZE,
    DATASET_DIR,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    OUTPUTS_DIR,
    SEED,
    V2_BEST_MODEL_PATH,
    V2_LAST_MODEL_PATH,
    WEIGHT_DECAY,
)
from src.data.dataset import build_dataloaders
from src.models.resnet18 import build_resnet18
from src.utils.checkpoint import build_checkpoint, save_checkpoint
from src.utils.device import print_device_info, seed_everything


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    correct = 0
    total = 0
    iterator = tqdm(loader, leave=False, desc="train" if training else "eval")
    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, enabled=use_amp
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        running_loss += loss.item() * labels.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        iterator.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def plot_curves(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASL ResNet18 classifier")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--best-model", type=Path, default=V2_BEST_MODEL_PATH)
    parser.add_argument("--last-model", type=Path, default=V2_LAST_MODEL_PATH)
    parser.add_argument("--history", type=Path, default=OUTPUTS_DIR / "training_history.json")
    parser.add_argument("--curves", type=Path, default=OUTPUTS_DIR / "training_curves.png")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    device = print_device_info()

    train_loader, val_loader, _, class_names, splits = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_per_class=args.max_per_class,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train images: {len(splits['train'])}")
    print(f"Val images:   {len(splits['val'])}")
    print(f"Test images:  {len(splits['test'])}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Epochs:       {args.epochs}")

    model = build_resnet18(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"Mixed precision: {'enabled' if use_amp else 'disabled'}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = -1.0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler=scaler, use_amp=use_amp
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%")

        extra = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "best_val_accuracy": max(best_acc, val_acc),
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
            "mixed_precision": use_amp,
            "training_policy": "sequential per-class split; no horizontal flip",
        }
        save_checkpoint(build_checkpoint(model, class_names, extra), args.last_model)

        if val_acc > best_acc:
            best_acc = val_acc
            stale = 0
            extra["best_val_accuracy"] = best_acc
            extra["accuracy"] = best_acc
            save_checkpoint(build_checkpoint(model, class_names, extra), args.best_model)
            print(f"Best model saved ({best_acc:.2f}%) -> {args.best_model}")
        else:
            stale += 1
            print(f"No improvement ({stale}/{args.patience})")
            if stale >= args.patience:
                print("Early stopping.")
                break

    args.history.parent.mkdir(parents=True, exist_ok=True)
    args.history.write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_curves(history, args.curves)
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Best model: {args.best_model}")
    print(f"Last model: {args.last_model}")
    print("=" * 60)


if __name__ == "__main__":
    main()
