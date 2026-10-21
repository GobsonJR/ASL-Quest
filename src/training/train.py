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
from src.data.preprocessing import eval_transforms, robust_train_transforms
from src.data.webcam import WebcamCropDataset, mixed_train_loader
from src.models.resnet18 import build_resnet18
from src.utils.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
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
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=None,
        help="DataLoader workers for validation/test (default: same as --num-workers). Use 0 on low-memory machines.",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="start from an existing checkpoint (same architecture and class order) instead of ImageNet weights, e.g. to fine-tune with --augment robust",
    )
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--best-model", type=Path, default=V2_BEST_MODEL_PATH)
    parser.add_argument("--last-model", type=Path, default=V2_LAST_MODEL_PATH)
    parser.add_argument("--history", type=Path, default=OUTPUTS_DIR / "training_history.json")
    parser.add_argument("--curves", type=Path, default=OUTPUTS_DIR / "training_curves.png")
    parser.add_argument(
        "--augment",
        choices=["default", "robust"],
        default="default",
        help="training augmentation policy; 'robust' adds mirroring, scale/rotation, lighting and occlusion (see src/data/preprocessing.py)",
    )
    parser.add_argument("--webcam-export", type=Path, default=None, help="dataset/webcam_export produced by scripts/export_webcam_dataset.py: mix real webcam crops into training")
    parser.add_argument("--webcam-share", type=float, default=0.3, help="fraction of each epoch drawn from webcam crops")
    parser.add_argument("--epoch-samples", type=int, default=20000, help="images per epoch when mixing webcam data")
    parser.add_argument("--webcam-val-weight", type=float, default=0.5, help="weight of unseen-room webcam validation accuracy in model selection (the rest is the original validation split)")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = print_device_info()

    train_loader, val_loader, _, class_names, splits = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_per_class=args.max_per_class,
        augment=args.augment,
        eval_workers=args.eval_workers,
    )
    webcam_val_loader = None
    webcam_info: dict | None = None
    if args.webcam_export is not None:
        # Webcam crops are augmented like the (robust) original images; validation crops never are.
        webcam_train = WebcamCropDataset(args.webcam_export, "train", robust_train_transforms())
        webcam_val = WebcamCropDataset(args.webcam_export, "val", eval_transforms(), kinds=("single",))
        train_loader = mixed_train_loader(
            train_loader.dataset, webcam_train, batch_size=args.batch_size, webcam_share=args.webcam_share,
            epoch_samples=args.epoch_samples, num_workers=args.num_workers,
        )
        if len(webcam_val) > 0:
            webcam_val_loader = torch.utils.data.DataLoader(webcam_val, batch_size=args.batch_size, num_workers=0)
        rooms = sorted({r["room"] for r in webcam_train.rows})
        webcam_info = {"export": str(args.webcam_export), "train_crops": len(webcam_train), "val_crops": len(webcam_val),
                       "train_rooms": rooms, "share": args.webcam_share}
        print(f"Webcam data: {len(webcam_train)} train crops from rooms {rooms}, {len(webcam_val)} validation crops (unseen rooms)")
        if len(webcam_val) == 0:
            print("WARNING: no webcam validation room; model selection uses the original validation split only")
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train images: {len(splits['train'])}")
    print(f"Val images:   {len(splits['val'])}")
    print(f"Test images:  {len(splits['test'])}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Epochs:       {args.epochs}")

    model = build_resnet18(num_classes=num_classes, pretrained=args.init_from is None).to(device)
    if args.init_from is not None:
        init = load_checkpoint(args.init_from, map_location=device)
        if init["class_names"] != class_names:
            raise SystemExit(f"--init-from class order {init['class_names']} differs from the dataset {class_names}")
        model.load_state_dict(init["model_state_dict"])
        print(f"Initialised from {args.init_from}")
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"Mixed precision: {'enabled' if use_amp else 'disabled'}")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = -1.0
    best_score = -1.0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler=scaler, use_amp=use_amp
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        webcam_val_acc = None
        if webcam_val_loader is not None:
            _, webcam_val_acc = run_epoch(model, webcam_val_loader, criterion, device, use_amp=use_amp)
            print(f"Webcam (unseen-room) Val Acc: {webcam_val_acc:.2f}%")
        # Model selection: original validation alone, or blended with unseen-room webcam validation.
        score = val_acc if webcam_val_acc is None else (1 - args.webcam_val_weight) * val_acc + args.webcam_val_weight * webcam_val_acc
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
            "training_policy": (
                "sequential per-class split; no horizontal flip"
                if args.augment == "default"
                else "sequential per-class split; robust augmentation (flip, affine, colour, blur, JPEG, noise, erasing) on train only"
            ),
            "augment": args.augment,
            "webcam": webcam_info,
            "webcam_val_accuracy": webcam_val_acc,
            "selection_score": score,
            "init_from": str(args.init_from) if args.init_from else None,
            "seed": args.seed,
        }
        save_checkpoint(build_checkpoint(model, class_names, extra), args.last_model)

        if score > best_score:
            best_score = score
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
