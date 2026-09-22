"""Training entry point for ASL Citizen 100-class model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.checkpoint import build_checkpoint, save_checkpoint
from src.asl_citizen.config import (
    BEST_CHECKPOINT_PATH,
    LAST_CHECKPOINT_PATH,
    LOGS_DIR,
    METRICS_DIR,
    TRAINING_HISTORY_PATH,
    default_config,
)
from src.asl_citizen.dataset import build_dataloader
from src.asl_citizen.manifest import build_manifests
from src.asl_citizen.metrics import batch_accuracy, top_k_accuracy
from src.asl_citizen.model import (
    build_model,
    build_optimizer,
    encoder_trainability_report,
    parameter_counts,
    resolve_backbone_mode,
)
from src.asl_citizen.utils import (
    choose_batch_size,
    describe_runtime_device,
    load_class_mappings,
    seed_pipeline,
)
from src.utils.device import print_device_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASL Citizen 100-class ResNet18+GRU")
    parser.add_argument("--build-manifest", action="store_true", help="Generate 100-class manifests first")
    parser.add_argument("--validate-dataloader", action="store_true", help="Print dataloader diagnostics and exit")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-2 train batches with backward pass and exit")
    parser.add_argument("--confirm-train", action="store_true", help="Required for full training run")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Head (GRU/classifier) learning rate")
    parser.add_argument("--head-lr", type=float, default=None, help="Alias for --lr")
    parser.add_argument("--backbone-lr", type=float, default=None, help="Learning rate for trainable ResNet layers")
    parser.add_argument(
        "--backbone-mode",
        choices=("frozen", "layer4", "full"),
        default=None,
        help="Which ResNet18 layers receive gradients",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def persist_training_history(path: Path, entry: dict) -> None:
    """Append one epoch record immediately so interrupted runs keep prior metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {"epochs": []}
    else:
        payload = {"epochs": []}
    if not isinstance(payload, dict):
        payload = {"epochs": []}
    payload.setdefault("epochs", []).append(entry)
    payload["updated_at"] = entry.get("timestamp")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_runtime_config(args: argparse.Namespace):
    overrides = {}
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    head_lr = args.head_lr if args.head_lr is not None else args.lr
    if head_lr is not None:
        overrides["learning_rate"] = head_lr
        overrides["head_learning_rate"] = head_lr
    if args.backbone_lr is not None:
        overrides["backbone_learning_rate"] = args.backbone_lr
    if args.backbone_mode is not None:
        overrides["backbone_train_mode"] = args.backbone_mode
        overrides["freeze_backbone"] = args.backbone_mode == "frozen"
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    return default_config(**overrides) if overrides else default_config()


def print_dataloader_validation(cfg, split: str = "train") -> None:
    loader, dataset = build_dataloader(split, cfg, batch_size=cfg.batch_size, shuffle=False)
    sample = dataset[0]
    batch = next(iter(loader))
    class_to_idx, idx_to_class = load_class_mappings(cfg.class_to_idx_path, cfg.idx_to_class_path)

    print("=" * 60)
    print("Dataset: ASL Citizen 100")
    print(f"Split: {split}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Frames per clip: {cfg.num_frames}")
    print(f"Frame shape: {list(sample['video'].shape[1:])}")
    print(f"Batch shape: {list(batch['video'].shape)}")
    print(f"Class range: 0–{len(class_to_idx) - 1}")
    print(f"Class names (first 10): {[idx_to_class[str(i)] for i in range(min(10, len(idx_to_class)))]}")
    print("=" * 60)

    expected = (batch["video"].shape[1], batch["video"].shape[2], batch["video"].shape[3], batch["video"].shape[4])
    if expected != (cfg.num_frames, 3, cfg.image_size, cfg.image_size):
        raise RuntimeError(f"Unexpected batch video shape: {batch['video'].shape}")
    labels = batch["label"]
    if labels.min() < 0 or labels.max() >= len(class_to_idx):
        raise RuntimeError(f"Label out of range: min={labels.min()} max={labels.max()}")


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total = 0

    iterator = tqdm(loader, leave=False, desc="train" if training else "eval")
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        clips = batch["video"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(clips)
            loss = criterion(logits, labels)
            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())
        total += labels.size(0)
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    if total == 0:
        return 0.0, 0.0, 0.0
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    avg_loss = running_loss / total
    top1 = batch_accuracy(logits_cat, labels_cat)
    top5 = top_k_accuracy(logits_cat, labels_cat, k=5)
    return avg_loss, top1, top5


def run_dry_run(cfg, device: torch.device) -> dict:
    batch_size = choose_batch_size(cfg.batch_size, device)
    train_loader, _ = build_dataloader("train", cfg, batch_size=batch_size)
    class_to_idx, idx_to_class = load_class_mappings(cfg.class_to_idx_path, cfg.idx_to_class_path)

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        head_lr=cfg.head_learning_rate,
        backbone_lr=cfg.backbone_learning_rate,
        weight_decay=cfg.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_loss, train_acc, train_top5 = run_epoch(
        model,
        train_loader,
        criterion,
        device,
        optimizer,
        scaler,
        use_amp,
        max_batches=2,
    )
    val_loss, val_acc, val_top5 = run_epoch(
        model,
        build_dataloader("val", cfg, batch_size=batch_size, shuffle=False)[0],
        criterion,
        device,
        None,
        None,
        use_amp,
        max_batches=1,
    )

    test_ckpt_path = cfg.checkpoints_dir / "dry_run_test.pth"
    payload = build_checkpoint(
        model,
        class_to_idx,
        idx_to_class,
        cfg,
        epoch=0,
        val_accuracy=val_acc,
        val_top5=val_top5,
        optimizer=optimizer,
        extra={"dry_run": True},
    )
    save_checkpoint(payload, test_ckpt_path)

    peak_mb = None
    if device.type == "cuda":
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1)

    params = parameter_counts(model)
    sample_batch = next(iter(train_loader))
    return {
        "dry_run": True,
        "batch_shape": list(sample_batch["video"].shape),
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_top5": train_top5,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_top5": val_top5,
        "trainable_parameters": params["trainable"],
        "frozen_parameters": params["frozen"],
        "total_parameters": params["total"],
        "gpu_memory_peak_mb": peak_mb,
        "checkpoint_saved": str(test_ckpt_path),
        "device": str(device),
    }


def main() -> None:
    args = parse_args()
    cfg = build_runtime_config(args)

    if args.build_manifest:
        report = build_manifests(cfg=cfg)
        print(json.dumps(report["split_counts"], indent=2))
        print(f"Manifest report -> {cfg.outputs_dir / 'reports' / 'manifest_report.json'}")

    if not cfg.train_manifest.exists():
        print("Manifest not found; building...")
        build_manifests(cfg=cfg)

    seed_pipeline(cfg.seed)
    device = print_device_info()
    print(json.dumps(describe_runtime_device(), indent=2))

    class_to_idx, idx_to_class = load_class_mappings(cfg.class_to_idx_path, cfg.idx_to_class_path)
    print(f"Classes: {len(class_to_idx)}")
    print(f"Sample class names: {sorted(class_to_idx.keys())[:10]}")

    if args.validate_dataloader:
        print_dataloader_validation(cfg, "train")
        print_dataloader_validation(cfg, "val")
        return

    batch_size = choose_batch_size(cfg.batch_size, device)
    backbone_mode = resolve_backbone_mode(cfg)
    print(
        f"Using batch_size={batch_size} backbone_mode={backbone_mode} "
        f"head_lr={cfg.head_learning_rate} backbone_lr={cfg.backbone_learning_rate}"
    )

    model = build_model(cfg).to(device)
    params = parameter_counts(model)
    print(json.dumps(encoder_trainability_report(model), indent=2))
    print(f"Parameters: total={params['total']} trainable={params['trainable']} frozen={params['frozen']}")

    if args.dry_run:
        result = run_dry_run(cfg, device)
        out_path = LOGS_DIR / "dry_run_result.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"Dry-run log -> {out_path}")
        return

    if not args.confirm_train:
        print("Refusing full training. Use --dry-run, --validate-dataloader, or pass --confirm-train.")
        return

    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, _ = build_dataloader("train", cfg, batch_size=batch_size)
    val_loader, _ = build_dataloader("val", cfg, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        head_lr=cfg.head_learning_rate,
        backbone_lr=cfg.backbone_learning_rate,
        weight_decay=cfg.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    best_val_acc = float("-inf")

    for epoch in range(1, cfg.num_epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc, train_top5 = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, use_amp
        )
        val_loss, val_acc, val_top5 = run_epoch(
            model, val_loader, criterion, device, None, None, use_amp
        )
        duration = time.perf_counter() - started
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "train_top5": round(train_top5, 4),
            "val_loss": round(val_loss, 6),
            "val_top1": round(val_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "val_top5": round(val_top5, 4),
            "learning_rate": current_lrs[0] if len(current_lrs) == 1 else current_lrs,
            "backbone_mode": backbone_mode,
            "duration_sec": round(duration, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        persist_training_history(TRAINING_HISTORY_PATH, entry)
        print(
            f"epoch {epoch}/{cfg.num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2%} val_top5={val_top5:.2%}"
        )

        ckpt = build_checkpoint(
            model, class_to_idx, idx_to_class, cfg,
            epoch=epoch, val_accuracy=val_acc, val_top5=val_top5, optimizer=optimizer,
        )
        save_checkpoint(ckpt, LAST_CHECKPOINT_PATH)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(ckpt, BEST_CHECKPOINT_PATH)
            print(f"  saved best checkpoint (val_acc={val_acc:.2%})")

    print(f"Training complete. Best val_acc={best_val_acc:.2%}")
    print(f"Best checkpoint -> {BEST_CHECKPOINT_PATH}")
    print(f"Training history -> {TRAINING_HISTORY_PATH}")


if __name__ == "__main__":
    main()
