"""Training entry point for the WLASL100 word model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.word_model.checkpoint import build_word_checkpoint, save_word_checkpoint
from src.word_model.config import (
    TRAINING_HISTORY_PATH,
    WORD_BEST_MODEL_PATH,
    WORD_LAST_MODEL_PATH,
    WordModelConfig,
    default_config,
)
from src.word_model.dataset import build_word_dataloader
from src.word_model.model import build_word_model, trainable_parameter_count
from src.word_model.training_utils import (
    EarlyStopping,
    build_history_entry,
    resolve_training_history_path,
    seed_word_experiment,
    write_training_history,
)
from src.word_model.utils import choose_batch_size, describe_cuda, get_device, load_class_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WLASL100 word-level ResNet18+GRU (separate from alphabet)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience on validation accuracy")
    parser.add_argument("--experiment-name", type=str, default=None, help="Optional experiment suffix for history file")
    parser.add_argument("--replace-history", action="store_true", help="Overwrite default training history path")
    parser.add_argument("--dry-run", action="store_true", help="Run one train batch with backward/optimizer and exit")
    parser.add_argument("--confirm-train", action="store_true", help="Required to start a full training run")
    return parser.parse_args()


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
    for clips, labels, _meta in iterator:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
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
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        iterator.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def build_runtime_config(args: argparse.Namespace) -> WordModelConfig:
    overrides: dict = {}
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr
    if args.frame_count is not None:
        overrides["frame_count"] = args.frame_count
    if args.patience is not None:
        overrides["early_stopping_patience"] = args.patience
    return default_config(**overrides) if overrides else default_config()


def run_dry_run(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    clips, labels, _meta = next(iter(train_loader))
    clips = clips.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        logits = model(clips)
        loss = criterion(logits, labels)
    if scaler is not None and use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    trainable_with_grad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not trainable_with_grad:
        raise RuntimeError("Dry-run failed: no gradients found on trainable parameters.")

    peak_mb = None
    if device.type == "cuda":
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1)

    params = trainable_parameter_count(model)
    return {
        "input_shape": list(clips.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "trainable_parameters": params["trainable"],
        "total_parameters": params["total"],
        "cuda": describe_cuda(),
        "gpu_memory_peak_mb": peak_mb,
        "trainable_params_with_grad": len(trainable_with_grad),
        "backward_ok": True,
        "optimizer_step_ok": True,
    }


def main() -> None:
    args = parse_args()
    cfg = build_runtime_config(args)
    seed_word_experiment(cfg.seed)
    device = get_device()
    print(json.dumps(describe_cuda(), indent=2))

    batch_size = choose_batch_size(cfg.batch_size, device)
    print(f"Using batch_size={batch_size} freeze_backbone={cfg.freeze_backbone}")

    train_loader, _ = build_word_dataloader("train", cfg, batch_size=batch_size)
    val_loader, _ = build_word_dataloader("val", cfg, batch_size=batch_size, shuffle=False)
    class_names = load_class_names(cfg.classes_path)
    model = build_word_model(cfg).to(device)
    print("parameters", trainable_parameter_count(model))

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    if args.dry_run:
        result = run_dry_run(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        print(json.dumps(result, indent=2))
        return

    if not args.confirm_train:
        print("Refusing to start full training. Pass --confirm-train or use --dry-run.")
        return

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    early_stop = EarlyStopping(patience=cfg.early_stopping_patience, min_delta=cfg.early_stopping_min_delta)
    history_path = resolve_training_history_path(
        TRAINING_HISTORY_PATH,
        experiment_name=args.experiment_name,
        replace_history=args.replace_history,
    )
    epoch_history: list[dict] = []
    stopped_early = False

    for epoch in range(1, cfg.num_epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, None, None, use_amp)
        duration = time.perf_counter() - started
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_history.append(
            build_history_entry(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                learning_rate=current_lr,
                epoch_duration_seconds=duration,
            )
        )
        improved = early_stop.step(val_acc, epoch)
        print(
            f"epoch {epoch} train_acc={train_acc:.2f} val_acc={val_acc:.2f} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={current_lr:.6f}"
        )

        checkpoint = build_word_checkpoint(
            model,
            class_names,
            cfg,
            epoch=epoch,
            best_epoch=early_stop.best_epoch,
            val_accuracy=val_acc,
            best_val_accuracy=early_stop.best_val_accuracy,
            optimizer=optimizer,
        )
        save_word_checkpoint(checkpoint, WORD_LAST_MODEL_PATH)
        if improved:
            save_word_checkpoint(checkpoint, WORD_BEST_MODEL_PATH)
            print(f"saved best word checkpoint -> {WORD_BEST_MODEL_PATH} (val_acc={val_acc:.2f})")

        if early_stop.should_stop():
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch={early_stop.best_epoch} best_val_acc={early_stop.best_val_accuracy:.2f}"
            )
            break

    history_payload = {
        "seed": cfg.seed,
        "experiment_name": args.experiment_name,
        "config": {
            "frame_count": cfg.frame_count,
            "image_size": cfg.image_size,
            "batch_size": batch_size,
            "learning_rate": cfg.learning_rate,
            "freeze_backbone": cfg.freeze_backbone,
            "early_stopping_patience": cfg.early_stopping_patience,
        },
        "epochs": epoch_history,
        "best_epoch": early_stop.best_epoch,
        "best_val_accuracy": round(early_stop.best_val_accuracy, 4),
        "stopped_early": stopped_early,
        "total_epochs_completed": len(epoch_history),
    }
    write_training_history(history_path, history_payload)
    print(f"Wrote training history -> {history_path}")
    print(
        f"Training finished. best_epoch={early_stop.best_epoch} "
        f"best_val_accuracy={early_stop.best_val_accuracy:.2f}"
    )


if __name__ == "__main__":
    main()
