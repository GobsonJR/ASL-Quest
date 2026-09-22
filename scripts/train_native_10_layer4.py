"""Phase 4C isolated native-10 layer4-only fine-tuning experiment."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_asl_citizen_native_10_manifest import native_10_config
from scripts.train_native_10 import prediction_diversity
from src.asl_citizen.checkpoint import build_checkpoint, save_checkpoint
from src.asl_citizen.dataset import build_dataloader
from src.asl_citizen.manifest import verify_signer_leakage
from src.asl_citizen.model import (
    build_model,
    build_optimizer,
    encoder_trainability_report,
    gradient_flow_report,
    parameter_counts,
    resolve_backbone_mode,
)
from src.asl_citizen.train import persist_training_history, run_epoch
from src.asl_citizen.utils import choose_batch_size, describe_runtime_device, load_class_mappings, seed_pipeline
from src.utils.device import print_device_info

OUTPUTS_DIR = ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "layer4"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _config():
    base = native_10_config()
    checkpoints = OUTPUTS_DIR / "checkpoints"
    metrics = OUTPUTS_DIR / "metrics"
    return replace(
        base,
        experiment_name="asl_citizen_native_10_layer4",
        outputs_dir=OUTPUTS_DIR,
        checkpoints_dir=checkpoints,
        reports_dir=OUTPUTS_DIR / "reports",
        metrics_dir=metrics,
        logs_dir=OUTPUTS_DIR / "logs",
        training_history_path=metrics / "training_history.json",
        best_checkpoint_path=checkpoints / "asl_citizen_native_10_layer4_resnet18_gru_best.pth",
        last_checkpoint_path=checkpoints / "asl_citizen_native_10_layer4_resnet18_gru_last.pth",
        num_epochs=30,
        early_stopping_patience=12,
        batch_size=4,
        num_workers=2,
        head_learning_rate=1e-3,
        learning_rate=1e-3,
        backbone_learning_rate=1e-4,
        dropout=0.3,
        backbone_train_mode="layer4",
        freeze_backbone=False,
    )


def preflight(cfg) -> tuple[dict[str, int], dict[str, str]]:
    class_to_idx, idx_to_class = load_class_mappings(cfg.class_to_idx_path, cfg.idx_to_class_path)
    train_rows = _rows(cfg.train_manifest)
    val_rows = _rows(cfg.val_manifest)
    test_rows = _rows(cfg.test_manifest)
    for rows, split in ((train_rows, "train"), (val_rows, "val"), (test_rows, "test")):
        for row in rows:
            row["split"] = split
    leakage = verify_signer_leakage(train_rows + val_rows + test_rows)
    paths = [cfg.outputs_dir, cfg.checkpoints_dir, cfg.reports_dir, cfg.metrics_dir, cfg.logs_dir]
    assert all("asl_citizen_native_10" in str(path) for path in paths)
    assert all("asl_citizen_100" not in str(path) for path in paths)
    assert (len(train_rows), len(val_rows), len(test_rows)) == (149, 39, 129)
    assert leakage["signer_independent"]
    print("PRE-FLIGHT")
    print(json.dumps({
        "dataset_path": str(cfg.data_dir),
        "output_path": str(cfg.outputs_dir),
        "class_mapping": idx_to_class,
        "counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "signer_split": leakage,
        "architecture": "resnet18_gru",
        "backbone_mode": "layer4",
        "head_learning_rate": cfg.head_learning_rate,
        "backbone_learning_rate": cfg.backbone_learning_rate,
        "batch_size": cfg.batch_size,
        "frames": cfg.num_frames,
        "epochs": cfg.num_epochs,
        "patience": cfg.early_stopping_patience,
        "seed": cfg.seed,
    }, indent=2))
    return class_to_idx, idx_to_class


def main() -> None:
    cfg = _config()
    class_to_idx, idx_to_class = preflight(cfg)
    seed_pipeline(cfg.seed)
    device = print_device_info()
    model = build_model(cfg).to(device)
    counts = parameter_counts(model)
    print(json.dumps({
        "runtime": describe_runtime_device(),
        "parameter_counts": counts,
        "trainability": encoder_trainability_report(model),
        "SAFE TO START NATIVE-10 LAYER4 EXPERIMENT": True,
    }, indent=2))
    assert resolve_backbone_mode(cfg) == "layer4"
    assert any(p.requires_grad for p in model.encoder.layer4.parameters())

    for path in (cfg.checkpoints_dir, cfg.reports_dir, cfg.metrics_dir, cfg.logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    train_loader, _ = build_dataloader("train", cfg, batch_size=cfg.batch_size)
    val_loader, _ = build_dataloader("val", cfg, batch_size=cfg.batch_size, shuffle=False)
    optimizer = build_optimizer(
        model,
        head_lr=cfg.head_learning_rate,
        backbone_lr=cfg.backbone_learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    history: list[dict] = []
    best_val = float("-inf")
    best_epoch = 0
    without_improvement = 0
    nan_or_inf = False
    stop_reason = "completed_all_epochs"

    for epoch in range(1, cfg.num_epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc, train_top5 = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, use_amp
        )
        val_loss, val_acc, val_top5 = run_epoch(
            model, val_loader, criterion, device, None, None, use_amp
        )
        diversity = prediction_diversity(model, val_loader, device, num_classes=10)
        gradients = gradient_flow_report(model)
        layer4_gradient_ok = (
            gradients["layer4"]["has_grad"]
            and gradients["layer4"]["finite"]
            and gradients["layer4"]["mean_abs"] > 0.0
        )
        nan_or_inf = nan_or_inf or any(
            math.isnan(value) or math.isinf(value)
            for value in (train_loss, val_loss, gradients["layer4"]["mean_abs"])
        )
        lrs = [group["lr"] for group in optimizer.param_groups]
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "train_top5": round(train_top5, 4),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 4),
            "val_top5": round(val_top5, 4),
            "val_unique_predicted_classes": diversity["unique_predicted_classes"],
            "val_prediction_distribution": diversity["distribution"],
            "learning_rate": lrs,
            "layer4_gradient": gradients["layer4"],
            "layer4_gradient_finite_nonzero": layer4_gradient_ok,
            "duration_sec": round(time.perf_counter() - started, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        history.append(entry)
        persist_training_history(cfg.training_history_path, entry)
        print(
            f"epoch {epoch}/{cfg.num_epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2%} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.2%} val_top5={val_top5:.2%} "
            f"unique_val={diversity['unique_predicted_classes']}/10 "
            f"layer4_grad_ok={layer4_gradient_ok}"
        )
        checkpoint = build_checkpoint(
            model, class_to_idx, idx_to_class, cfg,
            epoch=epoch, val_accuracy=val_acc, val_top5=val_top5,
            optimizer=optimizer,
            extra={"layer4_gradient_report": gradients["layer4"]},
        )
        save_checkpoint(checkpoint, cfg.last_checkpoint_path)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            without_improvement = 0
            save_checkpoint(checkpoint, cfg.best_checkpoint_path)
        else:
            without_improvement += 1
        if without_improvement >= cfg.early_stopping_patience:
            stop_reason = f"early_stopping: no val_accuracy improvement for {cfg.early_stopping_patience} consecutive epochs"
            print(stop_reason)
            break

    summary = {
        "experiment_name": cfg.experiment_name,
        "epochs_requested": cfg.num_epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "stop_reason": stop_reason,
        "nan_or_inf_detected": nan_or_inf,
        "backbone_mode": "layer4",
        "parameter_counts": counts,
        "history": history,
        "best_checkpoint_path": str(cfg.best_checkpoint_path),
        "last_checkpoint_path": str(cfg.last_checkpoint_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = cfg.reports_dir / "train_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training complete. Best val_acc={best_val:.2%} at epoch {best_epoch}")
    print(f"Run summary -> {summary_path}")


if __name__ == "__main__":
    main()
