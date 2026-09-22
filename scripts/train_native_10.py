"""Phase 4: first real training run for the asl_citizen_native_10 experiment.

Reuses the existing, generic asl_citizen infrastructure (dataset.py, model.py,
checkpoint.py, metrics.py, and train.py's run_epoch/persist_training_history) without
modifying any of those files.

IMPORTANT: src/asl_citizen/train.py's main() is NOT reused directly. It imports
BEST_CHECKPOINT_PATH / LAST_CHECKPOINT_PATH / LOGS_DIR / METRICS_DIR /
TRAINING_HISTORY_PATH as hardcoded module-level constants scoped to asl_citizen_100
(not derived from the passed-in cfg), and build_runtime_config() has no way to point it
at a different data_dir. Calling it as-is for this experiment would silently write into
outputs/asl_citizen_100/checkpoints/. This script instead drives the same run_epoch
loop directly against native_10_config(), so every path used (checkpoints, logs,
metrics, history) is native_10-only by construction. This is flagged explicitly per the
Phase 4 instruction to report any deviation from just calling the existing entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
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
from src.asl_citizen.checkpoint import build_checkpoint, save_checkpoint
from src.asl_citizen.config import BEST_CHECKPOINT_PATH as ASL100_BEST_CHECKPOINT_PATH
from src.asl_citizen.config import LAST_CHECKPOINT_PATH as ASL100_LAST_CHECKPOINT_PATH
from src.asl_citizen.dataset import build_dataloader
from src.asl_citizen.manifest import verify_signer_leakage
from src.asl_citizen.model import build_model, build_optimizer, encoder_trainability_report, parameter_counts, resolve_backbone_mode
from src.asl_citizen.train import persist_training_history, run_epoch
from src.asl_citizen.utils import choose_batch_size, describe_runtime_device, load_class_mappings, seed_pipeline
from src.utils.device import print_device_info

EARLY_STOPPING_METRIC = "val_accuracy"


@torch.no_grad()
def prediction_diversity(model: nn.Module, loader, device: torch.device, num_classes: int) -> dict:
    """Extra no-grad pass to report per-epoch prediction diversity. Does not affect
    training/eval metrics, which still come from the unmodified run_epoch() above."""
    model.eval()
    from collections import Counter

    counts: Counter = Counter()
    for batch in loader:
        clips = batch["video"].to(device, non_blocking=True)
        logits = model(clips)
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        counts.update(preds)
    return {
        "unique_predicted_classes": len(counts),
        "distribution": {str(k): v for k, v in sorted(counts.items())},
        "collapse_warning": len(counts) <= 1 and num_classes > 1,
    }


def _load_manifest_participants(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def archive_prior_run(cfg, run_label: str) -> dict:
    """Move any prior run's checkpoints/history/reports aside before this run writes new
    ones, so nothing is silently overwritten. Returns a manifest of what was archived."""
    archive_dir = cfg.outputs_dir / "archive" / run_label
    candidates = [
        cfg.best_checkpoint_path,
        cfg.last_checkpoint_path,
        cfg.training_history_path,
        cfg.reports_dir / "train_run_summary.json",
        cfg.reports_dir / "test_evaluation_report.json",
    ]
    moved: list[dict[str, str]] = []
    for path in candidates:
        if path.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
            destination = archive_dir / path.name
            shutil.move(str(path), str(destination))
            moved.append({"from": str(path), "to": str(destination)})
    return {"archive_dir": str(archive_dir), "moved": moved}


def preflight(cfg) -> dict:
    print("=" * 70)
    print("PRE-FLIGHT: resolved configuration")
    print("=" * 70)
    resolved = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    print(json.dumps(resolved, indent=2))

    # 2. class mapping
    class_to_idx, idx_to_class = load_class_mappings(cfg.class_to_idx_path, cfg.idx_to_class_path)
    assert len(class_to_idx) == 10, f"Expected 10 classes, got {len(class_to_idx)}"
    assert cfg.num_classes == 10, f"cfg.num_classes={cfg.num_classes}, expected 10"
    print("\nPRE-FLIGHT: class mapping (10 classes)")
    for idx in range(10):
        print(f"  {idx}: {idx_to_class[str(idx)]}")

    # 3. split counts
    train_rows = _load_manifest_participants(cfg.train_manifest)
    val_rows = _load_manifest_participants(cfg.val_manifest)
    test_rows = _load_manifest_participants(cfg.test_manifest)
    print("\nPRE-FLIGHT: split counts")
    print(f"  train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    assert len(train_rows) == 149, f"train count changed: {len(train_rows)}"
    assert len(val_rows) == 39, f"val count changed: {len(val_rows)}"
    assert len(test_rows) == 129, f"test count changed: {len(test_rows)}"

    # 4. signer overlap (re-verify at train time, not just at manifest build time)
    for row in train_rows:
        row["split"] = "train"
    for row in val_rows:
        row["split"] = "val"
    for row in test_rows:
        row["split"] = "test"
    leakage = verify_signer_leakage(train_rows + val_rows + test_rows)
    print("\nPRE-FLIGHT: signer overlap check")
    print(f"  signer_independent={leakage['signer_independent']} "
          f"train_val_overlap={leakage['train_val_overlap']} "
          f"train_test_overlap={leakage['train_test_overlap']} "
          f"val_test_overlap={leakage['val_test_overlap']}")
    assert leakage["signer_independent"]
    assert not leakage["train_val_overlap"] and not leakage["train_test_overlap"] and not leakage["val_test_overlap"]

    # 5. output directory is native_10-only
    print("\nPRE-FLIGHT: output isolation check")
    for label, path in (
        ("outputs_dir", cfg.outputs_dir),
        ("checkpoints_dir", cfg.checkpoints_dir),
        ("data_dir", cfg.data_dir),
    ):
        path_str = str(path)
        print(f"  {label} = {path_str}")
        assert "asl_citizen_native_10" in path_str, f"{label} not scoped to native_10: {path_str}"
        assert "asl_citizen_100" not in path_str, f"{label} unexpectedly touches asl_citizen_100: {path_str}"

    # 6. checkpoint will not collide with the 100-class experiment
    print("\nPRE-FLIGHT: checkpoint collision check")
    print(f"  native_10 best -> {cfg.best_checkpoint_path}")
    print(f"  native_10 last -> {cfg.last_checkpoint_path}")
    print(f"  asl_citizen_100 best (must differ) -> {ASL100_BEST_CHECKPOINT_PATH}")
    print(f"  asl_citizen_100 last (must differ) -> {ASL100_LAST_CHECKPOINT_PATH}")
    assert cfg.best_checkpoint_path != ASL100_BEST_CHECKPOINT_PATH
    assert cfg.last_checkpoint_path != ASL100_LAST_CHECKPOINT_PATH
    assert not cfg.best_checkpoint_path.exists(), (
        f"best checkpoint still present at {cfg.best_checkpoint_path} after archiving; aborting"
    )

    print("\nPRE-FLIGHT: all checks passed.\n")
    return {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class, "leakage": leakage}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="native_10 real training run")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--run-label", type=str, default="prior_run",
                         help="Archive folder name (under outputs_dir/archive/) for any pre-existing "
                              "checkpoint/history/report this run would otherwise overwrite.")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    cfg = native_10_config()
    cfg = replace(
        cfg,
        num_epochs=args.epochs,
        batch_size=4,
        num_workers=2,
        head_learning_rate=1e-3,
        learning_rate=1e-3,
        backbone_learning_rate=1e-4,
        dropout=0.3,
        backbone_train_mode="frozen",
        freeze_backbone=True,
        early_stopping_patience=args.early_stopping_patience,
    )

    archive_report = archive_prior_run(cfg, args.run_label)
    print("=" * 70)
    print("PRE-FLIGHT: prior-run archival (explicit replacement check)")
    print("=" * 70)
    if archive_report["moved"]:
        for item in archive_report["moved"]:
            print(f"  archived: {item['from']} -> {item['to']}")
    else:
        print("  no prior checkpoint/history/report found to archive")
    print()

    preflight_info = preflight(cfg)
    class_to_idx, idx_to_class = preflight_info["class_to_idx"], preflight_info["idx_to_class"]

    seed_pipeline(cfg.seed)
    device = print_device_info()
    print(json.dumps(describe_runtime_device(), indent=2))

    batch_size = choose_batch_size(cfg.batch_size, device)
    backbone_mode = resolve_backbone_mode(cfg)
    print(f"Using batch_size={batch_size} backbone_mode={backbone_mode} "
          f"head_lr={cfg.head_learning_rate} backbone_lr={cfg.backbone_learning_rate} dropout={cfg.dropout}")

    model = build_model(cfg).to(device)
    params = parameter_counts(model)
    print(json.dumps(encoder_trainability_report(model), indent=2))
    print(f"Parameters: total={params['total']} trainable={params['trainable']} frozen={params['frozen']}")

    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)

    train_loader, _ = build_dataloader("train", cfg, batch_size=batch_size)
    val_loader, _ = build_dataloader("val", cfg, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model, head_lr=cfg.head_learning_rate, backbone_lr=cfg.backbone_learning_rate, weight_decay=cfg.weight_decay
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    best_val_acc = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []
    nan_or_inf_detected = False
    stop_reason = "completed_all_epochs"

    for epoch in range(1, cfg.num_epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc, train_top5 = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_loss, val_acc, val_top5 = run_epoch(model, val_loader, criterion, device, None, None, use_amp)
        val_diversity = prediction_diversity(model, val_loader, device, num_classes=10)
        duration = time.perf_counter() - started

        if any(math.isnan(v) or math.isinf(v) for v in (train_loss, val_loss)):
            nan_or_inf_detected = True

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
            "val_unique_predicted_classes": val_diversity["unique_predicted_classes"],
            "val_prediction_distribution": val_diversity["distribution"],
            "val_collapse_warning": val_diversity["collapse_warning"],
            "learning_rate": current_lrs[0] if len(current_lrs) == 1 else current_lrs,
            "backbone_mode": backbone_mode,
            "duration_sec": round(duration, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        history.append(entry)
        persist_training_history(cfg.training_history_path, entry)
        print(
            f"epoch {epoch}/{cfg.num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2%} val_top5={val_top5:.2%} "
            f"val_unique_preds={val_diversity['unique_predicted_classes']}/10"
            + (" COLLAPSE" if val_diversity["collapse_warning"] else "")
        )

        ckpt = build_checkpoint(
            model, class_to_idx, idx_to_class, cfg,
            epoch=epoch, val_accuracy=val_acc, val_top5=val_top5, optimizer=optimizer,
        )
        save_checkpoint(ckpt, cfg.last_checkpoint_path)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(ckpt, cfg.best_checkpoint_path)
            print(f"  saved best checkpoint (val_acc={val_acc:.2%})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.early_stopping_patience:
            stop_reason = (
                f"early_stopping: no {EARLY_STOPPING_METRIC} improvement for "
                f"{cfg.early_stopping_patience} consecutive epochs"
            )
            print(f"  {stop_reason}")
            break

    result = {
        "experiment_name": cfg.experiment_name,
        "epochs_requested": cfg.num_epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "stop_reason": stop_reason,
        "nan_or_inf_detected": nan_or_inf_detected,
        "history": history,
        "best_checkpoint_path": str(cfg.best_checkpoint_path),
        "last_checkpoint_path": str(cfg.last_checkpoint_path),
        "backbone_mode": backbone_mode,
        "batch_size": batch_size,
        "dropout": cfg.dropout,
        "head_learning_rate": cfg.head_learning_rate,
        "backbone_learning_rate": cfg.backbone_learning_rate,
        "early_stopping_patience": cfg.early_stopping_patience,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = cfg.reports_dir / "train_run_summary.json"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nTraining complete. Best val_acc={best_val_acc:.2%} at epoch {best_epoch}. Stop reason: {stop_reason}")
    print(f"Best checkpoint -> {cfg.best_checkpoint_path}")
    print(f"Run summary -> {summary_path}")
    return result


if __name__ == "__main__":
    main()
