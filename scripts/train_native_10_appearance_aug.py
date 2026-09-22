"""Phase 6 isolated native-10 appearance-decorrelation augmentation experiment.

Tests ONE hypothesis: does moderate brightness/contrast/saturation jitter on the
TRAINING clips only reduce the signer-dependent shortcut identified in Phase 5
(signer silhouette 0.205 frozen / 0.094 layer4, both with negative class silhouette)
and improve sign-discriminative generalization?

Everything except the new color augmentation is held EXACTLY at the Phase 4B frozen
baseline (scripts/train_native_10.py): same dataset, same 149/39/129 split, same
architecture, same frozen backbone, same learning rates/dropout/batch size/workers/
optimizer/weight decay/seed/epoch budget/early-stopping patience, same temporal
sampling and jitter, same spatial resize/crop, same normalization, same class mapping.

The only intentional ML change is passing a torchvision.ColorJitter into
build_dataloader("train", ...), which src/asl_citizen/dataset.py and
src/asl_citizen/preprocessing.py apply consistently across every frame of a clip
(new, additive, default-None optional parameters — existing callers/experiments are
byte-for-byte unaffected). It is never applied to val/test (dataset.py forces
color_jitter=None whenever training=False, independent of what is passed in).
"""

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
from torchvision import transforms

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
    parameter_counts,
    resolve_backbone_mode,
)
from src.asl_citizen.train import persist_training_history, run_epoch
from src.asl_citizen.utils import choose_batch_size, describe_runtime_device, load_class_mappings, seed_pipeline
from src.utils.device import print_device_info

OUTPUTS_DIR = ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "appearance_aug"

# Moderate, conservative appearance-decorrelating jitter.
# +/-20% brightness, +/-20% contrast, +/-20% saturation. No hue shift (not requested;
# hue can distort skin tone in a way unrelated to appearance decorrelation). Values are
# deliberately mild relative to the existing train_spatial_transforms geometric crop
# (which already only removes up to ~15% of frame area, scale 0.85-1.0) so hand/sign
# visibility is preserved — only appearance statistics shift, not sign content.
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2


def build_color_jitter() -> transforms.ColorJitter:
    return transforms.ColorJitter(
        brightness=COLOR_JITTER_BRIGHTNESS,
        contrast=COLOR_JITTER_CONTRAST,
        saturation=COLOR_JITTER_SATURATION,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _config():
    base = native_10_config()
    checkpoints = OUTPUTS_DIR / "checkpoints"
    metrics = OUTPUTS_DIR / "metrics"
    return replace(
        base,
        experiment_name="asl_citizen_native_10_appearance_aug",
        outputs_dir=OUTPUTS_DIR,
        checkpoints_dir=checkpoints,
        reports_dir=OUTPUTS_DIR / "reports",
        metrics_dir=metrics,
        logs_dir=OUTPUTS_DIR / "logs",
        training_history_path=metrics / "training_history.json",
        best_checkpoint_path=checkpoints / "asl_citizen_native_10_appearance_aug_resnet18_gru_best.pth",
        last_checkpoint_path=checkpoints / "asl_citizen_native_10_appearance_aug_resnet18_gru_last.pth",
        # Everything below is an EXACT match to the Phase 4B frozen baseline
        # (scripts/train_native_10.py's main()).
        num_epochs=30,
        early_stopping_patience=12,
        batch_size=4,
        num_workers=2,
        head_learning_rate=1e-3,
        learning_rate=1e-3,
        backbone_learning_rate=1e-4,
        dropout=0.3,
        backbone_train_mode="frozen",
        freeze_backbone=True,
        seed=42,
    )


def print_transform_configuration(cfg) -> dict:
    """Print the exact transform configuration before training, per Phase 6 pre-flight
    requirement. Train/eval geometric crop and temporal sampling are UNCHANGED from
    Phase 4B; the only addition is the train-only ColorJitter block."""
    config = {
        "train_transform": {
            "temporal_sampling": "jittered_temporal_indices (uniform base +/- train_jitter_frames)",
            "train_jitter_frames": cfg.train_jitter_frames,
            "spatial_resize": [cfg.image_size + 32, cfg.image_size + 32],
            "spatial_crop": "RandomResizedCrop, SAME crop applied to every frame in a clip",
            "random_resized_crop_scale": [0.85, 1.0],
            "random_resized_crop_ratio": [0.9, 1.1],
            "color_jitter": {
                "applied": True,
                "scope": "TRAIN ONLY; SAME sampled factors applied to every frame in a clip",
                "brightness": COLOR_JITTER_BRIGHTNESS,
                "contrast": COLOR_JITTER_CONTRAST,
                "saturation": COLOR_JITTER_SATURATION,
                "hue": 0.0,
                "horizontal_flip": False,
                "rotation": False,
                "perspective": False,
                "random_erasing": False,
                "blur": False,
                "noise": False,
            },
            "normalization": {"mean": list(cfg.imagenet_mean), "std": list(cfg.imagenet_std)},
        },
        "eval_transform": {
            "temporal_sampling": "uniform_temporal_indices (deterministic, no jitter)",
            "spatial_resize": [cfg.image_size + 16, cfg.image_size + 16],
            "spatial_crop": "CenterCrop (deterministic)",
            "color_jitter": {"applied": False, "note": "color augmentation is NEVER applied to val/test"},
            "normalization": {"mean": list(cfg.imagenet_mean), "std": list(cfg.imagenet_std)},
        },
    }
    print("=" * 70)
    print("EXACT TRANSFORM CONFIGURATION")
    print("=" * 70)
    print(json.dumps(config, indent=2))
    return config


def preflight(cfg) -> tuple[dict[str, int], dict[str, str]]:
    print("=" * 70)
    print("PRE-FLIGHT: resolved configuration")
    print("=" * 70)
    resolved = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    print(json.dumps(resolved, indent=2))

    class_to_idx, idx_to_class = load_class_mappings(cfg.class_to_idx_path, cfg.idx_to_class_path)
    assert len(class_to_idx) == 10, f"Expected 10 classes, got {len(class_to_idx)}"
    assert cfg.num_classes == 10, f"cfg.num_classes={cfg.num_classes}, expected 10"
    expected_glosses = {"BOOK", "EAT1", "HELLO", "HELP", "MOTHER", "NO", "PLEASE", "THANKYOU", "WATER", "YES"}
    assert set(class_to_idx) == expected_glosses, f"class set changed: {sorted(class_to_idx)}"

    train_rows = _rows(cfg.train_manifest)
    val_rows = _rows(cfg.val_manifest)
    test_rows = _rows(cfg.test_manifest)
    for rows, split in ((train_rows, "train"), (val_rows, "val"), (test_rows, "test")):
        for row in rows:
            row["split"] = split
    assert (len(train_rows), len(val_rows), len(test_rows)) == (149, 39, 129), (
        f"split counts changed: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
    )

    leakage = verify_signer_leakage(train_rows + val_rows + test_rows)
    assert leakage["signer_independent"]
    assert not leakage["train_val_overlap"] and not leakage["train_test_overlap"] and not leakage["val_test_overlap"]

    paths_that_must_isolate = [
        cfg.outputs_dir, cfg.checkpoints_dir, cfg.reports_dir, cfg.metrics_dir, cfg.logs_dir,
    ]
    assert all("appearance_aug" in str(p) for p in paths_that_must_isolate), (
        "output paths not scoped to the appearance_aug experiment"
    )
    assert all("asl_citizen_100" not in str(p) for p in paths_that_must_isolate)
    assert not cfg.best_checkpoint_path.exists(), (
        f"best checkpoint already exists at {cfg.best_checkpoint_path}; refusing to silently overwrite Phase 6"
    )

    # Guard against colliding with any prior experiment's checkpoints.
    phase4b_best = ROOT / "outputs" / "asl_citizen_native_10" / "checkpoints" / "asl_citizen_native_10_resnet18_gru_best.pth"
    phase4c_best = (
        ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "layer4" / "checkpoints"
        / "asl_citizen_native_10_layer4_resnet18_gru_best.pth"
    )
    assert cfg.best_checkpoint_path not in (phase4b_best, phase4c_best)

    print("\nPRE-FLIGHT")
    print(json.dumps({
        "dataset_path": str(cfg.data_dir),
        "output_path": str(cfg.outputs_dir),
        "class_mapping": idx_to_class,
        "counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "signer_split": leakage,
        "architecture": "resnet18_gru",
        "backbone_mode": "frozen",
        "head_learning_rate": cfg.head_learning_rate,
        "backbone_learning_rate": cfg.backbone_learning_rate,
        "dropout": cfg.dropout,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "frames": cfg.num_frames,
        "epochs": cfg.num_epochs,
        "patience": cfg.early_stopping_patience,
        "seed": cfg.seed,
        "phase4b_best_checkpoint_untouched": phase4b_best.exists(),
        "phase4c_best_checkpoint_untouched": phase4c_best.exists(),
    }, indent=2))
    return class_to_idx, idx_to_class


def main() -> None:
    cfg = _config()
    class_to_idx, idx_to_class = preflight(cfg)
    print_transform_configuration(cfg)
    print("\nSAFE TO START PHASE 6\n")

    seed_pipeline(cfg.seed)
    device = print_device_info()
    print(json.dumps(describe_runtime_device(), indent=2))

    batch_size = choose_batch_size(cfg.batch_size, device)
    backbone_mode = resolve_backbone_mode(cfg)
    assert backbone_mode == "frozen"
    model = build_model(cfg).to(device)
    counts = parameter_counts(model)
    print(json.dumps({
        "parameter_counts": counts,
        "trainability": encoder_trainability_report(model),
    }, indent=2))

    for path in (cfg.checkpoints_dir, cfg.reports_dir, cfg.metrics_dir, cfg.logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    color_jitter = build_color_jitter()
    train_loader, _ = build_dataloader("train", cfg, batch_size=batch_size, color_jitter=color_jitter)
    val_loader, _ = build_dataloader("val", cfg, batch_size=batch_size, shuffle=False)  # color_jitter NOT passed
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model, head_lr=cfg.head_learning_rate, backbone_lr=cfg.backbone_learning_rate, weight_decay=cfg.weight_decay
    )
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
        train_loss, train_acc, train_top5 = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_loss, val_acc, val_top5 = run_epoch(model, val_loader, criterion, device, None, None, use_amp)
        diversity = prediction_diversity(model, val_loader, device, num_classes=10)
        duration = time.perf_counter() - started

        if any(math.isnan(v) or math.isinf(v) for v in (train_loss, val_loss)):
            nan_or_inf = True

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
            "val_collapse_warning": diversity["collapse_warning"],
            "learning_rate": lrs,
            "backbone_mode": backbone_mode,
            "duration_sec": round(duration, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        history.append(entry)
        persist_training_history(cfg.training_history_path, entry)
        print(
            f"epoch {epoch}/{cfg.num_epochs} train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2%} val_top5={val_top5:.2%} "
            f"unique_val={diversity['unique_predicted_classes']}/10"
            + (" COLLAPSE" if diversity["collapse_warning"] else "")
        )

        checkpoint = build_checkpoint(
            model, class_to_idx, idx_to_class, cfg,
            epoch=epoch, val_accuracy=val_acc, val_top5=val_top5, optimizer=optimizer,
            extra={"appearance_augmentation": {
                "brightness": COLOR_JITTER_BRIGHTNESS,
                "contrast": COLOR_JITTER_CONTRAST,
                "saturation": COLOR_JITTER_SATURATION,
            }},
        )
        save_checkpoint(checkpoint, cfg.last_checkpoint_path)
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            without_improvement = 0
            save_checkpoint(checkpoint, cfg.best_checkpoint_path)
            print(f"  saved best checkpoint (val_acc={val_acc:.2%})")
        else:
            without_improvement += 1
        if without_improvement >= cfg.early_stopping_patience:
            stop_reason = f"early_stopping: no val_accuracy improvement for {cfg.early_stopping_patience} consecutive epochs"
            print(f"  {stop_reason}")
            break

    result = {
        "experiment_name": cfg.experiment_name,
        "epochs_requested": cfg.num_epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "stop_reason": stop_reason,
        "nan_or_inf_detected": nan_or_inf,
        "backbone_mode": backbone_mode,
        "parameter_counts": counts,
        "color_jitter": {
            "brightness": COLOR_JITTER_BRIGHTNESS,
            "contrast": COLOR_JITTER_CONTRAST,
            "saturation": COLOR_JITTER_SATURATION,
        },
        "history": history,
        "best_checkpoint_path": str(cfg.best_checkpoint_path),
        "last_checkpoint_path": str(cfg.last_checkpoint_path),
        "batch_size": batch_size,
        "dropout": cfg.dropout,
        "head_learning_rate": cfg.head_learning_rate,
        "backbone_learning_rate": cfg.backbone_learning_rate,
        "early_stopping_patience": cfg.early_stopping_patience,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = cfg.reports_dir / "train_run_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nTraining complete. Best val_acc={best_val:.2%} at epoch {best_epoch}. Stop reason: {stop_reason}")
    print(f"Best checkpoint -> {cfg.best_checkpoint_path}")
    print(f"Run summary -> {summary_path}")


if __name__ == "__main__":
    main()
