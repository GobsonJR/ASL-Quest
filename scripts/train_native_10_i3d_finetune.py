"""Phase 8: controlled partial fine-tuning of the frozen ASL Citizen I3D backbone.

Uses the I3D architecture's EXISTING n_tune_layers/pretrained freeze mechanism in
InceptionI3d.forward() (src/i3d_transfer/pytorch_i3d.py) — no new freezing mechanism
is invented. n_tune_layers=3 is the smallest practical partial fine-tune available
from that mechanism: it unfreezes exactly ONE backbone block, Mixed_5c (the deepest
Inception block), plus the always-trainable head (avg_pool/dropout/logits, which sit
outside forward()'s freeze loop entirely). Confirmed by direct inspection before
writing this script (see Phase 8 report §2 for the exact endpoint list).

Two additional, necessary pieces of glue code not present in the vendored
architecture file (writing them is unavoidable work, not "inventing a new fine-tuning
mechanism" — the freeze/tune SPLIT itself is entirely n_tune_layers' decision):

1. Discriminative learning rates: forward()'s freeze mechanism only decides which
   parameters CAN receive gradients; it says nothing about what learning rate to use
   for them. Two AdamW param groups are built here from the resulting
   requires_grad=True parameter sets.
2. Frozen BatchNorm eval-mode: PyTorch's BatchNorm layers update their running
   mean/var during any forward pass in .train() mode, REGARDLESS of requires_grad.
   Left unhandled, the 15 frozen backbone blocks' BatchNorm statistics would drift
   away from their pretrained (thousands-of-clips) values using only our 149 training
   clips' statistics every epoch — silently un-freezing part of the "frozen" backbone
   in practice, even though its weights never move. Every frozen end_point submodule
   is explicitly kept in .eval() mode (Mixed_5c and the head are in .train() mode).
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.config import ASL_CITIZEN_VIDEOS_DIR
from src.asl_citizen.manifest import verify_signer_leakage
from src.asl_citizen.utils import load_class_mappings, load_manifest_rows
from src.i3d_transfer.dataset import I3DClipDataset
from src.i3d_transfer.pytorch_i3d import InceptionI3d

DATA_DIR = ROOT / "data" / "asl_citizen_native_10_i3d"
CHECKPOINT = ROOT / "models" / "pretrained" / "asl_citizen_i3d" / "ASL_citizen_I3D_weights.pt"
FROZEN_BASELINE_CKPT = (
    ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "i3d_frozen" / "checkpoints" / "head_best.pt"
)
OUT_DIR = ROOT / "outputs" / "asl_citizen_native_10_i3d" / "finetune"
CHECKPOINTS_DIR = OUT_DIR / "checkpoints"
METRICS_DIR = OUT_DIR / "metrics"
REPORTS_DIR = OUT_DIR / "reports"
LOGS_DIR = OUT_DIR / "logs"

N_TUNE_LAYERS = 3  # unfreezes exactly Mixed_5c (see module docstring)
CLASSIFIER_LR = 1e-3   # matches Phase 7's head_lr precedent for a freshly-init classifier
BACKBONE_LR = 1e-5     # 100x lower than the classifier LR: conservative for 149 training clips
WEIGHT_DECAY = 1e-3    # matches Phase 7's head weight_decay
BATCH_SIZE = 4
NUM_WORKERS = 2
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
SEED = 42
NUM_CLASSES = 10


@dataclass(frozen=True)
class FinetuneConfig:
    n_tune_layers: int = N_TUNE_LAYERS
    classifier_lr: float = CLASSIFIER_LR
    backbone_lr: float = BACKBONE_LR
    weight_decay: float = WEIGHT_DECAY
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    max_epochs: int = MAX_EPOCHS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    seed: int = SEED
    num_classes: int = NUM_CLASSES
    trainable_backbone_block: str = "Mixed_5c (deepest Inception block only)"


def build_model_for_finetune(device: torch.device) -> tuple[InceptionI3d, list[str], list[str]]:
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(2731)  # matches the checkpoint's own logits shape (trained for 2731 glosses)
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    load_result = i3d.load_state_dict(state_dict, strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys, (
        f"checkpoint mismatch: missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}"
    )
    i3d.replace_logits(NUM_CLASSES)  # discard the 2731-way head, fresh trainable 10-way head (existing mechanism)

    freeze_eps = InceptionI3d.VALID_ENDPOINTS[:-N_TUNE_LAYERS]
    tune_eps = InceptionI3d.VALID_ENDPOINTS[-N_TUNE_LAYERS:]
    frozen_endpoint_names = [e for e in freeze_eps if e in i3d.end_points]
    trainable_endpoint_names = [e for e in tune_eps if e in i3d.end_points]

    for name in frozen_endpoint_names:
        for p in i3d.end_points[name].parameters():
            p.requires_grad_(False)
    for name in trainable_endpoint_names:
        for p in i3d.end_points[name].parameters():
            p.requires_grad_(True)
    for p in i3d.logits.parameters():
        p.requires_grad_(True)

    i3d.to(device)
    return i3d, frozen_endpoint_names, trainable_endpoint_names


def set_bn_eval_for_frozen(i3d: InceptionI3d, frozen_endpoint_names: list[str]) -> None:
    """Call i3d.train() BEFORE this. Puts every frozen end_point submodule back into
    eval() mode so its BatchNorm running stats do not drift (see module docstring)."""
    for name in frozen_endpoint_names:
        i3d.end_points[name].eval()


def forward_pooled_logits(i3d: InceptionI3d, clips: torch.Tensor) -> torch.Tensor:
    """Runs the model through the EXISTING pretrained=True/n_tune_layers freeze
    mechanism, then mean-pools the resulting (B, num_classes, T') per-frame logits
    over the temporal dimension to a single (B, num_classes) prediction — the same
    mean-pooling choice documented and used for Phase 7's frozen features, applied
    here to logits since fine-tuning must run through the real forward() path rather
    than extract_features()."""
    per_frame_logits = i3d(clips, pretrained=True, n_tune_layers=N_TUNE_LAYERS)
    return per_frame_logits.mean(dim=2)


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = min(k, logits.size(1))
    topk = logits.topk(k, dim=1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean())


def preflight() -> dict:
    print("=" * 70)
    print("PHASE 8 PRE-FLIGHT")
    print("=" * 70)

    print("\n[1] Verifying official I3D checkpoint...")
    assert CHECKPOINT.exists(), f"checkpoint not found: {CHECKPOINT}"
    probe = InceptionI3d(400, in_channels=3)
    probe.replace_logits(2731)
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    print("    checkpoint:", CHECKPOINT)

    print("\n[2] Verifying strict checkpoint loading...")
    load_result = probe.load_state_dict(state_dict, strict=True)
    print(f"    missing_keys={load_result.missing_keys} unexpected_keys={load_result.unexpected_keys}")
    assert not load_result.missing_keys and not load_result.unexpected_keys

    print("\n[3] Verifying preprocessing module constants...")
    from src.i3d_transfer.preprocessing import I3D_CROP_SIZE, I3D_MAX_FRAMES
    print(f"    I3D_MAX_FRAMES={I3D_MAX_FRAMES} I3D_CROP_SIZE={I3D_CROP_SIZE}")
    print("\n[4] Verifying 64-frame input...")
    assert I3D_MAX_FRAMES == 64
    print("    OK: 64 frames")
    print("\n[5] Verifying BGR (no cv2.cvtColor CALL in the frame-loading function)...")
    import inspect
    from src.i3d_transfer import preprocessing as i3d_pp
    frame_loader_src = inspect.getsource(i3d_pp.load_rgb_frames_from_video)
    assert "cvtColor(" not in frame_loader_src, "load_rgb_frames_from_video unexpectedly converts color space"
    print("    OK: no color-space conversion call present in load_rgb_frames_from_video, BGR preserved")
    print("\n[6] Verifying [-1, 1] normalization...")
    assert "* 2 - 1" in frame_loader_src
    print("    OK: (img/255)*2-1 normalization present")

    print("\n[7] Verifying class mapping...")
    class_to_idx, idx_to_class = load_class_mappings(
        DATA_DIR / "class_to_idx.json", DATA_DIR / "idx_to_class.json"
    )
    expected = {"BOOK", "EAT1", "HELLO", "HELP", "MOTHER", "NO", "PLEASE", "THANKYOU", "WATER", "YES"}
    assert set(class_to_idx) == expected, f"class set changed: {sorted(class_to_idx)}"
    print("    class mapping:", class_to_idx)

    print("\n[8] Verifying signer-independent split...")
    train_rows = load_manifest_rows(DATA_DIR / "train.csv")
    val_rows = load_manifest_rows(DATA_DIR / "val.csv")
    test_rows = load_manifest_rows(DATA_DIR / "test.csv")
    for rows, split in ((train_rows, "train"), (val_rows, "val"), (test_rows, "test")):
        for row in rows:
            row["split"] = split
    leakage = verify_signer_leakage(train_rows + val_rows + test_rows)
    print(f"    signer_independent={leakage['signer_independent']} "
          f"train_val_overlap={leakage['train_val_overlap']} "
          f"train_test_overlap={leakage['train_test_overlap']} "
          f"val_test_overlap={leakage['val_test_overlap']}")
    assert leakage["signer_independent"]
    assert not leakage["train_val_overlap"] and not leakage["train_test_overlap"] and not leakage["val_test_overlap"]

    print("\n[9] Verifying train/val/test counts...")
    counts = {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}
    print("    counts:", counts)
    assert counts == {"train": 149, "val": 39, "test": 129}, f"counts changed: {counts}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i3d, frozen_names, trainable_names = build_model_for_finetune(device)

    print("\n[10] Trainable layers (backbone end_points + head):")
    print("     TRAINABLE:", trainable_names, "+ logits (head)")
    print("     FROZEN:   ", frozen_names)

    trainable_params = sum(p.numel() for p in i3d.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in i3d.parameters() if not p.requires_grad)
    total_params = trainable_params + frozen_params
    print("\n[11] Trainable parameter count:", trainable_params)
    print("[12] Frozen parameter count:   ", frozen_params)
    print("     Total parameter count:    ", total_params)
    print(f"     Trainable fraction: {trainable_params / total_params:.4%}")

    cfg = FinetuneConfig()
    print("\n[13] Learning rates:")
    print(f"     classifier (logits) LR = {cfg.classifier_lr}")
    print(f"     Mixed_5c (backbone)  LR = {cfg.backbone_lr}  ({cfg.classifier_lr / cfg.backbone_lr:.0f}x lower than classifier)")

    print("\n[14] Verifying output directory isolation...")
    for label, path in (
        ("outputs_dir", OUT_DIR), ("checkpoints_dir", CHECKPOINTS_DIR),
        ("metrics_dir", METRICS_DIR), ("reports_dir", REPORTS_DIR),
    ):
        print(f"     {label} = {path}")
        assert "asl_citizen_native_10_i3d" in str(path) and "finetune" in str(path), f"{label} not isolated: {path}"
    best_ckpt = CHECKPOINTS_DIR / "finetune_best.pt"
    assert not best_ckpt.exists(), f"refusing to overwrite existing checkpoint at {best_ckpt}"

    print("\n[15] Verifying the frozen-I3D baseline is untouched...")
    assert FROZEN_BASELINE_CKPT.exists(), f"frozen baseline checkpoint missing: {FROZEN_BASELINE_CKPT}"
    baseline_mtime_before = FROZEN_BASELINE_CKPT.stat().st_mtime
    print(f"     frozen baseline checkpoint present, mtime={baseline_mtime_before} (will be re-checked after training)")

    print(f"\n     device: {device}")
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"     GPU: {torch.cuda.get_device_properties(0).name}, {vram_gb:.2f} GB VRAM")

    print("\nSAFE TO START PHASE 8\n")
    return {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "device": device,
        "cfg": cfg,
        "i3d": i3d,
        "frozen_endpoint_names": frozen_names,
        "trainable_endpoint_names": trainable_names,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "baseline_mtime_before": baseline_mtime_before,
    }


@torch.no_grad()
def run_eval_epoch(i3d: InceptionI3d, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    i3d.eval()
    total_loss, total_correct, total_top5, total_n = 0.0, 0, 0.0, 0
    unique_preds: set[int] = set()
    for batch in loader:
        clips = batch["video"].to(device)
        labels = torch.tensor(batch["label"], device=device) if not torch.is_tensor(batch["label"]) else batch["label"].to(device)
        logits = forward_pooled_logits(i3d, clips)
        loss = criterion(logits, labels)
        total_loss += float(loss) * labels.size(0)
        preds = logits.argmax(dim=1)
        unique_preds.update(preds.cpu().tolist())
        total_correct += int((preds == labels).sum())
        total_top5 += top_k_accuracy(logits, labels, 5) * labels.size(0)
        total_n += labels.size(0)
    return {
        "loss": total_loss / total_n,
        "accuracy": total_correct / total_n,
        "top5": total_top5 / total_n,
        "unique_predicted_classes": len(unique_preds),
    }


def main() -> None:
    info = preflight()
    device = info["device"]
    cfg: FinetuneConfig = info["cfg"]
    i3d: InceptionI3d = info["i3d"]
    frozen_names = info["frozen_endpoint_names"]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_ds = I3DClipDataset(DATA_DIR / "train.csv", ASL_CITIZEN_VIDEOS_DIR, info["class_to_idx"], seed=cfg.seed)
    val_ds = I3DClipDataset(DATA_DIR / "val.csv", ASL_CITIZEN_VIDEOS_DIR, info["class_to_idx"], seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    classifier_params = list(i3d.logits.parameters())
    backbone_params = [p for name in InceptionI3d.VALID_ENDPOINTS if name in i3d.end_points and name not in frozen_names
                        for p in i3d.end_points[name].parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": classifier_params, "lr": cfg.classifier_lr, "weight_decay": cfg.weight_decay},
            {"params": backbone_params, "lr": cfg.backbone_lr, "weight_decay": cfg.weight_decay},
        ]
    )
    criterion = nn.CrossEntropyLoss()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_val_acc = float("-inf")
    best_epoch = 0
    without_improvement = 0
    nan_or_inf = False
    stop_reason = "completed_all_epochs"

    for epoch in range(1, cfg.max_epochs + 1):
        started = time.perf_counter()
        i3d.train()
        set_bn_eval_for_frozen(i3d, frozen_names)  # re-assert every epoch (train() above resets everything)

        train_loss_sum, train_correct, train_n = 0.0, 0, 0
        for batch in train_loader:
            clips = batch["video"].to(device)
            labels = batch["label"].to(device) if torch.is_tensor(batch["label"]) else torch.tensor(batch["label"], device=device)

            optimizer.zero_grad()
            logits = forward_pooled_logits(i3d, clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss) * labels.size(0)
            train_correct += int((logits.argmax(dim=1) == labels).sum())
            train_n += labels.size(0)

        train_loss = train_loss_sum / train_n
        train_acc = train_correct / train_n

        val_metrics = run_eval_epoch(i3d, val_loader, criterion, device)
        duration = time.perf_counter() - started

        if any(math.isnan(v) or math.isinf(v) for v in (train_loss, val_metrics["loss"])):
            nan_or_inf = True

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_top5": round(val_metrics["top5"], 4),
            "val_unique_predicted_classes": val_metrics["unique_predicted_classes"],
            "duration_sec": round(duration, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        history.append(entry)
        print(
            f"epoch {epoch}/{cfg.max_epochs} train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.2%} "
            f"val_top5={val_metrics['top5']:.2%} unique_val={val_metrics['unique_predicted_classes']}/10 "
            f"({duration:.1f}s)"
        )

        if nan_or_inf:
            stop_reason = "stopped: NaN/Inf detected in loss"
            print(f"  {stop_reason}")
            break

        checkpoint_payload = {
            "model_state_dict": i3d.state_dict(),
            "class_to_idx": info["class_to_idx"],
            "idx_to_class": info["idx_to_class"],
            "epoch": epoch,
            "val_accuracy": val_metrics["accuracy"],
            "val_top5_accuracy": val_metrics["top5"],
            "config": asdict(cfg),
            "frozen_endpoint_names": frozen_names,
            "trainable_endpoint_names": info["trainable_endpoint_names"],
        }
        torch.save(checkpoint_payload, CHECKPOINTS_DIR / "finetune_last.pt")

        improved = val_metrics["accuracy"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            without_improvement = 0
            torch.save(checkpoint_payload, CHECKPOINTS_DIR / "finetune_best.pt")
            print(f"  saved best checkpoint (val_acc={val_metrics['accuracy']:.2%})")
        else:
            without_improvement += 1

        if without_improvement >= cfg.early_stopping_patience:
            stop_reason = f"early_stopping: no val_accuracy improvement for {cfg.early_stopping_patience} consecutive epochs"
            print(f"  {stop_reason}")
            break

    history_path = METRICS_DIR / "training_history.json"
    history_path.write_text(json.dumps({"epochs": history}, indent=2), encoding="utf-8")

    baseline_mtime_after = FROZEN_BASELINE_CKPT.stat().st_mtime
    baseline_untouched = baseline_mtime_after == info["baseline_mtime_before"]

    result = {
        "experiment_name": "asl_citizen_native_10_i3d_finetune",
        "epochs_requested": cfg.max_epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "stop_reason": stop_reason,
        "nan_or_inf_detected": nan_or_inf,
        "config": asdict(cfg),
        "trainable_params": info["trainable_params"],
        "frozen_params": info["frozen_params"],
        "frozen_endpoint_names": frozen_names,
        "trainable_endpoint_names": info["trainable_endpoint_names"],
        "history": history,
        "best_checkpoint_path": str(CHECKPOINTS_DIR / "finetune_best.pt"),
        "last_checkpoint_path": str(CHECKPOINTS_DIR / "finetune_last.pt"),
        "frozen_baseline_checkpoint_untouched": baseline_untouched,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = REPORTS_DIR / "train_run_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nTraining complete. Best val_acc={best_val_acc:.2%} at epoch {best_epoch}. Stop reason: {stop_reason}")
    print(f"Frozen baseline checkpoint untouched: {baseline_untouched}")
    print(f"Best checkpoint -> {CHECKPOINTS_DIR / 'finetune_best.pt'}")
    print(f"Run summary -> {summary_path}")


if __name__ == "__main__":
    main()
