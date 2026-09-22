"""Phase 7 Task 6: train ONLY a lightweight 10-class classifier head on cached,
FROZEN I3D features (scripts/extract_i3d_features_native_10.py's output).

The I3D backbone is not imported or touched anywhere in this script — it operates
purely on the cached (N, 1024) feature arrays, so "backbone frozen" is enforced by
construction (there is no backbone object in this process to accidentally unfreeze).

Head: a single linear layer (1024 -> 10) with input feature dropout. This is the
simplest suitable head per Task 3 — I3D's extract_features() output is already a
temporally-pooled, spatially-pooled embedding (see script above), so no GRU or other
temporal module is added on top (Task 3's explicit instruction).

Training configuration (conservative, chosen for 149 training examples, not swept):
  - head_lr = 1e-3: matches this project's existing head_learning_rate convention
    for classifier heads (src/asl_citizen/config.py's HEAD_LEARNING_RATE), the value
    Task 6 pointed at ("determine a reasonable head learning rate from the existing
    implementation") rather than reusing the ResNet BACKBONE learning rate.
  - weight_decay = 1e-3: higher than this project's usual 1e-4, because a 1024-dim
    input to a linear layer with only 149 training examples (10,250 weights) is far
    more overparameterized relative to its training set than the existing GRU/head
    combination was; matched with feature dropout (below) as a second regularizer.
  - feature_dropout = 0.5: applied to the input embedding before the linear layer
    (there is no hidden layer to place standard dropout within, since the head is a
    single linear layer by design).
  - optimizer = AdamW, full-batch gradient descent (149 training examples fit in one
    batch trivially; avoids introducing minibatch-order as a new experimental
    variable for a linear probe this small).
  - seed = 42, max_epochs = 100, early_stopping_patience = 15 on val_accuracy
    (modest budget, matching Task 6's "maximum epochs should be modest").

No hyperparameter sweep: exactly one configuration is run, per Task 6.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.utils import load_class_mappings

DATA_DIR = ROOT / "data" / "asl_citizen_native_10_i3d"
FEATURES_DIR = ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "i3d_frozen" / "features"
OUT_DIR = ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "i3d_frozen"
CHECKPOINTS_DIR = OUT_DIR / "checkpoints"
METRICS_DIR = OUT_DIR / "metrics"
REPORTS_DIR = OUT_DIR / "reports"

SEED = 42
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-3
FEATURE_DROPOUT = 0.5
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
FEATURE_DIM = 1024
NUM_CLASSES = 10


@dataclass(frozen=True)
class HeadConfig:
    head_lr: float = HEAD_LR
    weight_decay: float = WEIGHT_DECAY
    feature_dropout: float = FEATURE_DROPOUT
    max_epochs: int = MAX_EPOCHS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    seed: int = SEED
    feature_dim: int = FEATURE_DIM
    num_classes: int = NUM_CLASSES
    architecture: str = "linear_probe (1024 -> dropout(0.5) -> Linear(1024,10))"
    backbone: str = "frozen I3D (InceptionI3d, ASL Citizen checkpoint) — not trainable, not loaded in this script"


class LinearHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x))


def load_split(split: str) -> dict:
    path = FEATURES_DIR / f"{split}_features.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/extract_i3d_features_native_10.py first"
        )
    data = np.load(path, allow_pickle=False)
    return {
        "features": data["features"].astype(np.float32),
        "labels": data["labels"].astype(np.int64),
        "participants": data["participants"].tolist(),
        "glosses": data["glosses"].tolist(),
        "video_files": data["video_files"].tolist(),
    }


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = min(k, logits.size(1))
    topk = logits.topk(k, dim=1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean())


def preflight() -> dict:
    print("=" * 70)
    print("PRE-FLIGHT")
    print("=" * 70)
    class_to_idx, idx_to_class = load_class_mappings(
        DATA_DIR / "class_to_idx.json", DATA_DIR / "idx_to_class.json"
    )
    assert len(class_to_idx) == 10
    expected = {"BOOK", "EAT1", "HELLO", "HELP", "MOTHER", "NO", "PLEASE", "THANKYOU", "WATER", "YES"}
    assert set(class_to_idx) == expected, f"class set changed: {sorted(class_to_idx)}"

    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")
    counts = {
        "train": len(train_data["labels"]),
        "val": len(val_data["labels"]),
        "test": len(test_data["labels"]),
    }
    print("split counts:", counts)
    assert counts == {"train": 149, "val": 39, "test": 129}, f"split counts changed: {counts}"

    train_signers = set(train_data["participants"])
    val_signers = set(val_data["participants"])
    test_signers = set(test_data["participants"])
    overlap_tv = train_signers & val_signers
    overlap_tt = train_signers & test_signers
    overlap_vt = val_signers & test_signers
    print(f"signers: train={len(train_signers)} val={len(val_signers)} test={len(test_signers)}")
    print(f"overlaps: train/val={overlap_tv} train/test={overlap_tt} val/test={overlap_vt}")
    assert not overlap_tv and not overlap_tt and not overlap_vt, "signer leakage across splits"

    for label, path in (
        ("outputs_dir", OUT_DIR), ("checkpoints_dir", CHECKPOINTS_DIR),
        ("metrics_dir", METRICS_DIR), ("reports_dir", REPORTS_DIR),
    ):
        assert "i3d_frozen" in str(path), f"{label} not scoped to i3d_frozen: {path}"
        assert "asl_citizen_100" not in str(path)
    best_ckpt = CHECKPOINTS_DIR / "head_best.pt"
    assert not best_ckpt.exists(), f"refusing to overwrite existing checkpoint at {best_ckpt}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_properties(0).name}, {vram_gb:.2f} GB VRAM")

    cfg = HeadConfig()
    print("\nhead configuration:")
    print(json.dumps(asdict(cfg), indent=2))

    print("\nSAFE TO START I3D TRANSFER EXPERIMENT\n")
    return {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class, "device": device, "cfg": cfg}


def main() -> None:
    info = preflight()
    device = info["device"]
    cfg: HeadConfig = info["cfg"]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_data = load_split("train")
    val_data = load_split("val")

    X_train = torch.from_numpy(train_data["features"]).to(device)
    y_train = torch.from_numpy(train_data["labels"]).to(device)
    X_val = torch.from_numpy(val_data["features"]).to(device)
    y_val = torch.from_numpy(val_data["labels"]).to(device)

    model = LinearHead(cfg.feature_dim, cfg.num_classes, cfg.feature_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_val_acc = float("-inf")
    best_epoch = 0
    without_improvement = 0
    stop_reason = "completed_all_epochs"

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(X_train)
            train_loss = float(criterion(train_logits, y_train))
            train_acc = float((train_logits.argmax(1) == y_train).float().mean())
            train_top5 = top_k_accuracy(train_logits, y_train, 5)

            val_logits = model(X_val)
            val_loss = float(criterion(val_logits, y_val))
            val_acc = float((val_logits.argmax(1) == y_val).float().mean())
            val_top5 = top_k_accuracy(val_logits, y_val, 5)
            val_preds = val_logits.argmax(1)
            val_unique = int(len(torch.unique(val_preds)))

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "train_top5": round(train_top5, 4),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 4),
            "val_top5": round(val_top5, 4),
            "val_unique_predicted_classes": val_unique,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        history.append(entry)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": info["class_to_idx"],
                    "idx_to_class": info["idx_to_class"],
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "val_top5_accuracy": val_top5,
                    "config": asdict(cfg),
                },
                CHECKPOINTS_DIR / "head_best.pt",
            )
        else:
            without_improvement += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            marker = "  saved best" if improved else ""
            print(
                f"epoch {epoch}/{cfg.max_epochs} train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2%} val_top5={val_top5:.2%} "
                f"unique_val={val_unique}/10{marker}"
            )

        if without_improvement >= cfg.early_stopping_patience:
            stop_reason = f"early_stopping: no val_accuracy improvement for {cfg.early_stopping_patience} consecutive epochs"
            print(f"  {stop_reason}")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": info["class_to_idx"],
            "idx_to_class": info["idx_to_class"],
            "epoch": history[-1]["epoch"],
            "val_accuracy": history[-1]["val_accuracy"],
            "val_top5_accuracy": history[-1]["val_top5"],
            "config": asdict(cfg),
        },
        CHECKPOINTS_DIR / "head_last.pt",
    )

    history_path = METRICS_DIR / "training_history.json"
    history_path.write_text(json.dumps({"epochs": history}, indent=2), encoding="utf-8")

    result = {
        "experiment_name": "asl_citizen_native_10_i3d_frozen",
        "epochs_requested": cfg.max_epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "stop_reason": stop_reason,
        "config": asdict(cfg),
        "history": history,
        "best_checkpoint_path": str(CHECKPOINTS_DIR / "head_best.pt"),
        "last_checkpoint_path": str(CHECKPOINTS_DIR / "head_last.pt"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = REPORTS_DIR / "train_run_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nTraining complete. Best val_acc={best_val_acc:.2%} at epoch {best_epoch}. Stop reason: {stop_reason}")
    print(f"Best checkpoint -> {CHECKPOINTS_DIR / 'head_best.pt'}")
    print(f"Run summary -> {summary_path}")


if __name__ == "__main__":
    main()
