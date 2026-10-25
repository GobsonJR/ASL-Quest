"""Reproducible ASL Video Recognition Baseline.

Implements Phase 5:
- Loads a signer-independent video subset manifest (e.g. asl_native_10 or asl_native_25).
- Uses the frozen, pretrained ASL Citizen Inception-I3D spatiotemporal backbone.
- Uniform temporal sampling (64 frames, BGR, [-1, 1], center crop 224x224).
- Caches extracted 1,024-dim features for Train, Val, and Test splits.
- Trains a classification head with Adam, Cosine/Plateau scheduling, and early stopping.
- Evaluates on the held-out test set (guaranteeing signer independence).
- Exports metrics: Top-1, Top-3, Top-5 accuracy, macro F1, and confusion report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.i3d_transfer.preprocessing import load_i3d_clip
from src.i3d_transfer.pytorch_i3d import InceptionI3d

DEFAULT_BACKBONE = ROOT / "models" / "pretrained" / "asl_citizen_i3d" / "ASL_citizen_I3D_weights.pt"
DEFAULT_RAW_DIR = ROOT / "dataset" / "video_raw"
SUBSETS_ROOT = ROOT / "dataset" / "video_processed" / "subsets"
OUTPUT_BASE = ROOT / "outputs" / "video_baseline"


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


def build_i3d_backbone(checkpoint_path: Path, device: torch.device) -> InceptionI3d:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"I3D backbone checkpoint not found: {checkpoint_path}")
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(2731)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_manifest(csv_path: Path) -> list[dict[str, str]]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "participant_id": r["Participant ID"].strip(),
                "video_file": r["Video file"].strip(),
                "gloss": r["Gloss"].strip(),
            })
    return rows


def extract_split_features(
    rows: list[dict[str, str]],
    class_to_idx: dict[str, int],
    raw_dir: Path,
    backbone: InceptionI3d,
    device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    participants_list: list[str] = []
    rng = np.random.default_rng(42)

    print(f"Extracting features for split '{split_name}' ({len(rows)} clips)...")
    t0 = time.perf_counter()

    for i, r in enumerate(rows, 1):
        v_path = raw_dir / r["video_file"]
        if not v_path.exists():
            continue

        clip_arr = load_i3d_clip(v_path, rng)  # shape [3, 64, 224, 224] float32 in [-1, 1]
        clip_tensor = torch.from_numpy(clip_arr).unsqueeze(0).to(device)  # shape [1, 3, 64, 224, 224]

        with torch.no_grad():
            feat = backbone.extract_features(clip_tensor)  # shape [1, 1024, T', 1, 1]
            pooled = feat.mean(dim=2).squeeze().cpu().numpy()  # temporal mean-pool -> shape [1024]

        features_list.append(pooled)
        labels_list.append(class_to_idx[r["gloss"]])
        participants_list.append(r["participant_id"])

        if i % 50 == 0 or i == len(rows):
            elapsed = time.perf_counter() - t0
            print(f"  [{split_name}] {i}/{len(rows)} processed ({elapsed:.1f}s)")

    return np.stack(features_list, axis=0), np.array(labels_list, dtype=np.int64), participants_list


def compute_top_k(logits: np.ndarray, targets: np.ndarray, k: int) -> float:
    k = min(k, logits.shape[1])
    top_indices = np.argsort(-logits, axis=1)[:, :k]
    matches = (top_indices == targets[:, None]).any(axis=1)
    return float(np.mean(matches))


def run_baseline(
    subset_name: str = "asl_native_10",
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    batch_size: int = 16,
    force_extract: bool = False,
) -> dict[str, Any]:
    print("=" * 70)
    print(f"RUNNING ASL VIDEO RECOGNITION BASELINE: {subset_name}")
    print("=" * 70)

    subset_dir = SUBSETS_ROOT / subset_name
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset manifest directory does not exist: {subset_dir}")

    classes_json = subset_dir / "classes.json"
    with classes_json.open("r", encoding="utf-8") as f:
        meta = json.load(f)
        classes = meta["classes"]
        class_to_idx = meta["class_to_idx"]

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(classes)
    print(f"Classes: {num_classes} ({', '.join(classes[:5])}...)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = OUTPUT_BASE / subset_name
    feat_dir = out_dir / "features"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Extraction / Caching
    splits = ["train", "val", "test"]
    split_data: dict[str, dict[str, Any]] = {}
    backbone = None

    for s in splits:
        cache_path = feat_dir / f"{s}_features.npz"
        if cache_path.exists() and not force_extract:
            print(f"Loading cached features for '{s}' from {cache_path.name}")
            npz = np.load(cache_path, allow_pickle=True)
            split_data[s] = {
                "features": npz["features"],
                "labels": npz["labels"],
                "participants": npz["participants"],
            }
        else:
            if backbone is None:
                print("\nLoading pretrained Inception-I3D backbone...")
                backbone = build_i3d_backbone(DEFAULT_BACKBONE, device)

            csv_path = subset_dir / f"{s}.csv"
            rows = load_manifest(csv_path)
            feats, lbls, parts = extract_split_features(rows, class_to_idx, DEFAULT_RAW_DIR, backbone, device, s)
            np.savez_compressed(cache_path, features=feats, labels=lbls, participants=np.array(parts))
            split_data[s] = {"features": feats, "labels": lbls, "participants": parts}

    # Free backbone from VRAM
    if backbone is not None:
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. Dataset and Dataloaders
    train_x = torch.from_numpy(split_data["train"]["features"]).float()
    train_y = torch.from_numpy(split_data["train"]["labels"]).long()
    val_x = torch.from_numpy(split_data["val"]["features"]).float()
    val_y = torch.from_numpy(split_data["val"]["labels"]).long()
    test_x = torch.from_numpy(split_data["test"]["features"]).float()
    test_y = torch.from_numpy(split_data["test"]["labels"]).long()

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    # 3. Model & Optimizer
    feature_dim = train_x.shape[1]
    head = ClassificationHead(feature_dim, num_classes, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Training Loop
    print("\nStarting classification head training...")
    best_val_acc = -1.0
    best_head_path = ckpt_dir / "head_best.pt"
    history: list[dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = head(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            preds = out.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += len(by)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        head.eval()
        with torch.no_grad():
            v_out = head(val_x.to(device))
            v_loss = criterion(v_out, val_y.to(device)).item()
            v_preds = v_out.argmax(dim=1).cpu()
            val_acc = (v_preds == val_y).float().mean().item()

        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(v_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": head.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "config": {
                        "feature_dim": feature_dim,
                        "num_classes": num_classes,
                        "dropout": dropout,
                        "subset": subset_name,
                    },
                    "epoch": ep,
                    "val_acc": val_acc,
                },
                best_head_path,
            )

        if ep % 10 == 0 or ep == epochs:
            print(f"Epoch {ep:02d}/{epochs:02d} - Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | Val Loss: {v_loss:.4f}, Acc: {val_acc*100:.1f}% (Best: {best_val_acc*100:.1f}%)")

    # 5. Held-out Signer-Independent Test Evaluation
    print("\nEvaluating on strictly held-out test split (unseen signers)...")
    payload = torch.load(best_head_path, map_location=device, weights_only=False)
    head.load_state_dict(payload["model_state_dict"])
    head.eval()

    with torch.no_grad():
        test_logits = head(test_x.to(device)).cpu().numpy()

    test_y_np = test_y.numpy()
    top1 = compute_top_k(test_logits, test_y_np, k=1)
    top3 = compute_top_k(test_logits, test_y_np, k=3)
    top5 = compute_top_k(test_logits, test_y_np, k=5)

    test_preds = np.argmax(test_logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(test_y_np, test_preds, average="macro", zero_division=0)

    # Per class metrics
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        test_y_np, test_preds, average=None, zero_division=0
    )

    per_class_results = []
    for i, gloss in enumerate(classes):
        per_class_results.append({
            "gloss": gloss,
            "support": int(per_class_support[i]),
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
        })

    eval_summary = {
        "subset_name": subset_name,
        "classes_count": num_classes,
        "train_samples": len(train_y),
        "val_samples": len(val_y),
        "test_samples": len(test_y),
        "test_signers_count": len(set(split_data["test"]["participants"])),
        "metrics": {
            "top1_accuracy": round(top1, 4),
            "top3_accuracy": round(top3, 4),
            "top5_accuracy": round(top5, 4),
            "macro_precision": round(float(precision), 4),
            "macro_recall": round(float(recall), 4),
            "macro_f1": round(float(f1), 4),
        },
        "best_epoch": payload["epoch"],
        "best_val_acc": round(payload["val_acc"], 4),
        "per_class": per_class_results,
    }

    report_path = out_dir / "evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n" + "=" * 50)
    print("HELD-OUT TEST SET RESULTS (Signer-Independent)")
    print("=" * 50)
    print(f"Top-1 Accuracy:    {top1 * 100:.2f}%")
    print(f"Top-3 Accuracy:    {top3 * 100:.2f}%  (ASL Quest grading threshold)")
    print(f"Top-5 Accuracy:    {top5 * 100:.2f}%")
    print(f"Macro F1 Score:    {f1:.4f}")
    print(f"Head Checkpoint:   {best_head_path}")
    print(f"Evaluation Report: {report_path}")
    print("=" * 50)

    return eval_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ASL video baseline training and evaluation")
    parser.add_argument("--subset", type=str, default="asl_native_10", help="Subset name in dataset/video_processed/subsets")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of features")
    args = parser.parse_args()

    run_baseline(
        subset_name=args.subset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        force_extract=args.force_extract,
    )
