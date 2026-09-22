from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BATCH_SIZE, BEST_MODEL_PATH, DATASET_DIR, LEGACY_MODEL_PATH, NUM_WORKERS, OUTPUTS_DIR, SPLITS_PATH
from src.data.dataset import full_sequential_splits, load_image_folder, load_split_indices, save_split
from src.data.preprocessing import eval_transforms
from src.models.resnet18 import build_resnet18
from src.utils.checkpoint import load_checkpoint
from src.utils.device import print_device_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASL classifier on the controlled held-out split")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--splits", type=Path, default=None, help="Optional existing split file; ignored if undersized")
    parser.add_argument("--save-splits", type=Path, default=OUTPUTS_DIR / "evaluation" / "splits.json")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    return parser.parse_args()


def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_conf: list[float] = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="test"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(pred.cpu().tolist())
            y_conf.extend(conf.cpu().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_conf)


def plot_confusion(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("ASL Confusion Matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_example_grid(
    items: list[tuple[str, str, str, float]],
    path: Path,
    title: str,
    limit: int = 8,
) -> None:
    if not items:
        return
    chosen = items[:limit]
    cols = min(4, len(chosen))
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (img_path, actual, predicted, conf) in zip(axes, chosen):
        image = Image.open(img_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"Actual: {actual}\nPredicted: {predicted}\nConfidence: {conf * 100:.1f}%")
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = print_device_info()
    payload = load_checkpoint(args.model, map_location=device)
    class_names = payload["class_names"]
    image_size = int(payload["image_size"])

    dataset = load_image_folder(args.data_dir, eval_transforms(image_size))
    if list(dataset.classes) != list(class_names):
        print("WARNING: dataset folder order does not match checkpoint class_names.")

    expected = full_sequential_splits(dataset.samples)
    splits = expected
    source = "computed sequential per-class split by numeric filename"
    candidate_path = args.splits if args.splits is not None else SPLITS_PATH
    if candidate_path.exists():
        loaded = load_split_indices(candidate_path)
        # Reject smoke/limited splits (e.g. 26 images) so evaluation is not silently invalid.
        if {name: len(loaded[name]) for name in loaded} == {name: len(expected[name]) for name in expected}:
            splits = loaded
            source = f"loaded matching split file: {candidate_path}"
        else:
            print(
                f"Ignoring {candidate_path}; sizes { {k: len(v) for k, v in loaded.items()} } "
                f"do not match the full sequential split { {k: len(v) for k, v in expected.items()} }."
            )
    test_idx = splits["test"]
    split_meta = {
        "data_dir": str(args.data_dir),
        "model": str(args.model),
        "policy": "sequential per-class split by numeric filename; no horizontal flip; dataset unmodified",
        "source": source,
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "class_names": class_names,
        "horizontal_flip": False,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    }
    save_split(splits, args.save_splits, extra=split_meta)
    save_split(splits, SPLITS_PATH, extra=split_meta)
    loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_resnet18(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)

    y_true, y_pred, y_conf = collect_predictions(model, loader, device)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class = (cm.diagonal() / np.maximum(cm.sum(axis=1), 1)) * 100

    print("\nTest metrics")
    print("-" * 40)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")
    print("\nPer-class accuracy")
    for name, value in zip(class_names, per_class):
        print(f"  {name}: {value:.2f}%")
    print("\n" + report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = args.output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, class_names, eval_dir / "confusion_matrix.png")
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True
    )

    legacy_note = (
        "The legacy checkpoint reports ~98.22% validation accuracy from an 80/20 random split "
        "that may leak near-duplicate frames. That figure is not this holdout result."
    )
    legacy_val = None
    if LEGACY_MODEL_PATH.exists():
        try:
            legacy_payload = load_checkpoint(LEGACY_MODEL_PATH, map_location="cpu")
            if "accuracy" in legacy_payload:
                legacy_val = float(legacy_payload["accuracy"])
        except Exception:
            legacy_val = None

    metrics = {
        "evaluation_kind": "controlled_sequential_holdout",
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "per_class_accuracy": {name: float(val) for name, val in zip(class_names, per_class)},
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": int(len(y_true)),
        "model": str(args.model),
        "split_policy": "sequential per-class split by numeric filename",
        "split_source": source,
        "split_file": str(args.save_splits),
        "horizontal_flip": False,
        "dataset_modified": False,
        "legacy_random_split_val_accuracy_reported": legacy_val,
        "legacy_random_split_note": legacy_note,
        "dynamic_j_z_supported": False,
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (eval_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    with (eval_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "support", "correct", "precision", "recall", "f1", "accuracy_percent"])
        for idx, name in enumerate(class_names):
            support = int(cm[idx].sum())
            correct = int(cm[idx, idx])
            row = report_dict[name]
            writer.writerow(
                [
                    name,
                    support,
                    correct,
                    float(row["precision"]),
                    float(row["recall"]),
                    float(row["f1-score"]),
                    float(per_class[idx]),
                ]
            )

    correct_items: list[tuple[str, str, str, float]] = []
    wrong_items: list[tuple[str, str, str, float]] = []
    for i, dataset_idx in enumerate(test_idx):
        path, _ = dataset.samples[dataset_idx]
        item = (
            path,
            class_names[int(y_true[i])],
            class_names[int(y_pred[i])],
            float(y_conf[i]),
        )
        if y_true[i] == y_pred[i]:
            correct_items.append(item)
        else:
            wrong_items.append(item)

    save_example_grid(correct_items, eval_dir / "correct_predictions.png", "Correct Predictions")
    save_example_grid(wrong_items, eval_dir / "incorrect_predictions.png", "Incorrect Predictions")
    print(f"Wrote metrics and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
