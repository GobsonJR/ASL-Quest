"""Evaluate a trained WLASL100 word checkpoint. Does not touch alphabet models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.word_model.checkpoint import load_word_checkpoint
from src.word_model.config import (
    CONFUSION_MATRIX_CSV_PATH,
    CONFUSION_MATRIX_PNG_PATH,
    EVALUATION_RESULTS_PATH,
    WORD_BEST_MODEL_PATH,
    WORD_OUTPUTS_DIR,
)
from src.word_model.dataset import build_word_dataloader
from src.word_model.metrics import compute_evaluation_metrics
from src.word_model.model import build_word_model
from src.word_model.utils import choose_batch_size, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WLASL100 word model")
    parser.add_argument("--checkpoint", type=Path, default=WORD_BEST_MODEL_PATH)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    return parser.parse_args()


def save_confusion_matrix_csv(matrix: list[list[int]], class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + class_names)
        for row_idx, row in enumerate(matrix):
            writer.writerow([class_names[row_idx]] + row)


def save_confusion_matrix_png(matrix: np.ndarray, class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="WLASL100 Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path: Path, split: str) -> dict:
    device = get_device()
    payload, cfg, class_names = load_word_checkpoint(checkpoint_path, map_location=device)
    model = build_word_model(cfg, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    loader, _ = build_word_dataloader(
        split,
        cfg,
        batch_size=choose_batch_size(cfg.batch_size, device),
        shuffle=False,
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    logits_batches: list[np.ndarray] = []
    for clips, labels, _meta in loader:
        logits = model(clips.to(device))
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        logits_batches.append(logits.cpu().numpy())
    logits_all = np.concatenate(logits_batches, axis=0) if logits_batches else np.zeros((0, cfg.num_classes))

    results = compute_evaluation_metrics(
        y_true,
        y_pred,
        logits_all,
        class_names,
        split=split,
        seed=cfg.seed,
    )
    results["checkpoint"] = str(checkpoint_path)
    results["frame_count"] = cfg.frame_count
    results["image_size"] = cfg.image_size
    results["architecture"] = payload.get("architecture")
    return results


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        print(f"No word-model checkpoint at {args.checkpoint}. Train before evaluating.")
        return

    results = evaluate_checkpoint(args.checkpoint, args.split)
    WORD_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    class_names = list(results["per_class"].keys())
    matrix = np.asarray(results["confusion_matrix"], dtype=int)
    save_confusion_matrix_csv(matrix.tolist(), class_names, CONFUSION_MATRIX_CSV_PATH)
    save_confusion_matrix_png(matrix, class_names, CONFUSION_MATRIX_PNG_PATH)

    summary = {
        "split": results["split"],
        "samples": results["samples"],
        "evaluated_classes": results["evaluated_classes"],
        "total_classes": results["total_classes"],
        "missing_class_count": results["missing_class_count"],
        "top1_accuracy": results["top1_accuracy"],
        "top5_accuracy": results["top5_accuracy"],
        "macro_precision": results["macro_precision"],
        "macro_recall": results["macro_recall"],
        "macro_f1": results["macro_f1"],
        "correct_predictions": results["correct_predictions"],
        "incorrect_predictions": results["incorrect_predictions"],
        "note": results.get("note"),
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {EVALUATION_RESULTS_PATH}")
    print(f"Wrote {CONFUSION_MATRIX_CSV_PATH}")
    print(f"Wrote {CONFUSION_MATRIX_PNG_PATH}")


if __name__ == "__main__":
    main()
