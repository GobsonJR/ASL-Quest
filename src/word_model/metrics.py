"""Evaluation metrics for WLASL100 word recognition."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    if logits.size == 0:
        return 0.0
    k = min(k, logits.shape[1])
    topk = np.argsort(-logits, axis=1)[:, :k]
    hits = sum(int(labels[i] in topk[i]) for i in range(len(labels)))
    return hits / len(labels)


def compute_evaluation_metrics(
    y_true: list[int],
    y_pred: list[int],
    logits: np.ndarray,
    class_names: list[str],
    *,
    split: str,
    seed: int | None = None,
) -> dict[str, Any]:
    total_classes = len(class_names)
    present_ids = sorted(set(y_true))
    missing_ids = [class_id for class_id in range(total_classes) if class_id not in present_ids]
    evaluated_classes = len(present_ids)

    top1 = accuracy_score(y_true, y_pred) if y_true else 0.0
    top5 = top_k_accuracy(logits, np.asarray(y_true), k=5) if y_true else 0.0
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=present_ids) if y_true else 0.0
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=present_ids) if y_true else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=present_ids) if y_true else 0.0

    per_class: dict[str, dict[str, Any]] = {}
    counts_true = Counter(y_true)
    counts_pred = Counter(y_pred)
    correct_by_class = Counter()
    for truth, pred in zip(y_true, y_pred):
        if truth == pred:
            correct_by_class[truth] += 1

    for class_id, name in enumerate(class_names):
        if class_id in missing_ids:
            per_class[name] = {
                "class_id": class_id,
                "status": "not_evaluated",
                "support": 0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
            }
            continue
        support = counts_true[class_id]
        correct = correct_by_class[class_id]
        class_preds = [pred for truth, pred in zip(y_true, y_pred) if truth == class_id]
        class_precision = correct / counts_pred[class_id] if counts_pred[class_id] else 0.0
        class_recall = correct / support if support else 0.0
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if (class_precision + class_recall) > 0
            else 0.0
        )
        per_class[name] = {
            "class_id": class_id,
            "status": "evaluated",
            "support": support,
            "accuracy": correct / support if support else 0.0,
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
            "correct": correct,
            "incorrect": support - correct,
        }

    evaluated_entries = [
        (name, values)
        for name, values in per_class.items()
        if values["status"] == "evaluated"
    ]
    evaluated_entries.sort(key=lambda item: item[1]["accuracy"], reverse=True)
    strongest = [
        {"word": name, "class_id": values["class_id"], "accuracy": values["accuracy"], "support": values["support"]}
        for name, values in evaluated_entries[:5]
    ]
    weakest = [
        {"word": name, "class_id": values["class_id"], "accuracy": values["accuracy"], "support": values["support"]}
        for name, values in sorted(evaluated_entries, key=lambda item: item[1]["accuracy"])[:5]
    ]
    zero_correct = [
        {"word": name, "class_id": values["class_id"], "support": values["support"]}
        for name, values in evaluated_entries
        if values["correct"] == 0
    ]
    missing_classes = [
        {"word": class_names[class_id], "class_id": class_id, "status": "not_evaluated"}
        for class_id in missing_ids
    ]

    cm = confusion_matrix(y_true, y_pred, labels=list(range(total_classes))) if y_true else np.zeros((total_classes, total_classes), dtype=int)
    correct_count = sum(int(t == p) for t, p in zip(y_true, y_pred))
    incorrect_count = len(y_true) - correct_count

    note = None
    if split == "test" and evaluated_classes < total_classes:
        note = (
            f"Official WLASL100 test split contains videos for only {evaluated_classes}/{total_classes} classes. "
            f"{len(missing_ids)} classes have zero test videos and are marked not_evaluated."
        )

    return {
        "split": split,
        "seed": seed,
        "total_classes": total_classes,
        "evaluated_classes": evaluated_classes,
        "missing_classes": missing_classes,
        "missing_class_count": len(missing_ids),
        "samples": len(y_true),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "per_class": per_class,
        "strongest_classes": strongest,
        "weakest_classes": weakest,
        "zero_correct_classes": zero_correct,
        "confusion_matrix_labels": list(range(total_classes)),
        "confusion_matrix": cm.tolist(),
        "note": note,
    }
