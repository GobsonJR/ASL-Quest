"""Phase 7 Task 7: evaluate the best I3D-frozen classifier head on the official
native-10 test split's cached features. Mirrors the metric set reported by
scripts/evaluate_native_10.py (Phase 4B/4C/6) so results are directly comparable."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_native_10_i3d_frozen import LinearHead, load_split

CHECKPOINT = (
    ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "i3d_frozen" / "checkpoints" / "head_best.pt"
)
REPORT_PATH = (
    ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "i3d_frozen" / "reports" / "test_evaluation_report.json"
)


def top_k_accuracy_np(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, logits.shape[1])
    topk = np.argsort(-logits, axis=1)[:, :k]
    correct = (topk == labels[:, None]).any(axis=1)
    return float(correct.mean())


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"head checkpoint not found: {CHECKPOINT}")

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    class_to_idx = {str(k): int(v) for k, v in payload["class_to_idx"].items()}
    idx_to_class = {str(k): str(v) for k, v in payload["idx_to_class"].items()}
    cfg = payload["config"]

    model = LinearHead(cfg["feature_dim"], cfg["num_classes"], cfg["feature_dropout"])
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    test_data = load_split("test")
    X_test = torch.from_numpy(test_data["features"])
    y_test = torch.from_numpy(test_data["labels"])
    participants = test_data["participants"]

    with torch.no_grad():
        logits = model(X_test)
    logits_np = logits.numpy()
    y_true = y_test.numpy()
    y_pred = logits_np.argmax(axis=1)

    finite = bool(np.isfinite(logits_np).all())
    top1 = float((y_pred == y_true).mean())
    top5 = top_k_accuracy_np(logits_np, y_true, 5)

    idx_order = list(range(len(idx_to_class)))
    gloss_order = [idx_to_class[str(i)] for i in idx_order]

    cm = confusion_matrix(y_true, y_pred, labels=idx_order)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=idx_order, zero_division=0
    )
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    pred_counts = {gloss_order[i]: int((y_pred == i).sum()) for i in idx_order}
    unique_predicted = int(len(set(y_pred.tolist())))
    max_class_share = max(pred_counts.values()) / len(y_pred) if len(y_pred) else 0.0
    dominant_class = max(pred_counts, key=pred_counts.get)

    per_class = [
        {
            "gloss": gloss_order[i],
            "support": int(support[i]),
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "predicted_count": pred_counts[gloss_order[i]],
        }
        for i in idx_order
    ]

    by_signer: dict[str, dict[str, int]] = {}
    for participant, true_idx, pred_idx in zip(participants, y_true.tolist(), y_pred.tolist()):
        entry = by_signer.setdefault(participant, {"correct": 0, "total": 0})
        entry["total"] += 1
        entry["correct"] += int(true_idx == pred_idx)
    signer_report = {
        p: {"correct": v["correct"], "total": v["total"], "accuracy": round(v["correct"] / v["total"], 4)}
        for p, v in sorted(by_signer.items())
    }
    signer_accuracies = [v["accuracy"] for v in signer_report.values()]

    results = {
        "experiment_name": "asl_citizen_native_10_i3d_frozen",
        "checkpoint": str(CHECKPOINT),
        "split": "test",
        "samples": len(y_true),
        "num_classes": len(class_to_idx),
        "logits_finite": finite,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": gloss_order, "matrix": cm.tolist()},
        "prediction_distribution": pred_counts,
        "unique_predicted_classes": unique_predicted,
        "dominant_predicted_class": dominant_class,
        "dominant_class_share": round(max_class_share, 4),
        "collapse_warning_one_class": unique_predicted <= 2,
        "collapse_warning_dominant_share": max_class_share >= 0.5,
        "signer_accuracy": signer_report,
        "signer_accuracy_min": round(min(signer_accuracies), 4) if signer_accuracies else None,
        "signer_accuracy_max": round(max(signer_accuracies), 4) if signer_accuracies else None,
        "training_val_accuracy_at_checkpoint": payload.get("val_accuracy"),
        "training_val_top5_at_checkpoint": payload.get("val_top5_accuracy"),
        "training_epoch": payload.get("epoch"),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k not in ("confusion_matrix", "signer_accuracy", "per_class")}, indent=2))
    print(f"Full report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
