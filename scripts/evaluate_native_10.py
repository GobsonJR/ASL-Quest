"""Phase 4: full test-set evaluation for the asl_citizen_native_10 best checkpoint.

Reuses checkpoint.py (load_checkpoint), model.py (build_model), dataset.py
(build_dataloader) unmodified. Adds the richer reporting (confusion matrix, per-class
P/R/F1, prediction distribution, signer breakdown, collapse/NaN diagnostics) that
src/asl_citizen/evaluate.py does not provide, without modifying that file.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.checkpoint import load_checkpoint
from src.asl_citizen.dataset import build_dataloader
from src.asl_citizen.metrics import batch_accuracy, top_k_accuracy
from src.asl_citizen.model import build_model
from src.asl_citizen.utils import choose_batch_size
from src.utils.device import get_device

CHECKPOINT_PATH = ROOT / "outputs" / "asl_citizen_native_10" / "checkpoints" / "asl_citizen_native_10_resnet18_gru_best.pth"
EXPECTED_CLASS_TO_IDX_PATH = ROOT / "data" / "asl_citizen_native_10" / "class_to_idx.json"
REPORT_PATH = ROOT / "outputs" / "asl_citizen_native_10" / "reports" / "test_evaluation_report.json"


@torch.no_grad()
def evaluate(checkpoint_path: Path, split: str = "test") -> dict:
    device = get_device()
    payload, cfg, class_to_idx, idx_to_class = load_checkpoint(checkpoint_path, map_location=device)

    # Checkpoint loading correctness: class mapping in the checkpoint must match the
    # current native_10 manifest's class_to_idx (same glosses, same indices).
    on_disk_mapping = json.loads(EXPECTED_CLASS_TO_IDX_PATH.read_text(encoding="utf-8"))
    on_disk_mapping = {k: int(v) for k, v in on_disk_mapping.items()}
    checkpoint_matches_manifest = class_to_idx == on_disk_mapping
    num_classes_correct = payload["num_classes"] == 10 == cfg.num_classes

    # KNOWN INFRA GAP (not modified here): checkpoint.py's config_to_dict/config_from_dict
    # do not persist class_to_idx_path/idx_to_class_path, so a reloaded cfg silently
    # defaults those two fields back to the asl_citizen_100 dataclass defaults for any
    # non-default experiment. build_dataloader() re-derives its own class_to_idx from
    # cfg.class_to_idx_path (ignoring the correct payload-level class_to_idx above), so
    # without this correction it would load the wrong (100-class) mapping and KeyError on
    # native_10 glosses. Restoring the correct paths here, locally, before building the
    # loader.
    idx_to_class_path = EXPECTED_CLASS_TO_IDX_PATH.parent / "idx_to_class.json"
    cfg = replace(cfg, class_to_idx_path=EXPECTED_CLASS_TO_IDX_PATH, idx_to_class_path=idx_to_class_path)

    model = build_model(cfg, pretrained=False)
    load_result = model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    loader, dataset = build_dataloader(
        split, cfg, batch_size=choose_batch_size(cfg.batch_size, device), shuffle=False
    )

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_participants: list[str] = []
    all_glosses: list[str] = []
    for batch in loader:
        clips = batch["video"].to(device)
        labels = batch["label"].to(device)
        logits = model(clips)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_participants.extend(batch["participant_id"])
        all_glosses.extend(batch["gloss"])

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)

    finite = torch.isfinite(logits_cat).all().item()
    preds = logits_cat.argmax(dim=1)

    idx_order = [int(i) for i in range(len(idx_to_class))]
    gloss_order = [idx_to_class[str(i)] for i in idx_order]

    y_true = labels_cat.numpy()
    y_pred = preds.numpy()

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

    # Per-signer breakdown (test split has 11 official participants; split is untouched,
    # this just groups already-computed predictions by participant_id for reporting).
    by_signer: dict[str, dict[str, int]] = {}
    for participant, true_idx, pred_idx in zip(all_participants, y_true.tolist(), y_pred.tolist()):
        entry = by_signer.setdefault(participant, {"correct": 0, "total": 0})
        entry["total"] += 1
        entry["correct"] += int(true_idx == pred_idx)
    signer_report = {
        participant: {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
        }
        for participant, v in sorted(by_signer.items())
    }
    signer_accuracies = [v["accuracy"] for v in signer_report.values()]

    results = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "samples": len(dataset),
        "num_classes": len(class_to_idx),
        "checkpoint_class_mapping_matches_manifest": checkpoint_matches_manifest,
        "checkpoint_num_classes_correct": num_classes_correct,
        "infra_note_class_to_idx_path_round_trip": (
            "checkpoint.py config_to_dict/config_from_dict do not persist "
            "class_to_idx_path/idx_to_class_path; corrected locally in this script "
            "before building the dataloader (not modified in checkpoint.py itself)."
        ),
        "state_dict_load_missing_keys": list(load_result.missing_keys),
        "state_dict_load_unexpected_keys": list(load_result.unexpected_keys),
        "logits_finite": finite,
        "top1_accuracy": batch_accuracy(logits_cat, labels_cat),
        "top5_accuracy": top_k_accuracy(logits_cat, labels_cat, k=5),
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
    return results


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint at {CHECKPOINT_PATH}")
        return
    results = evaluate(CHECKPOINT_PATH, "test")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k not in ("confusion_matrix", "signer_accuracy", "per_class")}, indent=2))
    print(f"Full report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
