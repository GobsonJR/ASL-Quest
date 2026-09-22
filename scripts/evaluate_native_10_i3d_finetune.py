"""Phase 8 Task 7: evaluate the best partial-fine-tune checkpoint on the official
native-10 test split, AND cache its 1024-dim pooled features (from the same forward
pass) for the representation analysis in
scripts/diagnose_native_10_i3d_finetune.py.

Unlike Phase 7 (frozen backbone -> features could be cached once and reused forever),
here the backbone's Mixed_5c block changed during training, so features must be
recomputed from THIS checkpoint specifically. One forward pass per clip produces both
the classification logits and the pooled feature (via extract_features + the model's
own dropout/logits head applied manually), avoiding a duplicate backbone pass.
"""

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

from src.asl_citizen.config import ASL_CITIZEN_VIDEOS_DIR
from src.asl_citizen.utils import load_class_mappings
from src.i3d_transfer.dataset import I3DClipDataset
from src.i3d_transfer.pytorch_i3d import InceptionI3d
from torch.utils.data import DataLoader

DATA_DIR = ROOT / "data" / "asl_citizen_native_10_i3d"
CHECKPOINT = ROOT / "outputs" / "asl_citizen_native_10_i3d" / "finetune" / "checkpoints" / "finetune_best.pt"
FEATURES_OUT = ROOT / "outputs" / "asl_citizen_native_10_i3d" / "finetune" / "features" / "test_features.npz"
REPORT_PATH = ROOT / "outputs" / "asl_citizen_native_10_i3d" / "finetune" / "reports" / "test_evaluation_report.json"


def top_k_accuracy_np(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, logits.shape[1])
    topk = np.argsort(-logits, axis=1)[:, :k]
    correct = (topk == labels[:, None]).any(axis=1)
    return float(correct.mean())


@torch.no_grad()
def run_inference(device: torch.device) -> dict:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    class_to_idx = {str(k): int(v) for k, v in payload["class_to_idx"].items()}
    idx_to_class = {str(k): str(v) for k, v in payload["idx_to_class"].items()}

    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(len(class_to_idx))
    load_result = i3d.load_state_dict(payload["model_state_dict"], strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys, (
        f"finetune checkpoint/architecture mismatch: missing={load_result.missing_keys} "
        f"unexpected={load_result.unexpected_keys}"
    )
    i3d.to(device)
    i3d.eval()

    test_ds = I3DClipDataset(DATA_DIR / "test.csv", ASL_CITIZEN_VIDEOS_DIR, class_to_idx, seed=42)
    loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

    all_feats, all_logits, all_labels = [], [], []
    all_participants, all_glosses, all_files = [], [], []

    for batch in loader:
        clips = batch["video"].to(device)
        labels = batch["label"]
        feat_map = i3d.extract_features(clips)  # (B, 1024, T', 1, 1)
        logits_per_frame = i3d.logits(i3d.dropout(feat_map)).squeeze(3).squeeze(3)  # (B, C, T')
        logits = logits_per_frame.mean(dim=2)  # (B, C)
        pooled_feat = feat_map.mean(dim=2).squeeze(-1).squeeze(-1)  # (B, 1024)

        all_feats.append(pooled_feat.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_labels.extend(labels.tolist() if torch.is_tensor(labels) else list(labels))
        all_participants.extend(batch["participant_id"])
        all_glosses.extend(batch["gloss"])
        all_files.extend(batch["video_file"])

    features = np.concatenate(all_feats, axis=0)
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.array(all_labels, dtype=np.int64)

    assert np.isfinite(features).all(), "non-finite pooled features"
    assert np.isfinite(logits_np).all(), "non-finite logits"

    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        FEATURES_OUT,
        features=features,
        labels=labels_np,
        participants=np.array(all_participants),
        glosses=np.array(all_glosses),
        video_files=np.array(all_files),
    )
    print(f"cached test features -> {FEATURES_OUT.relative_to(ROOT)}")

    return {
        "logits": logits_np,
        "labels": labels_np,
        "participants": all_participants,
        "idx_to_class": idx_to_class,
        "class_to_idx": class_to_idx,
        "payload": payload,
    }


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"finetune checkpoint not found: {CHECKPOINT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_inference(device)

    logits_np = result["logits"]
    y_true = result["labels"]
    y_pred = logits_np.argmax(axis=1)
    idx_to_class = result["idx_to_class"]
    participants = result["participants"]
    payload = result["payload"]

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
        "experiment_name": "asl_citizen_native_10_i3d_finetune",
        "checkpoint": str(CHECKPOINT),
        "split": "test",
        "samples": len(y_true),
        "num_classes": len(gloss_order),
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
