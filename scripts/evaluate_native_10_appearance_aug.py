"""Evaluate only the Phase 6 appearance-augmentation best checkpoint on the official
test split. Reuses scripts/evaluate_native_10.py's evaluate() (unmodified) — same
procedure as Phase 4B/4C so results are directly comparable. The eval dataloader this
calls never applies color jitter (build_dataloader only jitters the "train" split)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_native_10 import evaluate

CHECKPOINT = (
    ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "appearance_aug" / "checkpoints"
    / "asl_citizen_native_10_appearance_aug_resnet18_gru_best.pth"
)
REPORT = (
    ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "appearance_aug" / "reports"
    / "test_evaluation_report.json"
)


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Phase 6 best checkpoint not found: {CHECKPOINT}")
    results = evaluate(CHECKPOINT, "test")
    results["experiment_name"] = "asl_citizen_native_10_appearance_aug"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({
        key: results[key]
        for key in (
            "top1_accuracy", "top5_accuracy", "macro_precision", "macro_recall",
            "macro_f1", "unique_predicted_classes", "dominant_predicted_class",
            "dominant_class_share", "logits_finite", "training_epoch",
        )
    }, indent=2))
    print(f"Full report -> {REPORT}")


if __name__ == "__main__":
    main()
