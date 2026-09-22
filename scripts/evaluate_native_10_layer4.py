"""Evaluate only the Phase 4C best checkpoint on the official test split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_native_10 import evaluate
CHECKPOINT = ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "layer4" / "checkpoints" / "asl_citizen_native_10_layer4_resnet18_gru_best.pth"
REPORT = ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "layer4" / "reports" / "test_evaluation_report.json"


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Phase 4C best checkpoint not found: {CHECKPOINT}")
    results = evaluate(CHECKPOINT, "test")
    results["experiment_name"] = "asl_citizen_native_10_layer4"
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
