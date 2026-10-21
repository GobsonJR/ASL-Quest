"""Train the landmark-based classifier (Approach B) from exported webcam sessions.

    venv311\\Scripts\\python.exe scripts\\train_landmark_classifier.py

Uses the `train` rows of dataset/webcam_export (and its `val` rows, from rooms never trained on, to choose the feature
configuration and model type; they are added back for the final fit). With no validation room it falls back to
leave-room-out cross-validation, and with a single room it refuses: selecting a model on the data it was trained on
would only measure memorisation. Evaluate the saved model with scripts/evaluate_webcam.py --approach landmarks.
No torch is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.landmarks.classifier import save_model, select_and_train, with_deployed_hand  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_rows(export: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in (export / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["split"] == split and r.get("landmarks")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, default=ROOT / "dataset" / "webcam_export")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "landmark_classifier.joblib")
    ap.add_argument("--report", type=Path, default=ROOT / "outputs" / "landmarks" / "training_report.json")
    ap.add_argument("--augment", type=int, default=4, help="augmented landmark copies per training sample (training only)")
    ap.add_argument("--extra-dataset-landmarks", type=Path, default=None, help="optional .npz from scripts/extract_dataset_landmarks.py (experimental)")
    ap.add_argument("--hand-source", choices=["mediapipe", "label"], default="mediapipe", help="which hand to mirror by: MediaPipe's (what the live app uses, default) or your label")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = with_deployed_hand(load_rows(args.export, "train"), args.hand_source)
    val = with_deployed_hand(load_rows(args.export, "val"), args.hand_source)
    if args.extra_dataset_landmarks:
        data = np.load(args.extra_dataset_landmarks, allow_pickle=False)
        extra = [{"landmarks": lm.tolist(), "hand": str(h), "letter": str(l), "room": "original-dataset"} for lm, h, l in zip(data["landmarks"], data["hands"], data["letters"])]
        print(f"adding {len(extra)} landmark samples extracted from the original dataset (experimental)")
        train += extra
    print(f"train rows: {len(train)} ({len({r['room'] for r in train})} rooms), validation rows: {len(val)}")
    counts = Counter(r["letter"] for r in train)
    missing = [chr(65 + i) for i in range(26) if counts.get(chr(65 + i), 0) == 0]
    if missing:
        print(f"WARNING: no training samples for letters {missing}; the model can never predict them")
    try:
        model, report = select_and_train(train, val, augment=args.augment, seed=args.seed)
    except ValueError as error:
        sys.exit(f"cannot train: {error}")
    save_model(model, args.out)
    report.update(train_rows=len(train), val_rows=len(val), per_letter_train=dict(sorted(counts.items())), missing_letters=missing, model_file=str(args.out))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, score in sorted(report["selection_scores_macro_f1"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} macro-F1 {100 * score:5.1f}")
    print(f"selected {report['selected']} (chosen on {report['selection_data']})\nsaved {args.out}")


if __name__ == "__main__":
    main()
