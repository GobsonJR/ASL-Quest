"""Phase 3: experimental landmark classifiers, trained ONLY on data that is separated from the evaluation room.

    venv311\\Scripts\\python.exe scripts\\diag_landmark_train.py --tag L1          # original-dataset landmarks only
    venv311\\Scripts\\python.exe scripts\\diag_landmark_train.py --tag L2          # + the plain-wall webcam room (A, C, F)

Inputs : outputs/diagnostics/landmarks/dataset_landmarks.npz (scripts/extract_dataset_landmarks.py: TRAIN split of the
         original dataset only) and, for L2, the TRAIN split of dataset/webcam_export_holdout (room plain-wall).
Output : models/experiments/landmark_<tag>_s0.joblib (new file; refuses to overwrite) and outputs/diagnostics/landmark_<tag>_report.json
Model selection: leave-block-out cross-validation. The original dataset is split into 5 contiguous blocks per letter (filename order
follows capture order, so neighbouring near-identical frames stay in the same block); L2 adds plain-wall as another group.
The evaluation room (bedroom) is never used for training or selection. No torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.landmarks import classifier as lc  # noqa: E402

BLOCKS = 5


def dataset_rows(npz: Path) -> list[dict]:
    data = np.load(npz, allow_pickle=False)
    letters = [str(x) for x in data["letters"]]
    seen: Counter[str] = Counter()
    totals = Counter(letters)
    rows = []
    for lm, hand, letter in zip(data["landmarks"], data["hands"], letters):
        k = min(BLOCKS - 1, seen[letter] * BLOCKS // totals[letter])
        seen[letter] += 1
        rows.append({"landmarks": lm.tolist(), "hand": str(hand), "letter": letter, "room": f"original-block{k}"})
    return rows


def webcam_train_rows(export: Path, repeat: int) -> list[dict]:
    manifest = [json.loads(line) for line in (export / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in manifest if r["split"] == "train" and r.get("landmarks")]
    rows = lc.with_deployed_hand(rows, "mediapipe")
    return [{"landmarks": r["landmarks"], "hand": r["hand"], "letter": r["letter"], "room": r["room"]} for r in rows for _ in range(repeat)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["L1", "L2"])
    ap.add_argument("--npz", type=Path, default=ROOT / "outputs" / "diagnostics" / "landmarks" / "dataset_landmarks.npz")
    ap.add_argument("--export", type=Path, default=ROOT / "dataset" / "webcam_export_holdout")
    ap.add_argument("--webcam-repeat", type=int, default=10, help="L2: copies of each webcam row (the original set has ~100x more rows)")
    ap.add_argument("--augment", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = ROOT / "models" / "experiments" / f"landmark_{args.tag}_s{args.seed}.joblib"
    if out.exists():
        sys.exit(f"{out} already exists; choose a different --seed/--tag (existing experiment files are never overwritten)")

    rows = dataset_rows(args.npz)
    per_letter_original = Counter(r["letter"] for r in rows)
    if args.tag == "L2":
        rows += webcam_train_rows(args.export, args.webcam_repeat)
    print(f"{args.tag}: {len(rows)} training rows; letters with fewer than 100 original samples: "
          f"{[l for l, n in sorted(per_letter_original.items()) if n < 100]}")

    # two cheap candidate models are enough for a diagnostic (gradient boosting x 26 classes is slow on this machine)
    full = lc.candidate_models
    lc.candidate_models = lambda seed=0: {k: v for k, v in full(seed).items() if k in ("logreg", "mlp")}  # type: ignore[assignment]
    model, report = lc.select_and_train(rows, [], augment=args.augment, seed=args.seed)
    lc.save_model(model, out)
    report.update(tag=args.tag, model_file=str(out), n_rows=len(rows), original_per_letter=dict(sorted(per_letter_original.items())),
                  webcam_rows_in_training=sum(r["room"] == "plain-wall" for r in rows), webcam_repeat=args.webcam_repeat if args.tag == "L2" else 0,
                  candidates="logreg, mlp (hgb skipped for speed)", evaluation_room_used_for_training_or_selection=False)
    for name, score in sorted(report["selection_scores_macro_f1"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} leave-block-out macro-F1 {100 * score:5.1f}")
    print("selected", report["selected"], "->", out)
    (ROOT / "outputs" / "diagnostics" / f"landmark_{args.tag}_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
