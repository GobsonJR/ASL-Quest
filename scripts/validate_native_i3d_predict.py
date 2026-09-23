"""Smallest safe validation for src/i3d_transfer/predict.py.

Runs ONE known native-10 test clip through the new single-clip inference
function and checks that the pipeline mechanically works end-to-end:
video load -> preprocessing -> frozen I3D feature extraction -> linear head
-> a valid prediction among the 10 known native-10 classes.

This does NOT evaluate model accuracy and does NOT expect the prediction to
be correct for this arbitrary clip -- only that the pipeline runs and returns
a well-formed result. Read-only: does not modify any checkpoint, dataset, or
manifest file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.config import ASL_CITIZEN_VIDEOS_DIR
from src.asl_citizen.utils import load_manifest_rows, resolve_video_path
from src.i3d_transfer.predict import DEFAULT_HEAD_CHECKPOINT, predict_video

MANIFEST = ROOT / "data" / "asl_citizen_native_10_i3d" / "test.csv"
EXPECTED_CLASSES = {"BOOK", "EAT1", "HELLO", "HELP", "MOTHER", "NO", "PLEASE", "THANKYOU", "WATER", "YES"}


def main() -> None:
    rows = load_manifest_rows(MANIFEST)
    row = rows[0]
    video_path = resolve_video_path(ASL_CITIZEN_VIDEOS_DIR, row["video_file"])
    print(f"Validation clip: {row['video_file']} (true gloss: {row['gloss']}, participant: {row['participant_id']})")
    print(f"Resolved path: {video_path}")
    print(f"Head checkpoint: {DEFAULT_HEAD_CHECKPOINT}")

    result = predict_video(video_path, checkpoint=DEFAULT_HEAD_CHECKPOINT, top_k=5)
    print("\nRaw result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    # --- Validation checks (pipeline mechanics, not accuracy) ---
    assert result["prediction"] in EXPECTED_CLASSES, f"prediction {result['prediction']!r} not in the 10 known classes"
    assert result["num_classes"] == 10, f"expected 10 classes, got {result['num_classes']}"

    conf = result["confidence"]
    assert conf == conf, "confidence is NaN"  # NaN != NaN
    assert conf not in (float("inf"), float("-inf")), "confidence is infinite"
    assert 0.0 <= conf <= 1.0, f"confidence {conf} not in [0, 1]"

    top_k = result["top_k"]
    assert len(top_k) > 0, "top_k is empty"
    assert len(top_k) <= 5, f"expected at most 5 top_k entries, got {len(top_k)}"
    prev_conf = 1.0 + 1e-6
    for entry in top_k:
        assert entry["gloss"] in EXPECTED_CLASSES, f"top_k gloss {entry['gloss']!r} not in the 10 known classes"
        c = entry["confidence"]
        assert c == c and 0.0 <= c <= 1.0, f"top_k confidence {c} invalid"
        assert c <= prev_conf, "top_k entries are not sorted descending by confidence"
        prev_conf = c
    assert top_k[0]["gloss"] == result["prediction"], "top_k[0] does not match the reported prediction"

    print("\nALL VALIDATION CHECKS PASSED.")
    print("(Prediction correctness was not evaluated -- this validates the pipeline mechanics only.)")


if __name__ == "__main__":
    main()
