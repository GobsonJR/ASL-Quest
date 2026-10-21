"""Export collected webcam sessions into a training-ready, room-separated dataset.

    # hold out whole rooms automatically (needs >= 2 rooms; >= 3 for a validation room as well)
    venv311\\Scripts\\python.exe scripts\\export_webcam_dataset.py --auto

    # or choose rooms yourself (room names as shown in the report, e.g. "bedroom-shelf-wall")
    venv311\\Scripts\\python.exe scripts\\export_webcam_dataset.py --test-rooms kitchen --val-rooms hall

Crops are made with the same code path as the live app (square, aspect-preserving, 0.30 padding). Frames without a
detectable hand are skipped and counted. Nothing here ever uses a model prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.webcam_export import ExportError, assign_by_session, assign_splits, export_dataset, load_sessions  # noqa: E402
from src.inference.hands import HandDetector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "dataset" / "webcam")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset" / "webcam_export")
    ap.add_argument("--auto", action="store_true", help="hold out whole rooms deterministically")
    ap.add_argument("--test-rooms", nargs="*", default=[])
    ap.add_argument("--val-rooms", nargs="*", default=[])
    ap.add_argument("--allow-same-room", action="store_true", help="split by session when only one room exists (NOT an unseen-room test)")
    ap.add_argument("--test-sessions", nargs="*", default=[])
    ap.add_argument("--val-sessions", nargs="*", default=[])
    ap.add_argument("--pad", type=float, default=0.30)
    ap.add_argument("--hold-keep", type=int, default=3, help="frames kept per hold sequence in train/val")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sessions = load_sessions(args.src)
    total = sum(len(s["samples"]) for s in sessions.values())
    print(f"{len(sessions)} sessions, {total} collected frames; rooms: {sorted({s['room'] for s in sessions.values()})}")
    try:
        if args.allow_same_room and (args.test_sessions or args.val_sessions):
            assignment, notes = assign_by_session(sessions, args.test_sessions, args.val_sessions)
        else:
            assignment, notes = assign_splits(sessions, test_rooms=args.test_rooms, val_rooms=args.val_rooms, auto=args.auto, seed=args.seed)
    except ExportError as error:
        sys.exit(f"cannot export: {error}")

    detector = HandDetector(static_image_mode=True, detection_confidence=0.35)
    if not detector.available:
        sys.exit(f"hand detector unavailable: {detector.error}")
    report = export_dataset(args.src, args.out, detector, assignment=assignment, sessions=sessions, pad=args.pad, hold_keep=args.hold_keep, notes=notes)
    detector.close()

    print(json.dumps({k: report[k] for k in ("exported", "skipped", "per_split_total", "missing_letters", "leakage", "handedness_vs_mediapipe")}, indent=2))
    for note in report["notes"]:
        print("NOTE:", note)
    print(f"\nwritten to {args.out} (manifest.jsonl, splits.json, export_report.json)")


if __name__ == "__main__":
    main()
