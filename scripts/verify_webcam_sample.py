"""Verify saved webcam samples end to end (MediaPipe only, no torch).

    venv311\\Scripts\\python.exe scripts\\verify_webcam_sample.py            # newest sample
    venv311\\Scripts\\python.exe scripts\\verify_webcam_sample.py --all      # every sample, summary only

For each sample it checks: the image file exists and decodes; the JSON label equals the folder letter and was
person-confirmed; the session record exists and its room/lighting/background match the sample; the crop regenerated
with the live app's own function is the one recorded at capture time (landmark box); and it writes the crop to
outputs/verify_webcam/<sample_id>.png so you can LOOK at it. Exit code 1 if anything fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = ["sample_id", "session_id", "letter", "label_source", "captured_at", "image", "hand", "distance", "mode",
            "lighting", "background", "quality", "dhash"]


def check(meta_path: Path, detector, out_dir: Path) -> list[str]:
    problems: list[str] = []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    folder = meta_path.parent
    for key in REQUIRED:
        if key not in meta:
            problems.append(f"missing field {key!r}")
    if problems:
        return problems
    if meta["letter"] != folder.name:
        problems.append(f"label {meta['letter']} != folder {folder.name}")
    if meta["label_source"] != "person_confirmed":
        problems.append(f"label_source is {meta['label_source']!r}")
    if not meta["sample_id"].startswith(meta["letter"] + "_"):
        problems.append("sample_id does not start with its letter")
    session_path = folder.parent / "session.json"
    if not session_path.exists():
        return problems + ["session.json missing"]
    session = json.loads(session_path.read_text(encoding="utf-8"))
    for key in ("lighting", "background"):
        if session.get(key) != meta[key]:
            problems.append(f"{key}: session {session.get(key)!r} != sample {meta[key]!r}")
    if session["session_id"] != meta["session_id"] or session["session_id"] != folder.parent.name:
        problems.append("session id mismatch")
    image_path = folder / meta["image"]
    if not image_path.exists():
        return problems + [f"image file missing: {meta['image']}"]
    try:
        frame = np.array(Image.open(image_path).convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        return problems + [f"image does not decode: {exc}"]
    crop = detector.crop_rgb(frame, padding=0.30)
    if not crop.detected:
        return problems + ["hand NOT detected when re-run on the saved image (it was accepted at capture time)"]
    recorded = meta["quality"]["metrics"].get("landmark_box")
    now = [round(float(v), 1) for v in crop.hand_box]
    if recorded and max(abs(a - b) for a, b in zip(recorded, now)) > 12:
        problems.append(f"hand box moved between capture {recorded} and re-detection {now} (video-mode tracking state?)")
    out_dir.mkdir(parents=True, exist_ok=True)
    crop.image.save(out_dir / f"{meta['sample_id']}.png")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "dataset" / "webcam")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "verify_webcam")
    args = ap.parse_args()
    metas = sorted(args.src.glob("s*/*/*.json"), key=lambda p: p.stat().st_mtime)
    if not metas:
        print(f"No samples under {args.src}. Save one through More > Collect data first.")
        return 1
    from backend.settings import HAND_DETECTION_CONFIDENCE
    from src.inference.hands import HandDetector

    # the same configuration as the live backend (backend/main.py) and the exporter, or "not detected" is a false alarm
    detector = HandDetector(static_image_mode=True, detection_confidence=HAND_DETECTION_CONFIDENCE)
    if not detector.available:
        print("hand detector unavailable:", detector.error)
        return 1
    targets = metas if args.all else metas[-1:]
    bad = 0
    for path in targets:
        problems = check(path, detector, args.out)
        meta = json.loads(path.read_text(encoding="utf-8"))
        status = "OK " if not problems else "BAD"
        bad += bool(problems)
        if not args.all or problems:
            q = meta.get("quality", {}).get("metrics", {})
            print(f"{status} {path.parent.parent.name}/{path.parent.name}/{path.name}  letter={meta.get('letter')} hand={meta.get('hand')} "
                  f"distance={meta.get('distance')} sharp={q.get('sharpness')} hand_px={q.get('hand_px')} frame={q.get('frame_size')}")
            for p in problems:
                print("     -", p)
    print(f"\n{len(targets) - bad}/{len(targets)} samples verified. Crops for inspection: {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
