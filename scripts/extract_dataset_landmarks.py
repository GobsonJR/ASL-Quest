"""EXPERIMENTAL: extract MediaPipe landmarks from the original dataset's TRAIN split, for the landmark classifier.

    venv311\\Scripts\\python.exe scripts\\extract_dataset_landmarks.py --attempt-per-class 300

Caveats, measured on this project: MediaPipe finds a hand in only about a third of these close-up photos (fists such as A/S/E/M/N/T
are the hardest), and it is unknown whether the dataset photos are mirrored selfie frames, so the handedness correction that is
right for the webcam may be wrong for them. Hand labels here therefore come from MediaPipe's own (consistent) label. Treat results
with this data as an experiment and compare with and without it on unseen webcam rooms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASET_DIR, OUTPUTS_DIR, SPLITS_PATH  # noqa: E402
from src.inference.hands import HandDetector  # noqa: E402

LETTERS = [chr(65 + i) for i in range(26)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempt-per-class", type=int, default=300)
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "landmarks" / "dataset_landmarks.npz")
    args = ap.parse_args()

    train_idx = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))["splits"]["train"]
    files: list[Path] = []
    for cls in sorted(os.listdir(DATASET_DIR)):
        files.extend(DATASET_DIR / cls / f for f in sorted(os.listdir(DATASET_DIR / cls)))
    by_letter: dict[str, list[Path]] = {l: [] for l in LETTERS}
    for i in train_idx:
        by_letter[files[i].parent.name].append(files[i])

    detector = HandDetector(static_image_mode=True, detection_confidence=0.35)
    points, hands, letters = [], [], []
    for letter, paths in by_letter.items():
        chosen = paths[:: max(1, len(paths) // args.attempt_per_class)][: args.attempt_per_class]
        found = 0
        for path in chosen:
            obs = detector.observe(np.array(Image.open(path).convert("RGB")))
            if obs is None or obs.handedness is None:
                continue
            points.append(np.array(obs.landmarks))
            hands.append(obs.actual_hand or "right")
            letters.append(letter)
            found += 1
        print(f"{letter}: {found}/{len(chosen)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, landmarks=np.array(points), hands=np.array(hands), letters=np.array(letters))
    print(f"saved {len(points)} samples to {args.out}")


if __name__ == "__main__":
    main()
