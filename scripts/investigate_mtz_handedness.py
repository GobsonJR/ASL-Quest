"""READ-ONLY follow-up: does the MediaPipe handedness LABEL (not the hand's geometry) decide M/T/S/E... answers?

Phase 1 (`extract`)  detect landmarks once for every held-out Kaggle test image, store them with MediaPipe's raw label.
Phase 2 (`report`)   compare two ways of deciding whether to mirror a hand before classification, on identical landmarks:
                       label     what the app does now: mirror when MediaPipe says "Right" (-> actual "left")
                       geometry  mirror when the hand's own image-plane orientation says so (sign of index-MCP x pinky-MCP),
                                 which is what the label is standing in for, and does not depend on MediaPipe's classifier.
Also measures how often the hand's label disagrees with the geometry, and what a per-frame label flicker does to each.

    venv311\\Scripts\\python.exe scripts\\investigate_mtz_handedness.py extract
    venv311\\Scripts\\python.exe scripts\\investigate_mtz_handedness.py report

Nothing is trained or written into models/.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.landmarks.classifier import LETTERS, LandmarkClassifier  # noqa: E402
from src.landmarks.features import feature_vector  # noqa: E402

OUT = ROOT / "outputs" / "mtz_investigation"
NPZ = OUT / "test_landmarks.npz"
MODEL = ROOT / "models" / "experiments" / "landmark_L2_s0.joblib"
MP_TO_ACTUAL = {"Left": "right", "Right": "left"}


def geometry_hand(points) -> str:
    """'left' when the hand must be mirrored to reach the canonical orientation the model was fitted on, else 'right'."""
    p = np.asarray(points, dtype=np.float64)
    v1, v2 = p[5, :2] - p[0, :2], p[17, :2] - p[0, :2]
    return "left" if v1[0] * v2[1] - v1[1] * v2[0] < 0 else "right"


def extract() -> None:
    from backend.settings import HAND_DETECTION_CONFIDENCE
    from compare_resnet_landmarks import dataset_items
    from src.inference.hands import HandDetector

    OUT.mkdir(parents=True, exist_ok=True)
    detector = HandDetector(static_image_mode=True, detection_confidence=HAND_DETECTION_CONFIDENCE)
    items = dataset_items(None)
    lms, letters, raw, score, ids = [], [], [], [], []
    for n, item in enumerate(items, 1):
        obs = detector.observe(np.array(Image.open(item["path"]).convert("RGB")))
        if obs is None:
            continue
        lms.append(np.asarray(obs.landmarks))
        letters.append(item["letter"])
        raw.append(obs.handedness or "")
        score.append(obs.handedness_score or 0.0)
        ids.append(item["id"])
        if n % 1000 == 0:
            print(f"{n}/{len(items)}  kept {len(lms)}", flush=True)
    np.savez(NPZ, landmarks=np.array(lms), letters=np.array(letters), raw=np.array(raw), score=np.array(score), ids=np.array(ids))
    print(f"saved {len(lms)} detected test frames to {NPZ}")


def report() -> None:
    clf = LandmarkClassifier(MODEL)
    d = np.load(NPZ, allow_pickle=False)
    lms, letters, raw = d["landmarks"], d["letters"], d["raw"]
    label_hand = np.array([MP_TO_ACTUAL.get(r, "right") for r in raw])
    geo_hand = np.array([geometry_hand(p) for p in lms])
    flicker = np.array(["right" if h == "left" else "left" for h in label_hand])

    def predict(hands) -> np.ndarray:
        x = np.array([feature_vector(p, h, rotate=clf.rotate) for p, h in zip(lms, hands)])
        return np.array(LETTERS)[clf.pipeline.predict(x).astype(int)]

    preds = {"label": predict(label_hand), "geometry": predict(geo_hand), "label_flipped": predict(flicker)}
    lines = ["# Handedness label vs hand geometry (landmark_L2_s0, held-out Kaggle test frames with a detected hand)", "",
             f"{len(lms)} frames. Label and geometry disagree on {100 * np.mean(label_hand != geo_hand):.1f}% of them. "
             f"MediaPipe raw label counts: {dict(Counter(raw.tolist()))}.", "",
             "| letter | n | label (as deployed) | geometry | label flipped (flicker) | label-vs-geometry disagreements |", "|---|---|---|---|---|---|"]
    summary = {}
    for l in ["M", "T", "Z", "N", "S", "E", "D", "G", "H", "I", "K", "L", "P", "R"]:
        m = letters == l
        if not m.any():
            continue
        acc = {k: float(np.mean(v[m] == l)) for k, v in preds.items()}
        dis = float(np.mean(label_hand[m] != geo_hand[m]))
        summary[l] = {"n": int(m.sum()), **acc, "disagree": dis}
        lines.append(f"| {l} | {int(m.sum())} | {100 * acc['label']:.1f}% | {100 * acc['geometry']:.1f}% | {100 * acc['label_flipped']:.1f}% | {100 * dis:.1f}% |")
    allacc = {k: float(np.mean(v == letters)) for k, v in preds.items()}
    lines += ["", "All letters (detected frames): " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in allacc.items()) + ".", ""]
    for l in ("M", "T"):
        m = letters == l
        for k in ("label", "geometry"):
            wrong = Counter(preds[k][m & (preds[k] != l)].tolist())
            lines.append(f"- {l} / {k}: wrong answers {dict(wrong.most_common(4))}")
    summary["all"] = allacc
    (OUT / "HANDEDNESS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "handedness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("extract", "report"):
        sys.exit(__doc__)
    extract() if sys.argv[1] == "extract" else report()
