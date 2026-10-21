"""Does the SCENE around the hand (face, room, lighting) change what the classifier says?

Uses captured live requests (see backend/services/live_diagnostics.py). For each bundle it classifies, with the
current checkpoint and identical preprocessing:

  recorded        the exact crop the app sent to the model
  hand only       the SAME frame with everything except the hand replaced by a plain wall colour like the
                  training photos (hand region taken from MediaPipe's 21 landmarks, thickened to cover the flesh)
  silhouette      the same hand region filled with one flat colour (control: is the SHAPE of the mask enough?)

and, with --dataset-control, applies the same masking to held-out dataset hands of OTHER letters (control: does
masking push predictions towards your target letter?). It is a DIAGNOSTIC: the mask is rough and the model was not
trained on masked input, so do not use it as a production preprocessing step.

    venv311\\Scripts\\python.exe scripts\\diagnose_background_dependence.py --target C --dataset-control

Detection runs first and is closed before torch is loaded, because both together can exhaust memory on small machines.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BEST_MODEL_PATH, DATASET_DIR, OUTPUTS_DIR  # noqa: E402
from src.inference.hands import HandDetector, crop_square, square_window  # noqa: E402

CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20], [5, 9, 13, 17]]
SKIN = (205, 165, 140)


def training_background() -> tuple[int, int, int]:
    """Median pixel over many whole training photos (all letters): a typical wall colour."""
    rng = random.Random(0)
    files = [f for c in sorted(DATASET_DIR.iterdir()) for f in rng.sample(sorted(c.glob("*.jpg")), 12)]
    medians = [np.median(np.array(Image.open(f).convert("RGB")).reshape(-1, 3), axis=0) for f in files]
    return tuple(int(v) for v in np.median(np.array(medians), axis=0))  # type: ignore[return-value]


def hand_mask(size: tuple[int, int], pts: np.ndarray, side: float) -> Image.Image:
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    r = max(2, int(side * 0.075))
    for chain in CHAINS:
        for a, b in zip(chain, chain[1:]):
            draw.line([tuple(pts[a]), tuple(pts[b])], fill=255, width=2 * r)
    draw.polygon([tuple(pts[i]) for i in (0, 1, 5, 9, 13, 17)], fill=255)
    for p in pts:
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(3))


def isolate(detector: HandDetector, frame: Image.Image, bg: tuple[int, int, int]):
    """Returns (hand-only crop, silhouette crop) or None when no hand is found."""
    rgb = np.array(frame.convert("RGB"))
    landmarks = detector._landmarks(rgb)  # noqa: SLF001 - diagnostic
    if not landmarks:
        return None
    h, w = rgb.shape[:2]
    pts = np.array([(x * w, y * h) for x, y in landmarks])
    x1, y1, x2, y2 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
    mask = hand_mask((w, h), pts, max(x2 - x1, y2 - y1))
    window = square_window((x1, y1, x2, y2), 0.30)
    plain = Image.new("RGB", (w, h), bg)
    hand_only = Image.composite(frame.convert("RGB"), plain, mask)
    silhouette = Image.composite(Image.new("RGB", (w, h), SKIN), plain, mask)
    return crop_square(np.array(hand_only), window), crop_square(np.array(silhouette), window)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=OUTPUTS_DIR / "diagnostics" / "live_requests")
    ap.add_argument("--target", default=None)
    ap.add_argument("--last", type=int, default=60)
    ap.add_argument("--dataset-control", action="store_true", help="also mask held-out dataset hands of other letters")
    args = ap.parse_args()

    bundles = []
    for folder in sorted(p for p in args.dir.glob("*") if (p / "result.json").exists()):
        record = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if args.target and (record.get("client", {}).get("target") or "").upper() != args.target.upper():
            continue
        bundles.append((folder, record))
    bundles = bundles[-args.last :]
    if not bundles:
        sys.exit(f"No captured bundles in {args.dir} (see the README: 'Capturing a REAL failing prediction').")
    target = (args.target or bundles[-1][1]["client"].get("target") or "C").upper()

    # ---- stage 1: hand detection + masking (no torch loaded yet) ----
    bg = training_background()
    detector = HandDetector(static_image_mode=True, detection_confidence=0.35)
    prepared = []
    for folder, record in bundles:
        frame = Image.open(next(folder.glob("1_original_frame.*")))
        iso = isolate(detector, frame, bg)
        if iso:
            prepared.append((folder.name, Image.open(folder / "3_classifier_crop.png").convert("RGB"), *iso))
    control = []
    if args.dataset_control:
        for letter in sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - {target}):
            got = 0
            for n in range(2551, 3001, 3):
                frame = Image.open(DATASET_DIR / letter / f"{letter}{n}.jpg")
                iso = isolate(detector, frame, bg)
                if iso:
                    control.append((letter, iso[0]))
                    got += 1
                if got >= 4:
                    break
    detector.close()

    # ---- stage 2: classification with the current checkpoint ----
    from src.inference.engine import Predictor

    predictor = Predictor(BEST_MODEL_PATH)
    print(f"checkpoint: {predictor.model_path}   training-photo background colour used: {bg}\n")

    def top(image: Image.Image) -> list[tuple[str, float]]:
        return [(t["label"], t["confidence"]) for t in predictor.predict_pil(image, top_k=5)["top_predictions"]]

    rows = [(name, top(rec), top(hand), top(sil)) for name, rec, hand, sil in prepared]
    n = len(rows)
    fmt = lambda t: f"{t[0][0]}:{t[0][1]:.2f}"  # noqa: E731
    print(f"{'bundle':22s} {'recorded':>9s} {'hand only':>10s} {'silhouette':>11s}")
    for name, rec, hand, sil in rows:
        print(f"{name:22s} {fmt(rec):>9s} {fmt(hand):>10s} {fmt(sil):>11s}")
    print(f"\n{n} bundles with a detectable hand (target {target}); top-1 = {target}:")
    for idx, label in ((1, "recorded crop (what the app sent)"), (2, "HAND ONLY on a plain wall"), (3, "silhouette only (control)")):
        hits = sum(r[idx][0][0] == target for r in rows)
        print(f"  {label:36s} {hits:3d}/{n}   labels: {dict(Counter(r[idx][0][0] for r in rows).most_common(5))}")
    if control:
        right = sum(top(img)[0][0] == letter for letter, img in control)
        as_target = sum(top(img)[0][0] == target for _, img in control)
        print(f"\ncontrol: the same masking on held-out dataset hands of OTHER letters (n={len(control)}): "
              f"still correct {right}/{len(control)}, predicted {target}: {as_target}/{len(control)}")
    print("\nIf 'hand only' is far better than 'recorded', the classifier is being misled by the scene around the hand.")
    print("Compare with the control rows before concluding: the mask must not simply push every hand towards the target.")


if __name__ == "__main__":
    main()
