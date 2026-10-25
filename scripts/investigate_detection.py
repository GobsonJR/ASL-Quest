"""READ-ONLY: why does MediaPipe miss ~65% of the close-up dataset images? One factor at a time, on a fixed sample.

Nothing in the production pipeline is changed; every variant below is an experiment on copies of the images.

Candidate causes and the variant that isolates each:
  image size            upscale 200x200 -> 400 / 640 / 1024 px (content unchanged)
  crop geometry         pad the frame so the hand occupies a smaller share (grey / edge-replicated border, 50% and 100% per side)
  preprocessing         BGR instead of RGB; grayscale; brightness-normalised (CLAHE on luminance)
  model configuration   detection/presence confidence 0.50 (MediaPipe default), 0.35 (app), 0.20, 0.10, 0.05
Reverse experiment on real webcam frames (which MediaPipe finds 100% of the time as recorded): crop tightly around the hand and
shrink to 200x200, to see whether close-up FRAMING alone reproduces the failures.

    venv311\\Scripts\\python.exe scripts\\investigate_detection.py --per-class 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.hands import HandDetector  # noqa: E402
from src.landmarks.classifier import LandmarkClassifier  # noqa: E402

OUT = ROOT / "outputs" / "recognition_investigation"
WEBCAM_RAW = ROOT / "dataset" / "webcam"


def load_compare():
    spec = importlib.util.spec_from_file_location("compare_resnet_landmarks", ROOT / "scripts" / "compare_resnet_landmarks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pad(rgb: np.ndarray, fraction: float, mode: str) -> np.ndarray:
    h, w = rgb.shape[:2]
    p = int(round(max(h, w) * fraction))
    if mode == "edge":
        return np.pad(rgb, ((p, p), (p, p), (0, 0)), mode="edge")
    return np.pad(rgb, ((p, p), (p, p), (0, 0)), mode="constant", constant_values=128)


def upscale(rgb: np.ndarray, side: int) -> np.ndarray:
    return np.array(Image.fromarray(rgb).resize((side, side), Image.Resampling.LANCZOS))


def clahe(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def rate(detector: HandDetector, images: list[np.ndarray]) -> tuple[int, list]:
    found = [detector.observe(im) for im in images]
    return sum(o is not None for o in found), found


def line(name: str, hits: int, n: int, base: int | None = None) -> str:
    extra = "" if base is None else f"  ({hits - base:+d} vs baseline)"
    return f"| {name} | {hits}/{n} | {100 * hits / n:.1f}% | {100 * (n - hits) / n:.1f}%{extra} |"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    compare = load_compare()
    items = compare.dataset_items(args.per_class)
    images = [np.array(Image.open(i["path"]).convert("RGB")) for i in items]
    letters = [i["letter"] for i in items]
    n = len(images)
    print(f"{n} dataset test images, sizes {sorted({im.shape[:2] for im in images})}", flush=True)
    lines = ["# MediaPipe detection on close-up dataset images (measured)", "",
             f"Sample: {n} images ({args.per_class}/class, evenly spaced from the dataset TEST split). Native size {sorted({im.shape[:2] for im in images})}. "
             "Baseline = exactly what the backend builds: HandDetector(static_image_mode=True, detection_confidence=0.35), RGB, native size.", "",
             "| variant | detected | detection rate | failure rate |", "|---|---|---|---|"]

    base_det = HandDetector(static_image_mode=True, detection_confidence=0.35)
    base_hits, base_obs = rate(base_det, images)
    lines.append(line("baseline (native 200px, conf 0.35, RGB)", base_hits, n))
    results = {"baseline": base_hits}

    def run(name: str, transform, conf: float = 0.35, detector: HandDetector | None = None):
        det = detector or HandDetector(static_image_mode=True, detection_confidence=conf)
        hits, obs = rate(det, [transform(im) for im in images])
        lines.append(line(name, hits, n, base_hits))
        results[name] = hits
        return obs

    lines.append("| **image size** | | | |")
    for side in (400, 640, 1024):
        run(f"upscale to {side}px", lambda im, s=side: upscale(im, s))
    lines.append("| **crop geometry (hand share of frame)** | | | |")
    pad_obs = {}
    for frac in (0.25, 0.5, 1.0):
        pad_obs[f"grey {frac}"] = run(f"grey border {int(frac * 100)}% each side", lambda im, f=frac: pad(im, f, "grey"))
        run(f"edge-replicated border {int(frac * 100)}% each side", lambda im, f=frac: pad(im, f, "edge"))
    best_pad = run("grey border 100% + upscale to 640px", lambda im: upscale(pad(im, 1.0, "grey"), 640))
    lines.append("| **preprocessing** | | | |")
    run("BGR channel order (wrong order, as a control)", lambda im: np.ascontiguousarray(im[:, :, ::-1]))
    run("grayscale (3 channels)", lambda im: np.repeat(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)[:, :, None], 3, axis=2))
    run("CLAHE luminance normalisation", clahe)
    lines.append("| **model configuration** | | | |")
    for conf in (0.5, 0.2, 0.1, 0.05):
        run(f"confidence {conf}", lambda im: im, conf=conf)

    # per-letter failure at baseline, and failure vs image statistics
    lines += ["", "### Baseline failures by image statistics", ""]
    fail = np.array([o is None for o in base_obs])
    bright = np.array([im.mean() for im in images])
    contrast = np.array([im.std() for im in images])
    for title, vals in (("mean brightness", bright), ("contrast (std)", contrast)):
        cuts = np.quantile(vals, [1 / 3, 2 / 3])
        names = ["low", "mid", "high"]
        parts = [f"{nm} {np.mean(fail[(vals >= lo) & (vals <= hi)]) * 100:.0f}% failing" for nm, lo, hi in zip(names, [vals.min(), cuts[0], cuts[1]], [cuts[0], cuts[1], vals.max()])]
        lines.append(f"- by {title} tercile ({cuts[0]:.0f}/{cuts[1]:.0f}): " + ", ".join(parts))
    per_letter = {l: float(np.mean(fail[[i for i, x in enumerate(letters) if x == l]])) for l in sorted(set(letters))}
    lines.append("- baseline failure by letter: " + ", ".join(f"{l} {100 * v:.0f}%" for l, v in sorted(per_letter.items(), key=lambda kv: -kv[1])))

    # do recovered detections give usable landmarks?  (landmark models only; no torch)
    lines += ["", "### Are detections recovered by padding usable? (L2 landmark model, hand label as deployed)", ""]
    clf = LandmarkClassifier(ROOT / "models/experiments/landmark_L2_s0.joblib")
    groups = {"found at baseline": [], "recovered only with 100% grey border": []}
    for i, (b, p) in enumerate(zip(base_obs, pad_obs["grey 1.0"])):
        if p is None:
            continue
        key = "found at baseline" if b is not None else "recovered only with 100% grey border"
        groups[key].append((letters[i], clf.predict_observation(p)["prediction"]))
    for key, pairs in groups.items():
        lines.append(f"- {key}: n={len(pairs)}, L2 accuracy {100 * np.mean([t == q for t, q in pairs]):.1f}%" if pairs else f"- {key}: n=0")
    lines.append("(the same L2 model scores on the SAME images' padded-view landmarks, so this checks landmark quality, not the app's behaviour)")

    # reverse experiment on webcam frames
    lines += ["", "## Reverse experiment: real webcam frames made to look like the dataset's close-ups", ""]
    frames = []
    for sj in sorted(WEBCAM_RAW.glob("s*/session.json")):
        for meta in sorted(sj.parent.glob("*/*.json")):
            rec = json.loads(meta.read_text(encoding="utf-8"))
            frames.append(sj.parent / rec["letter"] / rec["image"])
    frames = [f for f in frames if f.exists()]
    web = [np.array(Image.open(f).convert("RGB")) for f in frames]
    wdet = HandDetector(static_image_mode=True, detection_confidence=0.35)
    whits, wobs = rate(wdet, web)
    lines += [f"{len(web)} collected webcam frames (these were saved only because a hand WAS detected, so 100% is by construction).", "",
              "| variant | detected | detection rate | failure rate |", "|---|---|---|---|", line("original frame", whits, len(web))]

    def tight(im, o, margin: float, side: int | None):
        h, w = im.shape[:2]
        xs = [p[0] * w for p in o.landmarks]; ys = [p[1] * h for p in o.landmarks]
        half = max(max(xs) - min(xs), max(ys) - min(ys)) * (0.5 + margin)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        window = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        crop = np.array(compare_crop(im, window))
        return upscale(crop, side) if side else crop

    from src.inference.hands import crop_square as compare_crop
    for margin in (0.05, 0.3, 0.6):
        for side in (None, 200):
            variants = [tight(im, o, margin, side) for im, o in zip(web, wobs) if o is not None]
            hits, _ = rate(HandDetector(static_image_mode=True, detection_confidence=0.35), variants)
            name = f"hand crop, margin {int(margin * 100)}%" + (", resized to 200x200" if side else ", native pixels")
            lines.append(line(name, hits, len(variants)))
    lines += ["", "Reading the table: if tightening the framing alone drops detection on webcam frames the way it does on the dataset, framing is a cause; "
              "if the webcam crops are still found, the dataset's failures come from something else in those images (e.g. dark backlit silhouettes, see the statistics above)."]

    (OUT / "detection.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "detection.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
