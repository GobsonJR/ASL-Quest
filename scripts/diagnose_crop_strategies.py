"""Measure how the input framing changes classifier accuracy on held-out dataset images.

Why this exists: the live /predict?require_hand=true path crops the image to the MediaPipe
landmark box (+30% padding) and the classifier then resizes that *non-square* crop to
224x224, stretching it. The model was trained on whole, roughly square dataset photos. This
script compares framings on the controlled test split (outputs/splits.json), so the choice is
backed by measurements instead of guesses.

Usage (from repo root):
    venv311\\Scripts\\python.exe scripts\\diagnose_crop_strategies.py --per-class 14
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASET_DIR, SPLITS_PATH  # noqa: E402
from src.data.dataset import load_image_folder, load_split_indices  # noqa: E402
from src.inference.engine import Predictor  # noqa: E402
from src.inference.hands import HandDetector, crop_square, square_window  # noqa: E402


def sample_test_images(per_class: int) -> list[tuple[Path, str]]:
    folder = load_image_folder(DATASET_DIR, transform=None)
    test = load_split_indices(SPLITS_PATH)["test"]
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx in test:
        by_class[folder.samples[idx][1]].append(idx)
    chosen: list[tuple[Path, str]] = []
    for label, indices in sorted(by_class.items()):
        step = max(1, len(indices) // per_class)
        for idx in indices[::step][:per_class]:
            chosen.append((Path(folder.samples[idx][0]), folder.classes[label]))
    return chosen


def legacy_tight_stretched(rgb: np.ndarray, box: tuple[float, float, float, float], padding: float = 0.30) -> Image.Image:
    """The ORIGINAL crop (kept only for comparison): landmark rectangle + padding, not square."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    left, top = max(0, int(x1 - bw * padding)), max(0, int(y1 - bh * padding))
    right, bottom = min(w, int(x2 + bw * padding)), min(h, int(y2 + bh * padding))
    return Image.fromarray(rgb[top:bottom, left:right])


def square_crop(rgb: np.ndarray, box: tuple[float, float, float, float], pad: float) -> Image.Image:
    """The production crop (src.inference.hands): square, aspect-preserving."""
    return crop_square(rgb, square_window(box, pad))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=14)
    parser.add_argument("--scene", action="store_true", help="also paste each photo small into a 960x720 scene (webcam-like framing)")
    args = parser.parse_args()

    predictor = Predictor()
    detector = HandDetector(static_image_mode=True)
    print(f"model={predictor.model_path.name} hand_detector_available={detector.available}")
    items = sample_test_images(args.per_class)

    strategies = ["full_image", "current_tight_stretched", "square_pad0.15", "square_pad0.30", "square_pad0.60"]
    correct = {s: 0 for s in strategies}
    attempted = {s: 0 for s in strategies}
    per_letter: dict[str, dict[str, list[int]]] = {s: defaultdict(lambda: [0, 0]) for s in strategies}
    detected = 0

    for path, truth in items:
        with Image.open(path) as im:
            im.load()
            rgb = np.array(im.convert("RGB"))
        full = Image.fromarray(rgb)

        def score(name: str, image: Image.Image) -> None:
            attempted[name] += 1
            per_letter[name][truth][1] += 1
            if predictor.predict_pil(image)["prediction"] == truth:
                correct[name] += 1
                per_letter[name][truth][0] += 1

        score("full_image", full)
        box = detector.landmark_box(rgb)
        if box is None:
            continue
        detected += 1
        score("current_tight_stretched", legacy_tight_stretched(rgb, box))
        for pad in (0.15, 0.30, 0.60):
            score(f"square_pad{pad:.2f}".replace("0.15", "0.15"), square_crop(rgb, box, pad))

    total = len(items)
    print(f"images={total} hand_detected={detected} ({100 * detected / total:.1f}%)")
    print(f"{'strategy':28s} {'n':>5s} {'accuracy':>9s}")
    for s in strategies:
        n = attempted[s]
        print(f"{s:28s} {n:5d} {100 * correct[s] / max(n, 1):8.1f}%")
    print("\nper-letter accuracy (correct/n):")
    letters = sorted({t for _, t in items})
    print("letter " + " ".join(f"{s[:14]:>14s}" for s in strategies))
    for letter in letters:
        cells = []
        for s in strategies:
            c, n = per_letter[s][letter]
            cells.append(f"{c:>6d}/{n:<7d}")
        print(f"{letter:6s} " + " ".join(cells))

    if args.scene:
        run_scene_simulation(items, predictor, detector)


def make_scene(photo: Image.Image, rng: np.random.Generator) -> Image.Image:
    """960x720 room (smooth colour field + noise) with the photo pasted at 25-55% of frame height.
    A synthetic stand-in for a webcam frame; it is NOT a substitute for testing with a real camera."""
    w, h = 960, 720
    base = rng.uniform(70, 200, size=3)
    gradient = np.linspace(-25, 25, w)[None, :, None] + np.linspace(-15, 15, h)[:, None, None]
    scene = np.clip(base + gradient + rng.normal(0, 6, size=(h, w, 3)), 0, 255).astype(np.uint8)
    side = int(h * rng.uniform(0.25, 0.55))
    x = int(rng.integers(0, w - side))
    y = int(rng.integers(0, h - side))
    scene[y : y + side, x : x + side] = np.array(photo.convert("RGB").resize((side, side)))
    return Image.fromarray(scene)


def run_scene_simulation(items, predictor, detector) -> None:
    rng = np.random.default_rng(7)
    names = ["current_tight_stretched", "square_pad0.30"]
    detected = 0
    ok = {n: 0 for n in names}
    for path, truth in items:
        with Image.open(path) as im:
            scene = make_scene(im, rng)
        rgb = np.array(scene)
        box = detector.landmark_box(rgb)
        if box is None:
            continue
        detected += 1
        ok["current_tight_stretched"] += predictor.predict_pil(legacy_tight_stretched(rgb, box))["prediction"] == truth
        ok["square_pad0.30"] += predictor.predict_pil(square_crop(rgb, box, 0.30))["prediction"] == truth
    total = len(items)
    print(f"\n[scene simulation, synthetic] images={total} hand_detected={detected} ({100 * detected / total:.1f}%)")
    for n in names:
        print(f"{n:28s} accuracy on detected: {100 * ok[n] / max(detected, 1):5.1f}%  (of all images: {100 * ok[n] / total:5.1f}%)")


if __name__ == "__main__":
    main()
