"""Controlled experiments on ONE letter's held-out images (default C), using the real checkpoint.

Every comparison uses the *same* images, so differences come from the preprocessing/perturbation,
not from different samples.

  A  standard evaluation pipeline (whole 200x200 photo -> resize 224 -> normalise)
  B  live pipeline: MediaPipe hand box -> square crop (+30% pad) -> same transform
  C  alternative crops of the same detected hands (pad 0.6, pad 1.0)
  E  the same pipelines on visually similar letters
  P  perturbation sensitivity (mirror, rotation, zoom-out) for this letter vs every other letter

Torch and MediaPipe are never loaded in the same process (together they exceed the memory of small
machines), so the work is split into two stages that talk through a folder of saved crops:

    venv311\\Scripts\\python.exe scripts\\diagnose_c_experiments.py --stage prepare --letter C
    venv311\\Scripts\\python.exe scripts\\diagnose_c_experiments.py --stage classify --letter C --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BEST_MODEL_PATH, DATASET_DIR, OUTPUTS_DIR, SPLITS_PATH  # noqa: E402

WORK = OUTPUTS_DIR / "diagnostics" / "c_experiments"
PADS = (0.30, 0.60, 1.00)
LETTERS = [chr(ord("A") + i) for i in range(26)]


def test_images(letter: str, limit: int | None) -> list[Path]:
    """Held-out (test split) image paths for one letter, in dataset order (torch-free)."""
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))["splits"]["test"]
    samples: list[Path] = []
    for cls in sorted(os.listdir(DATASET_DIR)):
        samples.extend(DATASET_DIR / cls / f for f in sorted(os.listdir(DATASET_DIR / cls)))
    paths = [samples[i] for i in splits]
    mine = [p for p in paths if p.parent.name == letter]
    assert all(p.name.upper().startswith(letter) for p in mine), "split indices do not line up with files"
    if limit:
        mine = mine[:: max(1, len(mine) // limit)][:limit]
    return mine


# --------------------------------------------------------------------------- stage 1 (no torch)
def stage_prepare(args) -> None:
    from src.inference.hands import HandDetector, crop_square, square_window

    detector = HandDetector(static_image_mode=True)
    print(f"hand_detector available={detector.available}")
    summary = {}
    for name in [args.letter] + args.similar:
        paths = test_images(name, None if name == args.letter else 150)
        detected = []
        for path in paths:
            rgb = np.array(Image.open(path).convert("RGB"))
            box = detector.landmark_box(rgb)
            if box is None:
                continue
            detected.append(path.name)
            for pad in PADS:
                out = WORK / name / f"pad{pad:.2f}"
                out.mkdir(parents=True, exist_ok=True)
                crop_square(rgb, square_window(box, pad)).save(out / (path.stem + ".png"))
        summary[name] = {"n": len(paths), "detected": detected}
        print(f"{name}: {len(paths)} images, hand detected in {len(detected)}")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "prepare.json").write_text(json.dumps(summary), encoding="utf-8")


# --------------------------------------------------------------------------- stage 2 (no mediapipe)
def zoom_out(im: Image.Image, scale: float) -> Image.Image:
    """Shrink the photo to `scale` of its size, centred on an edge-replicated canvas (hand appears smaller)."""
    w, h = im.size
    small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    arr = np.array(small.convert("RGB"))
    pad_x, pad_y = (w - arr.shape[1]) // 2, (h - arr.shape[0]) // 2
    arr = np.pad(arr, ((pad_y, h - arr.shape[0] - pad_y), (pad_x, w - arr.shape[1] - pad_x), (0, 0)), mode="edge")
    return Image.fromarray(arr)


def jpeg(im: Image.Image, quality: int) -> Image.Image:
    """Round-trip through JPEG at the quality the browser uses for live frames (0.72)."""
    import io

    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def enhance(im: Image.Image, kind: str, factor: float) -> Image.Image:
    from PIL import ImageEnhance

    return getattr(ImageEnhance, kind)(im.convert("RGB")).enhance(factor)


def noise(im: Image.Image, sigma: float = 14.0) -> Image.Image:
    arr = np.array(im.convert("RGB")).astype(np.float32)
    arr += np.random.default_rng(0).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def zoom_in(im: Image.Image, scale: float) -> Image.Image:
    """Centre-crop to 1/scale of the frame then resize back (hand appears larger)."""
    w, h = im.size
    cw, ch = int(w / scale), int(h / scale)
    left, top = (w - cw) // 2, (h - ch) // 2
    return im.crop((left, top, left + cw, top + ch)).resize((w, h))


PERTURBATIONS = {
    "identity": lambda im: im,
    "mirror": ImageOps.mirror,
    "rot+20": lambda im: im.rotate(20, resample=Image.BILINEAR),
    "rot-20": lambda im: im.rotate(-20, resample=Image.BILINEAR),
    "rot+45": lambda im: im.rotate(45, resample=Image.BILINEAR),
    "zoom0.6": lambda im: zoom_out(im, 0.6),
    "zoom0.4": lambda im: zoom_out(im, 0.4),
    "zoomin1.3": lambda im: zoom_in(im, 1.3),
    "jpeg72": lambda im: jpeg(im, 72),
    "dark0.6": lambda im: enhance(im, "Brightness", 0.6),
    "bright1.6": lambda im: enhance(im, "Brightness", 1.6),
    "lowcontrast": lambda im: enhance(im, "Contrast", 0.5),
    "blur": lambda im: im.filter(ImageFilter.GaussianBlur(2.5)),
    "noise": noise,
}


def server_classifier(base_url: str):
    """Classify through a running backend's POST /predict (whole image, no hand detection).

    Uses the server's own loaded checkpoint and preprocessing, so this also verifies the deployed
    pipeline. Needs no torch in this process. The server returns its top-3 predictions.
    """
    import io
    import urllib.request
    import uuid

    def top5(images):
        out = []
        for im in images:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "PNG")  # lossless, so the server sees exactly these pixels
            boundary = uuid.uuid4().hex
            crlf = b'\r\n'
            head = (
                f'--{boundary}'.encode() + crlf
                + b'Content-Disposition: form-data; name="file"; filename="x.png"' + crlf
                + b'Content-Type: image/png' + crlf + crlf
            )
            body = head + buf.getvalue() + crlf + f'--{boundary}--'.encode() + crlf
            req = urllib.request.Request(
                f"{base_url}/predict?require_hand=false", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}
            )
            reply = json.load(urllib.request.urlopen(req, timeout=60))
            out.append([(t["label"], t["confidence"]) for t in reply["top_predictions"]])
        return out

    return top5


def local_classifier(args):
    import torch

    from src.data.preprocessing import eval_transforms
    from src.models.resnet18 import build_resnet18
    from src.utils.checkpoint import load_checkpoint

    transform = eval_transforms()
    payload = load_checkpoint(args.model, map_location=args.device)
    classes = payload["class_names"]
    net = build_resnet18(num_classes=len(classes), pretrained=False)
    net.load_state_dict(payload["model_state_dict"])
    net.to(args.device).eval()

    @torch.no_grad()
    def top5(images):
        out = []
        for s in range(0, len(images), 16):
            batch = torch.stack([transform(im.convert("RGB")) for im in images[s : s + 16]]).to(args.device)
            conf, idx = torch.softmax(net(batch), dim=1).cpu().topk(5, dim=1)
            for c, i in zip(conf.tolist(), idx.tolist()):
                out.append([(classes[j], v) for j, v in zip(i, c)])
        return out

    return top5


def stage_classify(args) -> None:
    top5 = server_classifier(args.server.rstrip("/")) if args.server else local_classifier(args)
    letter = args.letter
    prep = json.loads((WORK / "prepare.json").read_text(encoding="utf-8")) if not args.perturb_only else {}
    print(f"classifier: {'server ' + args.server if args.server else 'local ' + args.model.name + ' on ' + args.device}", flush=True)

    # ---- A/B/C/E: same images through each pipeline ----
    for name in ([] if args.perturb_only else [letter] + args.similar):
        paths = test_images(name, None if name == letter else 150)
        full = top5([Image.open(p) for p in paths])
        full_by_name = {p.name: r for p, r in zip(paths, full)}
        det = prep[name]["detected"]
        print(f"\n=== {name}: {len(paths)} held-out images (200x200); hand detected in {len(det)} ({100 * len(det) / len(paths):.0f}%) ===")
        print(f"A standard pipeline, ALL images       : top-1 {100 * np.mean([r[0][0] == name for r in full]):5.1f}%  mean conf {np.mean([r[0][1] for r in full]):.2f}")
        if not det:
            continue
        print(f"A standard pipeline, detected subset  : top-1 {100 * np.mean([full_by_name[d][0][0] == name for d in det]):5.1f}%  (n={len(det)})")
        results = {}
        for pad in PADS:
            files = [WORK / name / f"pad{pad:.2f}" / (Path(d).stem + ".png") for d in det]
            res = top5([Image.open(f) for f in files])
            results[pad] = res
            wrong = Counter(r[0][0] for r in res if r[0][0] != name).most_common(3)
            print(f"B/C live crop pad {pad:.2f}            : top-1 {100 * np.mean([r[0][0] == name for r in res]):5.1f}%  mean conf {np.mean([r[0][1] for r in res]):.2f}  top-5 hit {100 * np.mean([name in [l for l, _ in r] for r in res]):5.1f}%  wrong->{wrong}")
        if name == letter:
            fmt = lambda r: " ".join(f"{l}:{c:.2f}" for l, c in r[:3])  # noqa: E731
            print(f"\nsame-image comparison, first {args.examples} detected {letter} images (top-3):")
            for pos, d in enumerate(det[: args.examples]):
                print(f"  {d:10s} standard[{fmt(full_by_name[d])}]  live0.30[{fmt(results[0.30][pos])}]  pad1.00[{fmt(results[1.00][pos])}]")

    # ---- P: perturbation sensitivity, this letter vs every class ----
    print(f"\n=== P: perturbation sensitivity (standard pipeline, {args.per_class_perturb} held-out images per class) ===")
    table: dict[str, dict[str, float]] = {}
    mirrored_as: dict[str, Counter] = {}
    for c_name in LETTERS:
        images = [Image.open(p).convert("RGB") for p in test_images(c_name, args.per_class_perturb)]
        table[c_name] = {}
        for p_name, fn in PERTURBATIONS.items():
            res = top5([fn(im) for im in images])
            table[c_name][p_name] = 100 * float(np.mean([r[0][0] == c_name for r in res]))
            if p_name == "mirror":
                mirrored_as[c_name] = Counter(r[0][0] for r in res)
    print(f"{'cls':3s} " + " ".join(f"{p[:9]:>9s}" for p in PERTURBATIONS))
    for c_name in [letter] + args.similar:
        print(f"{c_name:3s} " + " ".join(f"{table[c_name][p]:8.0f}%" for p in PERTURBATIONS))
    print("ALL " + " ".join(f"{np.mean([table[c][p] for c in LETTERS]):8.0f}%" for p in PERTURBATIONS) + "   (mean over 26 letters)")
    print(f"\n{letter} when mirrored is predicted as: {mirrored_as[letter].most_common(5)}")
    worst = sorted(LETTERS, key=lambda c: table[c]["mirror"])[:8]
    print("least robust to mirroring:", [(c, round(table[c]["mirror"])) for c in worst])
    (WORK / "perturbation_table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "classify"], required=True)
    ap.add_argument("--letter", default="C")
    ap.add_argument("--similar", nargs="*", default=["W", "O", "G"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--server", default=None, help="e.g. http://127.0.0.1:8000 : classify through a running backend instead of loading torch here")
    ap.add_argument("--model", type=Path, default=BEST_MODEL_PATH)
    ap.add_argument("--per-class-perturb", type=int, default=30)
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--perturb-only", action="store_true", help="skip the A/B/C/E crop comparison")
    args = ap.parse_args()
    args.letter = args.letter.upper()
    (stage_prepare if args.stage == "prepare" else stage_classify)(args)


if __name__ == "__main__":
    main()
