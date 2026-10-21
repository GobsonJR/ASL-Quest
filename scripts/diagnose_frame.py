"""Diagnose a SAVED webcam frame: what does the classifier see, and what does each framing predict?

For every frame this prints the hand box (and whether it touches the frame edge), then the top-5
predictions for: the exact live pipeline, the whole frame, and square crops with different padding,
each also mirrored. Crops are saved so you can look at exactly what the model was given.

Get a frame by running the app in development (npm run dev), opening Practice, holding the sign and
pressing "Save frame" in the Technical details panel. Name it with the letter you were signing,
e.g. frame_C_1760000000000.jpg (the button does this), then:

    venv311\\Scripts\\python.exe scripts\\diagnose_frame.py path\\to\\frame_C_*.jpg

Needs the backend running (it classifies through POST /predict, so no model is loaded here).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import OUTPUTS_DIR  # noqa: E402
from src.inference.hands import HandDetector, crop_square, square_window  # noqa: E402

CRLF = bytes([13, 10])


def classify(base_url: str, image: Image.Image, require_hand: bool = False) -> dict:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "PNG")  # lossless: the server sees exactly these pixels
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}".encode()
        + CRLF
        + b'Content-Disposition: form-data; name="file"; filename="frame.png"'
        + CRLF
        + b"Content-Type: image/png"
        + CRLF
        + CRLF
    )
    body = head + buf.getvalue() + CRLF + f"--{boundary}--".encode() + CRLF
    url = f"{base_url}/predict?require_hand={'true' if require_hand else 'false'}&top_k=5"
    request = urllib.request.Request(url, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(request, timeout=60))


def fmt(reply: dict) -> str:
    if reply.get("status") not in (None, "ok"):
        return f"<{reply['status']}>"
    return "  ".join(f"{t['label']}:{t['confidence']:.2f}" for t in reply["top_predictions"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="saved frame(s) or folder(s)")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--expect", default=None, help="the letter being signed (default: parsed from the filename)")
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "diagnostics" / "frames")
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        files.extend(sorted(p.glob("*.jp*g")) + sorted(p.glob("*.png")) if p.is_dir() else [p])
    detector = HandDetector(static_image_mode=True)
    print(f"hand detector available: {detector.available}   server: {args.server}")

    for path in files:
        expected = (args.expect or (re.search(r"frame_([A-Za-z])_", path.name) or [None, None])[1] or "").upper()
        frame = Image.open(path).convert("RGB")
        rgb = np.array(frame)
        h, w = rgb.shape[:2]
        print(f"\n=== {path.name}  frame {w}x{h}  expected letter: {expected or '?'} ===")
        box = detector.landmark_box(rgb)
        out_dir = args.out / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        variants: list[tuple[str, Image.Image, bool]] = [("whole frame (no crop)", frame, False)]
        if box is None:
            print("hand: NOT DETECTED by MediaPipe on this frame (the live app would report 'No hand').")
        else:
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            edge = [n for n, hit in (("left", x1 < 2), ("top", y1 < 2), ("right", x2 > w - 2), ("bottom", y2 > h - 2)) if hit]
            print(f"hand box: x {x1:.0f}-{x2:.0f}, y {y1:.0f}-{y2:.0f}  ({bw:.0f}x{bh:.0f}px = {100 * bh / h:.0f}% of frame height)"
                  + (f"  TOUCHES FRAME EDGE: {edge} (fingers/thumb may be cut off)" if edge else ""))
            if min(bw, bh) < 60:
                print("note: the hand is small in the frame (<60px); move closer for a sharper crop.")
            for pad in (0.30, 0.60, 1.00):
                variants.append((f"square crop pad {pad:.2f}", crop_square(rgb, square_window(box, pad)), True))
            try:
                live = classify(args.server, frame, require_hand=True)
                print(f"LIVE pipeline (require_hand=true, pad 0.30): {fmt(live)}   [{live.get('status')}]")
            except Exception as exc:  # noqa: BLE001
                print(f"LIVE pipeline request failed: {exc}")
        print(f"{'variant':26s} {'input':>9s}  top-5")
        for name, image, is_crop in variants:
            for mirrored in (False, True):
                img = ImageOps.mirror(image) if mirrored else image
                label = name + (" + MIRRORED" if mirrored else "")
                reply = classify(args.server, img)
                top = reply["top_predictions"][0]["label"] if reply.get("top_predictions") else None
                mark = "  <== correct" if expected and top == expected else ""
                print(f"{label:26s} {img.width:4d}x{img.height:<4d}  {fmt(reply)}{mark}")
                slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
                img.save(out_dir / f"{slug}.png")
        print(f"crops saved to {out_dir}")


if __name__ == "__main__":
    main()
