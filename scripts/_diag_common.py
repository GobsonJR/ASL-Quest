"""Shared helpers for the diagnostics in scripts/diag_*.py (no torch: the model is queried over HTTP, MediaPipe runs locally).

The backend must be running (venv311\\Scripts\\python.exe -m uvicorn backend.main:app --port 8000 --env-file .env).
Everything here is read-only with respect to checkpoints and the collected dataset.
"""

from __future__ import annotations

import io
import json
import sys
import uuid
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LETTERS = [chr(65 + i) for i in range(26)]
SERVER = "http://127.0.0.1:8000"
OUT = ROOT / "outputs" / "diagnostics"
WEBCAM = ROOT / "dataset" / "webcam"
EXPORT_ALL = ROOT / "dataset" / "webcam_export_alldiag"  # every collected frame, crops made by the live crop function


def predict(image: Image.Image | bytes, *, require_hand: bool = False, fmt: str = "PNG", quality: int = 92, top_k: int = 3) -> dict:
    """POST one image to the running backend's /predict (the app's real inference path)."""
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, fmt, **({"quality": quality} if fmt == "JPEG" else {}))
        data, name, ctype = buf.getvalue(), f"x.{fmt.lower()}", f"image/{fmt.lower()}"
    else:
        data, name, ctype = image, "x.jpg", "image/jpeg"
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: {ctype}\r\n\r\n').encode() + data + f"\r\n--{boundary}--\r\n".encode()
    url = f"{SERVER}/predict?require_hand={'true' if require_hand else 'false'}&top_k={top_k}"
    request = urllib.request.Request(url, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(request, timeout=120).read())


def top1(reply: dict) -> tuple[str | None, float]:
    top = reply.get("top_predictions") or []
    return (top[0]["label"], float(top[0]["confidence"])) if top else (None, 0.0)


def prob_of(reply: dict, letter: str) -> float:
    """Confidence the model gave `letter` if it is among the returned top-k, else 0 (use top_k=26 when this matters)."""
    for item in reply.get("top_predictions") or []:
        if item["label"] == letter:
            return float(item["confidence"])
    return 0.0


def thin_hold(rows: list[dict], keep: int = 3) -> list[dict]:
    """Singles + a few evenly spaced frames per hold sequence (same rule as scripts/evaluate_webcam.py)."""
    out, seqs = [], defaultdict(list)
    for r in rows:
        (seqs[r["sequence_id"]] if r["kind"] == "hold" else out).append(r)
    for seq in seqs.values():
        seq.sort(key=lambda r: r.get("frame_index") or 0)
        idx = sorted(set(np.linspace(0, len(seq) - 1, min(keep, len(seq))).round().astype(int).tolist()))
        out.extend(seq[i] for i in idx)
    return out


def load_webcam_rows(thin: bool = True) -> list[dict]:
    """Manifest rows of every collected frame, each with the original frame path (`frame_path`) and its saved crop (`crop_path`)."""
    rows = [json.loads(line) for line in (EXPORT_ALL / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    images = {}
    for meta in WEBCAM.glob("s*/*/*.json"):
        rec = json.loads(meta.read_text(encoding="utf-8"))
        images[rec["sample_id"]] = meta.parent / rec["image"]
    for r in rows:
        r["frame_path"] = images[r["sample_id"]]
        r["crop_path"] = EXPORT_ALL / r["crop"]
    return thin_hold(rows) if thin else rows


def make_detector():
    from backend.settings import HAND_DETECTION_CONFIDENCE
    from src.inference.hands import HandDetector

    detector = HandDetector(static_image_mode=True, detection_confidence=HAND_DETECTION_CONFIDENCE)  # same as the live backend
    if not detector.available:
        sys.exit(f"hand detector unavailable: {detector.error}")
    return detector


def sheet(images: list[Image.Image], path: Path, cols: int = 8, size: int = 150, labels: list[str] | None = None) -> None:
    from PIL import ImageDraw

    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * size, rows * (size + (14 if labels else 0))), "white")
    draw = ImageDraw.Draw(canvas)
    for i, im in enumerate(images):
        x, y = (i % cols) * size, (i // cols) * (size + (14 if labels else 0))
        canvas.paste(im.convert("RGB").resize((size, size)), (x, y))
        if labels:
            draw.text((x + 2, y + size), labels[i][:26], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
