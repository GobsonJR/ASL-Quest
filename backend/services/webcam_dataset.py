"""Labelled webcam data collection (development only).

Storage layout (``dataset/webcam`` is git-ignored: it contains your face and room):

    dataset/webcam/<session_id>/session.json
    dataset/webcam/<session_id>/<LETTER>/<sample_id>.jpg|png     the frame exactly as received
    dataset/webcam/<session_id>/<LETTER>/<sample_id>.json        label + metadata + quality metrics

Rules enforced here (not just in the UI):
  * the label is whatever the person confirmed -- it is NEVER taken from a model prediction;
  * a sample is saved only with ``confirmed=True`` and a single letter A-Z;
  * frames the classifier could not use anyway are rejected (no hand, blurry, too small, too dark/bright);
  * near-identical consecutive single frames are rejected so the data is varied, not hundreds of copies
    (frames of a deliberate "hold" sequence are exempt: they exist to measure prediction stability).
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

LETTERS = [chr(65 + i) for i in range(26)]
LIGHTING = ("daylight", "lamp", "dim", "backlit", "mixed", "bright_indoor")
BACKGROUNDS = ("plain_wall", "shelves", "curtains", "patterned", "furniture", "window", "people", "other")
HANDS = ("left", "right")
DISTANCES = ("near", "medium", "far")
MODES = ("single", "hold")

# Calibrated with THIS module's sharpness() on 60 real captured crops (not a different resize): motion-blurred crops
# measured 82-100 and skin close-ups 33-38, while the sharpest-of-the-rest started at 126 and the median was 243.
MIN_SHARPNESS = 110.0
MIN_HAND_PX = 100          # longest side of the detected hand box in the original frame
MIN_BRIGHTNESS = 45.0      # mean grey level of the whole frame
MAX_BRIGHTNESS = 225.0
NEAR_DUPLICATE_HAMMING = 4  # dHash bits (of 64) below which two frames count as the same picture
DUPLICATE_WINDOW = 8        # compare against the last N single frames of the same letter in the session
DEFAULT_TARGET_PER_LETTER = 150

_lock = threading.Lock()


class CollectError(ValueError):
    """A request the collector refuses (bad letter, missing confirmation, unknown session...)."""

    def __init__(self, message: str, status: int = 422, reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.reasons = reasons or []


@dataclass
class QualityReport:
    usable: bool
    reasons: list[str] = field(default_factory=list)   # why it was rejected
    warnings: list[str] = field(default_factory=list)  # usable, but worth knowing
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"usable": self.usable, "reasons": self.reasons, "warnings": self.warnings, "metrics": self.metrics}


def sharpness(image: Image.Image) -> float:
    """Variance of the Laplacian of the 224x224 grayscale crop (higher = sharper)."""
    gray = np.array(image.convert("L").resize((224, 224), Image.Resampling.LANCZOS))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(image: Image.Image) -> int:
    """64-bit difference hash: robust to small changes, so near-identical frames collide."""
    small = np.array(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def analyze_frame(image: Image.Image, detector: Any) -> tuple[QualityReport, Any]:
    """Quality-check one frame the way the live pipeline would see it. Returns (report, HandCrop|None)."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    gray_mean = float(np.array(rgb.convert("L")).mean())
    metrics: dict[str, Any] = {"frame_size": [width, height], "brightness": round(gray_mean, 1)}
    reasons: list[str] = []
    warnings: list[str] = []

    if detector is None or not getattr(detector, "available", False):
        return QualityReport(False, ["hand detector unavailable on the server"], [], metrics), None
    crop = detector.crop_pil(rgb)
    if not crop.detected:
        reasons.append("no hand detected")
        if gray_mean < MIN_BRIGHTNESS:
            reasons.append("too dark")
        return QualityReport(False, reasons, warnings, metrics), None

    x1, y1, x2, y2 = crop.hand_box or (0, 0, 0, 0)
    hand_px = max(x2 - x1, y2 - y1)
    sharp = sharpness(crop.image)
    metrics.update(
        hand_px=round(hand_px, 1),
        hand_height_fraction=round((y2 - y1) / height, 3),
        sharpness=round(sharp, 1),
        crop_size=list(crop.image.size),
        landmark_box=[round(v, 1) for v in (x1, y1, x2, y2)],
    )
    if sharp < MIN_SHARPNESS:
        reasons.append(f"blurry (sharpness {sharp:.0f} < {MIN_SHARPNESS:.0f}): hold still or add light")
    if hand_px < MIN_HAND_PX:
        reasons.append(f"hand too small ({hand_px:.0f}px < {MIN_HAND_PX}px): move closer")
    if gray_mean < MIN_BRIGHTNESS:
        reasons.append("too dark")
    if gray_mean > MAX_BRIGHTNESS:
        reasons.append("too bright / overexposed")
    if x1 < 2 or y1 < 2 or x2 > width - 2 or y2 > height - 2:
        warnings.append("hand touches the frame edge (fingers may be cut off)")
    return QualityReport(not reasons, reasons, warnings, metrics), crop


class WebcamDataset:
    def __init__(self, root: Path) -> None:
        self.root = root

    # -------------------------------------------------------------- sessions
    def create_session(self, meta: dict[str, Any]) -> dict[str, Any]:
        lighting, background = meta.get("lighting"), meta.get("background")
        if lighting not in LIGHTING:
            raise CollectError(f"lighting must be one of {LIGHTING}")
        if background not in BACKGROUNDS:
            raise CollectError(f"background must be one of {BACKGROUNDS}")
        room = re.sub(r"[^a-z0-9]+", "-", str(meta.get("room", "room")).lower()).strip("-")[:24] or "room"
        session_id = f"s{datetime.now().strftime('%Y%m%d_%H%M%S')}_{room}"
        record = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "room": str(meta.get("room", ""))[:80],
            "lighting": lighting,
            "background": background,
            "background_notes": str(meta.get("background_notes", ""))[:200],
            "face_visible": bool(meta.get("face_visible", True)),
            "glasses": bool(meta.get("glasses", False)),
            "camera": str(meta.get("camera", ""))[:80],
            "notes": str(meta.get("notes", ""))[:300],
        }
        with _lock:
            folder = self.root / session_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "session.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def session(self, session_id: str) -> dict[str, Any]:
        path = self.root / self._safe_id(session_id) / "session.json"
        if not path.exists():
            raise CollectError("unknown session", status=404)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _safe_id(session_id: str) -> str:
        if not re.fullmatch(r"s\d{8}_\d{6}_[a-z0-9-]*", session_id):
            raise CollectError("invalid session id", status=404)
        return session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        out = []
        for folder in sorted(p for p in self.root.iterdir() if (p / "session.json").exists()):
            summary = self.summary(folder.name)
            out.append(summary)
        return out

    def samples(self, session_id: str) -> list[dict[str, Any]]:
        folder = self.root / self._safe_id(session_id)
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*/*.json"))]

    def summary(self, session_id: str) -> dict[str, Any]:
        session = self.session(session_id)
        counts = {letter: 0 for letter in LETTERS}
        for sample in self.samples(session_id):
            counts[sample["letter"]] += 1
        return {**session, "counts": counts, "total": sum(counts.values())}

    def overall_counts(self) -> dict[str, int]:
        counts = {letter: 0 for letter in LETTERS}
        for session in self.list_sessions():
            for letter, n in session["counts"].items():
                counts[letter] += n
        return counts

    # -------------------------------------------------------------- samples
    def save_sample(
        self,
        *,
        session_id: str,
        letter: str,
        confirmed: bool,
        payload: bytes,
        image: Image.Image,
        report: QualityReport,
        hand: str,
        distance: str,
        mode: str = "single",
        sequence_id: str | None = None,
        frame_index: int | None = None,
        video_size: str | None = None,
    ) -> dict[str, Any]:
        if not confirmed:
            raise CollectError("the letter must be confirmed by the person before saving", status=422)
        letter = (letter or "").strip().upper()
        if letter not in LETTERS:
            raise CollectError("letter must be a single letter A-Z")
        if hand not in HANDS:
            raise CollectError(f"hand must be one of {HANDS}")
        if distance not in DISTANCES:
            raise CollectError(f"distance must be one of {DISTANCES}")
        if mode not in MODES:
            raise CollectError(f"mode must be one of {MODES}")
        if mode == "hold" and not sequence_id:
            raise CollectError("hold frames need a sequence_id")
        session = self.session(session_id)
        if not report.usable:
            self._log_rejection(session_id, letter, "quality", report.reasons)
            raise CollectError("frame rejected by the quality check", reasons=report.reasons)

        fingerprint = dhash(image)
        if mode == "single":
            recent = [
                s for s in self.samples(session_id) if s["letter"] == letter and s["mode"] == "single"
            ][-DUPLICATE_WINDOW:]
            for previous in recent:
                if hamming(fingerprint, int(previous["dhash"], 16)) <= NEAR_DUPLICATE_HAMMING:
                    self._log_rejection(session_id, letter, "near-duplicate", ["near-duplicate"])
                    raise CollectError(
                        "too similar to a recent frame of this letter: change your hand position, distance or angle",
                        reasons=["near-duplicate"],
                    )

        sample_id = f"{letter}_{int(time.time() * 1000)}_{secrets.token_hex(2)}"
        ext = "jpg" if (image.format or "").upper() in ("JPEG", "MPO") else "png"
        folder = self.root / session_id / letter
        record = {
            "sample_id": sample_id,
            "session_id": session_id,
            "letter": letter,
            "label_source": "person_confirmed",  # never a model prediction
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "image": f"{sample_id}.{ext}",
            "hand": hand,
            "distance": distance,
            "mode": mode,
            "sequence_id": sequence_id,
            "frame_index": frame_index,
            "video_size": video_size,
            "lighting": session["lighting"],
            "background": session["background"],
            "face_visible": session["face_visible"],
            "glasses": session["glasses"],
            "quality": report.as_dict(),
            "dhash": f"{fingerprint:016x}",
        }
        with _lock:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / record["image"]).write_bytes(payload)  # the frame exactly as received
            (folder / f"{sample_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def _log_rejection(self, session_id: str, letter: str, kind: str, reasons: list[str]) -> None:
        """Append-only count of refused save attempts (no image kept), so the pre-training audit can report them."""
        entry = {"at": datetime.now(timezone.utc).isoformat(), "letter": letter, "kind": kind, "reasons": reasons}
        with _lock:
            with (self.root / session_id / "rejections.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

    def delete_sample(self, session_id: str, sample_id: str) -> None:
        if not re.fullmatch(r"[A-Z]_\d+_[0-9a-f]{4}", sample_id):
            raise CollectError("invalid sample id", status=404)
        folder = self.root / self._safe_id(session_id) / sample_id[0]
        meta = folder / f"{sample_id}.json"
        if not meta.exists():
            raise CollectError("unknown sample", status=404)
        image_name = json.loads(meta.read_text(encoding="utf-8"))["image"]
        with _lock:
            (folder / image_name).unlink(missing_ok=True)
            meta.unlink(missing_ok=True)

    def delete_session(self, session_id: str) -> None:
        folder = self.root / self._safe_id(session_id)
        if folder.exists():
            shutil.rmtree(folder)


def open_image(payload: bytes) -> Image.Image:
    image = Image.open(BytesIO(payload))
    image.load()
    return image
