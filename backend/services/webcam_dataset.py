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
# Frames the recognizer must NOT answer on: no hand in view, or a hand cut by the frame edge. They are stored under
# <session>/negatives/<kind>/ (outside the <LETTER>/ folders, so export/audit/training never see them) and are used only to
# measure false detections and partial-hand behaviour in scripts/evaluate_webcam_heldout.py.
NEGATIVE_KINDS = ("no_hand", "partial_hand")

# Calibrated with THIS module's sharpness() on 60 real captured crops (not a different resize): motion-blurred crops
# measured 82-100 and skin close-ups 33-38, while the sharpest-of-the-rest started at 126 and the median was 243.
MIN_SHARPNESS = 110.0
MIN_HAND_PX = 100          # longest side of the detected hand box in the original frame
MIN_BRIGHTNESS = 45.0      # mean grey level of the whole frame
MAX_BRIGHTNESS = 225.0
NEAR_DUPLICATE_HAMMING = 4  # dHash bits (of 64) below which two frames count as the same picture
DUPLICATE_WINDOW = 8        # compare against the last N single frames of the same letter in the session
DEFAULT_TARGET_PER_LETTER = 150
DEFAULT_TARGET_PER_NATIVE_SIGN = 30

# Upload limits, enforced on the server (the UI is not trusted). Images are JPEG or PNG only, because the saved extension
# is derived from the decoded format and everything downstream (export, evaluation scripts) expects those two.
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 64
ALLOWED_IMAGE_FORMATS = ("JPEG", "MPO", "PNG")
# Native-sign clips are short webcam recordings (WebM from MediaRecorder; MP4 from Safari).
MAX_CLIP_BYTES = 60 * 1024 * 1024
ALLOWED_CLIP_TYPES = ("webm", "mp4")

_lock = threading.Lock()

EXPORT_README = """ASL Quest labelled dataset export
Exported at: {exported_at}
Sessions: {sessions} (skipped because they have no recorded consent: {skipped})
Files: {files}
Per label: {labels}

LAYOUT
  manifest.csv / manifest.jsonl   one row per file (label, label_type, session, capture time, hand, distance, ...)
  images/<LETTER>/                alphabet frames, exactly as captured (JPEG or PNG)
  native_clips/<GLOSS>/           recorded native-sign clips (WebM or MP4)
  negatives/<kind>/               frames the person confirmed show no usable hand (label starts with _)

LABELS  Every label was confirmed by the person recording; no model prediction was used to label anything.

PRIVACY  These files can show identifiable people and their rooms. Use them only as the contributors agreed
(consent was recorded per session, see consent_recorded_at), do not publish them, and delete them on request.

NOT TRAINING  This export is only a labelled collection. Splitting, training, evaluation and promoting a model are
separate steps (see WEBCAM_EVAL_PROTOCOL.md); keep whole sessions and whole people on one side of any split."""



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
        if meta.get("consent_confirmed") is not True:
            raise CollectError(
                "the person being recorded must consent first: images and clips can show an identifiable face and room",
                status=422,
            )
        room = re.sub(r"[^a-z0-9]+", "-", str(meta.get("room", "room")).lower()).strip("-")[:24] or "room"
        session_id = f"s{datetime.now().strftime('%Y%m%d_%H%M%S')}_{room}"
        record = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            # Who agreed, and when. The admin's username is recorded for accountability; no token or password is ever stored.
            "consent": {
                "confirmed": True,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "recorded_by": str(meta.get("recorded_by", ""))[:64],
            },
            "room": str(meta.get("room", ""))[:80],
            "person": re.sub(r"[^a-z0-9]+", "-", str(meta.get("person", "")).lower()).strip("-")[:32],
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

    def writable_session(self, session_id: str) -> dict[str, Any]:
        """The session, but only if it carries a consent record: nothing new is saved into a session without one."""
        session = self.session(session_id)
        if not (session.get("consent") or {}).get("confirmed"):
            raise CollectError(
                "this session has no recorded contributor consent; start a new session and confirm consent first",
                status=409,
            )
        return session

    def summary(self, session_id: str) -> dict[str, Any]:
        session = self.session(session_id)
        counts = {letter: 0 for letter in LETTERS}
        for sample in self.samples(session_id):
            counts[sample["letter"]] += 1
        native = self.native_counts(session_id)
        return {
            **session,
            "counts": counts,
            "total": sum(counts.values()),
            "negatives": self.negative_counts(session_id),
            "native_counts": native,
            "native_total": sum(native.values()),
        }

    def negative_counts(self, session_id: str) -> dict[str, int]:
        folder = self.root / self._safe_id(session_id) / "negatives"
        return {kind: len(list((folder / kind).glob("*.json"))) for kind in NEGATIVE_KINDS}

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
        session = self.writable_session(session_id)
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

    def save_negative(
        self,
        *,
        session_id: str,
        kind: str,
        confirmed: bool,
        payload: bytes,
        image: Image.Image,
        detector_found_hand: bool,
        touches_edge: bool,
        distance: str | None = None,
    ) -> dict[str, Any]:
        """Store a frame the person confirms has NO usable hand. The detector's verdict is recorded, never used as the label:
        a 'no_hand' frame on which MediaPipe nevertheless finds a hand is exactly the false detection we want to count."""
        if not confirmed:
            raise CollectError("the person must confirm what the frame shows before saving", status=422)
        if kind not in NEGATIVE_KINDS:
            raise CollectError(f"kind must be one of {NEGATIVE_KINDS}")
        session = self.writable_session(session_id)
        fingerprint = dhash(image)
        folder = self.root / session_id / "negatives" / kind
        recent = sorted(folder.glob("*.json"))[-DUPLICATE_WINDOW:] if folder.exists() else []
        for path in recent:
            if hamming(fingerprint, int(json.loads(path.read_text(encoding="utf-8"))["dhash"], 16)) <= NEAR_DUPLICATE_HAMMING:
                raise CollectError("too similar to a recent frame of this kind: change the scene", reasons=["near-duplicate"])
        sample_id = f"N_{kind}_{int(time.time() * 1000)}_{secrets.token_hex(2)}"
        ext = "jpg" if (image.format or "").upper() in ("JPEG", "MPO") else "png"
        record = {
            "sample_id": sample_id,
            "session_id": session_id,
            "kind": kind,
            "expected": "no_answer",
            "label_source": "person_confirmed",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "image": f"{sample_id}.{ext}",
            "frame_size": list(image.size),
            "detector_found_hand": bool(detector_found_hand),
            "touches_edge": bool(touches_edge),
            "distance": distance,
            "person": session.get("person", ""),
            "lighting": session["lighting"],
            "background": session["background"],
            "dhash": f"{fingerprint:016x}",
        }
        with _lock:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / record["image"]).write_bytes(payload)
            (folder / f"{sample_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def delete_negative(self, session_id: str, sample_id: str) -> None:
        match = re.fullmatch(r"N_(no_hand|partial_hand)_\d+_[0-9a-f]{4}", sample_id)
        if not match:
            raise CollectError("invalid sample id", status=404)
        folder = self.root / self._safe_id(session_id) / "negatives" / match.group(1)
        meta = folder / f"{sample_id}.json"
        if not meta.exists():
            raise CollectError("unknown sample", status=404)
        image_name = json.loads(meta.read_text(encoding="utf-8"))["image"]
        with _lock:
            (folder / image_name).unlink(missing_ok=True)
            meta.unlink(missing_ok=True)

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

    # -------------------------------------------------------------- gallery lookups
    def sample_image_path(self, session_id: str, sample_id: str) -> Path:
        """The stored image of a letter sample, resolved from OUR metadata (never from a client-supplied name)."""
        if not re.fullmatch(r"[A-Z]_\d+_[0-9a-f]{4}", sample_id):
            raise CollectError("invalid sample id", status=404)
        folder = self.root / self._safe_id(session_id) / sample_id[0]
        return self._stored_file(folder, sample_id)

    def negative_image_path(self, session_id: str, sample_id: str) -> Path:
        match = re.fullmatch(r"N_(no_hand|partial_hand)_\d+_[0-9a-f]{4}", sample_id)
        if not match:
            raise CollectError("invalid sample id", status=404)
        return self._stored_file(self.root / self._safe_id(session_id) / "negatives" / match.group(1), sample_id)

    @staticmethod
    def _stored_file(folder: Path, item_id: str) -> Path:
        meta = folder / f"{item_id}.json"
        if not meta.is_file():
            raise CollectError("unknown sample", status=404)
        record = json.loads(meta.read_text(encoding="utf-8"))
        name = record.get("image") or record.get("file")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+\.(jpg|png|webm|mp4)", name):
            raise CollectError("unknown sample", status=404)
        path = (folder / name).resolve()
        if path.parent != folder.resolve() or not path.is_file():
            raise CollectError("unknown sample", status=404)
        return path

    # -------------------------------------------------------------- native-sign clips
    def native_counts(self, session_id: str) -> dict[str, int]:
        folder = self.root / self._safe_id(session_id) / "native"
        if not folder.exists():
            return {}
        return {p.name: len(list(p.glob("*.json"))) for p in sorted(folder.iterdir()) if p.is_dir() and any(p.glob("*.json"))}

    def native_clips(self, session_id: str, gloss: str | None = None) -> list[dict[str, Any]]:
        folder = self.root / self._safe_id(session_id) / "native"
        if not folder.exists():
            return []
        pattern = f"{gloss}/*.json" if gloss else "*/*.json"
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob(pattern))]

    def save_native_clip(
        self,
        *,
        session_id: str,
        gloss: str,
        confirmed: bool,
        payload: bytes,
        kind: str,
        duration_ms: int | None = None,
        hand: str | None = None,
    ) -> dict[str, Any]:
        """Store one recorded clip of a native sign. `gloss` must already have been checked against the sign catalog by the
        caller; the clip's type (`kind`) is decided by sniffing its bytes, never by the client's file name or MIME type."""
        if not confirmed:
            raise CollectError("the sign must be confirmed by the person before saving", status=422)
        if not re.fullmatch(r"[A-Z0-9]{1,20}", gloss or ""):
            raise CollectError("invalid sign label")
        if kind not in ALLOWED_CLIP_TYPES:
            raise CollectError("unsupported video type: use WebM or MP4", status=415)
        if hand is not None and hand not in HANDS:
            raise CollectError(f"hand must be one of {HANDS}")
        if len(payload) > MAX_CLIP_BYTES:
            raise CollectError(f"clip is larger than {MAX_CLIP_BYTES // (1024 * 1024)} MB", status=413)
        session = self.writable_session(session_id)
        clip_id = f"C_{int(time.time() * 1000)}_{secrets.token_hex(2)}"
        folder = self.root / session_id / "native" / gloss
        record = {
            "sample_id": clip_id,
            "session_id": session_id,
            "gloss": gloss,
            "label_source": "person_confirmed",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "file": f"{clip_id}.{kind}",
            "bytes": len(payload),
            "duration_ms": duration_ms,
            "hand": hand,
            "lighting": session["lighting"],
            "background": session["background"],
            "face_visible": session["face_visible"],
            "person": session.get("person", ""),
        }
        with _lock:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / record["file"]).write_bytes(payload)
            (folder / f"{clip_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def native_clip_path(self, session_id: str, clip_id: str) -> Path:
        if not re.fullmatch(r"C_\d+_[0-9a-f]{4}", clip_id):
            raise CollectError("invalid clip id", status=404)
        base = self.root / self._safe_id(session_id) / "native"
        for gloss_folder in base.iterdir() if base.exists() else []:
            if gloss_folder.is_dir() and (gloss_folder / f"{clip_id}.json").is_file():
                return self._stored_file(gloss_folder, clip_id)
        raise CollectError("unknown clip", status=404)

    def delete_native_clip(self, session_id: str, clip_id: str) -> None:
        video = self.native_clip_path(session_id, clip_id)
        with _lock:
            video.unlink(missing_ok=True)
            video.with_suffix(".json").unlink(missing_ok=True)

    # -------------------------------------------------------------- export
    def export_zip(self, target: Path, session_ids: list[str] | None = None) -> dict[str, Any]:
        """Write a self-describing labelled dataset to `target` (a .zip).

        Layout:  README.txt, manifest.csv, manifest.jsonl,
                 images/<LETTER>/<sample_id>.<jpg|png>, native_clips/<GLOSS>/<clip_id>.<webm|mp4>,
                 negatives/<kind>/<sample_id>.<jpg|png>
        Only sessions with a recorded consent are exported. No tokens, e-mail addresses or account data are included:
        the only person information is the free-text contributor label the admin typed for the session.
        """
        import csv
        import zipfile
        from io import StringIO

        wanted = set(session_ids) if session_ids else None
        rows: list[dict[str, Any]] = []
        skipped: list[str] = []
        sessions_used: list[str] = []
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for summary in self.list_sessions():
                sid = summary["session_id"]
                if wanted is not None and sid not in wanted:
                    continue
                if not (summary.get("consent") or {}).get("confirmed"):
                    skipped.append(sid)
                    continue
                sessions_used.append(sid)
                common = {
                    "session_id": sid,
                    "contributor": summary.get("person", ""),
                    "room": summary.get("room", ""),
                    "lighting": summary.get("lighting", ""),
                    "background": summary.get("background", ""),
                    "consent_confirmed": True,
                    "consent_recorded_at": summary["consent"].get("recorded_at", ""),
                }
                for sample in self.samples(sid):
                    path = self.sample_image_path(sid, sample["sample_id"])
                    arc = f"images/{sample['letter']}/{path.name}"
                    archive.write(path, arc)
                    rows.append({**common, "file": arc, "label": sample["letter"], "label_type": "letter", "sample_id": sample["sample_id"],
                                 "captured_at": sample["captured_at"], "mode": sample.get("mode", ""), "hand": sample.get("hand", ""),
                                 "distance": sample.get("distance", "")})
                for clip in self.native_clips(sid):
                    path = self.native_clip_path(sid, clip["sample_id"])
                    arc = f"native_clips/{clip['gloss']}/{path.name}"
                    archive.write(path, arc)
                    rows.append({**common, "file": arc, "label": clip["gloss"], "label_type": "native_sign", "sample_id": clip["sample_id"],
                                 "captured_at": clip["captured_at"], "mode": "clip", "hand": clip.get("hand") or "", "distance": ""})
                neg_root = self.root / sid / "negatives"
                for kind in NEGATIVE_KINDS:
                    kind_dir = neg_root / kind
                    for meta in sorted(kind_dir.glob("*.json")) if kind_dir.exists() else []:
                        record = json.loads(meta.read_text(encoding="utf-8"))
                        path = self.negative_image_path(sid, record["sample_id"])
                        arc = f"negatives/{kind}/{path.name}"
                        archive.write(path, arc)
                        rows.append({**common, "file": arc, "label": f"_{kind}", "label_type": "negative", "sample_id": record["sample_id"],
                                     "captured_at": record["captured_at"], "mode": "single", "hand": "", "distance": record.get("distance") or ""})
            columns = ["file", "label", "label_type", "session_id", "sample_id", "captured_at", "mode", "hand", "distance", "contributor",
                       "room", "lighting", "background", "consent_confirmed", "consent_recorded_at"]
            buffer = StringIO()
            writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            archive.writestr("manifest.csv", buffer.getvalue())
            archive.writestr("manifest.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
            per_label: dict[str, int] = {}
            for row in rows:
                per_label[row["label"]] = per_label.get(row["label"], 0) + 1
            archive.writestr(
                "README.txt",
                EXPORT_README.format(
                    exported_at=datetime.now(timezone.utc).isoformat(),
                    files=len(rows),
                    sessions=len(sessions_used),
                    skipped=len(skipped),
                    labels=", ".join(f"{k}={v}" for k, v in sorted(per_label.items())) or "none",
                ),
            )
        return {"files": len(rows), "sessions": sessions_used, "skipped_without_consent": skipped, "per_label": per_label}

    def delete_session(self, session_id: str) -> None:
        folder = self.root / self._safe_id(session_id)
        if folder.exists():
            shutil.rmtree(folder)


def open_image(payload: bytes) -> Image.Image:
    image = Image.open(BytesIO(payload))
    image.load()
    return image


def validate_image_upload(payload: bytes) -> Image.Image:
    """Decode an uploaded frame and enforce the server-side limits (size, format, dimensions)."""
    if not payload:
        raise CollectError("empty upload", status=400)
    if len(payload) > MAX_IMAGE_BYTES:
        raise CollectError(f"image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB", status=413)
    try:
        image = open_image(payload)
    except Exception:  # noqa: BLE001 - corrupt data, decompression bombs and truncated files all mean "not a usable image"
        raise CollectError("invalid image", status=400) from None
    if (image.format or "").upper() not in ALLOWED_IMAGE_FORMATS:
        raise CollectError("unsupported image format: use JPEG or PNG", status=415)
    width, height = image.size
    if max(width, height) > MAX_IMAGE_SIDE or min(width, height) < MIN_IMAGE_SIDE:
        raise CollectError(f"image sides must be between {MIN_IMAGE_SIDE} and {MAX_IMAGE_SIDE} pixels", status=422)
    return image


def sniff_clip_type(payload: bytes) -> str | None:
    """'webm' or 'mp4' judged from the file's own first bytes (never from the client's name or MIME type), else None."""
    if payload[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if payload[4:8] == b"ftyp":
        return "mp4"
    return None
