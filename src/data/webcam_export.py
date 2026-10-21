"""Turn collected webcam sessions into a training-ready, leakage-safe dataset.

Input  : dataset/webcam/<session>/<LETTER>/<sample>.jpg + .json   (written by backend/routers/collect.py)
Output : dataset/webcam_export/<split>/<LETTER>/<sample_id>.png    crops made by the SAME function the live app uses
         dataset/webcam_export/manifest.jsonl                      one row per crop: label, room, hand, landmarks...
         dataset/webcam_export/splits.json, export_report.json

Leakage rules:
  * a whole ROOM goes to exactly one of train / val / test (so the same wall, furniture and lighting can never
    appear on both sides of an evaluation), not merely a whole session;
  * labels are copied from the person-confirmed metadata, never predicted;
  * hold-sequence frames are near-duplicates by design: train/val keep only a few evenly spaced frames per
    sequence, the test split keeps them all but flags them (kind="hold") so accuracy can be computed on
    de-duplicated frames and stability on whole sequences.
No torch is imported here (MediaPipe only), so it can run on a machine with little free memory.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LETTERS = [chr(65 + i) for i in range(26)]
SPLITS = ("train", "val", "test")


class ExportError(ValueError):
    pass


def room_key(session: dict[str, Any]) -> str:
    room = re.sub(r"[^a-z0-9]+", "-", str(session.get("room", "")).lower()).strip("-")
    return room or session["session_id"]


def load_sessions(root: Path) -> dict[str, dict[str, Any]]:
    sessions: dict[str, dict[str, Any]] = {}
    for folder in sorted(p for p in root.glob("s*") if (p / "session.json").exists()):
        session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        samples = []
        for meta in sorted(folder.glob("*/*.json")):
            record = json.loads(meta.read_text(encoding="utf-8"))
            record["_image_path"] = str(meta.parent / record["image"])
            samples.append(record)
        sessions[session["session_id"]] = {"session": session, "samples": samples, "room": room_key(session)}
    return sessions


def _hash_order(text: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{text}".encode()).hexdigest()


def assign_splits(
    sessions: dict[str, dict[str, Any]],
    *,
    test_rooms: list[str] | None = None,
    val_rooms: list[str] | None = None,
    auto: bool = False,
    seed: int = 0,
    test_fraction: float = 0.2,
    val_fraction: float = 0.15,
) -> tuple[dict[str, str], list[str]]:
    """Returns ({session_id: split}, notes). Rooms are indivisible."""
    notes: list[str] = []
    rooms: dict[str, list[str]] = defaultdict(list)
    sizes: Counter[str] = Counter()
    for sid, entry in sessions.items():
        rooms[entry["room"]].append(sid)
        sizes[entry["room"]] += len(entry["samples"])
    if not rooms:
        raise ExportError("no sessions found")

    room_split: dict[str, str] = {}
    for name in test_rooms or []:
        room_split[name] = "test"
    for name in val_rooms or []:
        if room_split.get(name) == "test":
            raise ExportError(f"room {name!r} cannot be both val and test")
        room_split[name] = "val"
    unknown = [r for r in room_split if r not in rooms]
    if unknown:
        raise ExportError(f"unknown room(s): {unknown}; available: {sorted(rooms)}")

    if auto and not room_split:
        if len(rooms) < 2:
            raise ExportError(
                "only one room has been recorded: an unseen-room test needs at least 2 rooms (3 with a validation room). "
                "Record another room, or pass --allow-same-room to split by session instead (NOT a valid test of generalisation)."
            )
        total = sum(sizes.values())
        order = sorted(rooms, key=lambda r: _hash_order(r, seed))
        got_test = got_val = 0
        for r in order:
            if got_test < test_fraction * total and len(rooms) - len(room_split) > 1:
                room_split[r] = "test"
                got_test += sizes[r]
            elif len(rooms) >= 3 and got_val < val_fraction * total and len(rooms) - len(room_split) > 1:
                room_split[r] = "val"
                got_val += sizes[r]
        if "val" not in room_split.values():
            notes.append("fewer than 3 rooms: no validation room, so no unseen-room validation (model selection will use the original dataset's validation split only)")
    for r in rooms:
        room_split.setdefault(r, "train")
    if "test" not in room_split.values():
        notes.append("WARNING: no test room selected, so there is no unseen-room evaluation")
    assignment = {sid: room_split[entry["room"]] for sid, entry in sessions.items()}
    return assignment, notes


def assign_by_session(sessions: dict[str, dict[str, Any]], test_sessions: list[str], val_sessions: list[str]) -> tuple[dict[str, str], list[str]]:
    """Fallback when only one room exists: split by session. Rooms then overlap across splits."""
    assignment = {sid: "train" for sid in sessions}
    for sid in test_sessions:
        assignment[sid] = "test"
    for sid in val_sessions:
        assignment[sid] = "val"
    return assignment, ["WARNING: split by SESSION within the same room(s): train and test share backgrounds, so this does not measure unseen-room performance"]


def leakage_report(assignment: dict[str, str], sessions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_room: dict[str, set[str]] = defaultdict(set)
    by_background: dict[str, set[str]] = defaultdict(set)
    for sid, split in assignment.items():
        by_room[sessions[sid]["room"]].add(split)
        by_background[sessions[sid]["session"]["background"]].add(split)
    shared_rooms = sorted(r for r, s in by_room.items() if len(s) > 1)
    return {
        "rooms_in_multiple_splits": shared_rooms,
        "backgrounds_in_multiple_splits": sorted(b for b, s in by_background.items() if len(s) > 1),
        "ok_room_separated": not shared_rooms,
    }


def evenly_spaced(items: list[Any], keep: int) -> list[Any]:
    if len(items) <= keep:
        return items
    idx = np.linspace(0, len(items) - 1, keep).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def export_dataset(
    src: Path,
    out: Path,
    detector: Any,
    *,
    assignment: dict[str, str],
    sessions: dict[str, dict[str, Any]],
    pad: float = 0.30,
    hold_keep: int = 3,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Crop every usable sample with the live app's own crop function and write the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    agree = disagree = 0
    for sid, entry in sessions.items():
        split = assignment[sid]
        samples = entry["samples"]
        # hold sequences: keep a few evenly spaced frames for train/val, every frame for test
        by_seq: dict[str, list[dict]] = defaultdict(list)
        keep: list[dict[str, Any]] = []
        for sample in samples:
            if sample["mode"] == "hold":
                by_seq[sample["sequence_id"]].append(sample)
            else:
                keep.append(sample)
        for seq in by_seq.values():
            seq.sort(key=lambda s: s.get("frame_index") or 0)
            keep.extend(seq if split == "test" else evenly_spaced(seq, hold_keep))
        for sample in keep:
            frame = np.array(Image.open(sample["_image_path"]).convert("RGB"))
            crop = detector.crop_rgb(frame, padding=pad)
            if not crop.detected:
                skipped["no hand detected"] += 1
                continue
            observation = detector.observe(frame)
            folder = out / split / sample["letter"]
            folder.mkdir(parents=True, exist_ok=True)
            crop_path = folder / f"{sample['sample_id']}.png"
            crop.image.save(crop_path)
            mp_hand = observation.actual_hand if observation else None
            if mp_hand:
                agree += mp_hand == sample["hand"]
                disagree += mp_hand != sample["hand"]
            rows.append({
                "sample_id": sample["sample_id"], "session_id": sid, "room": entry["room"], "split": split,
                "letter": sample["letter"], "label_source": sample["label_source"], "hand": sample["hand"],
                "distance": sample["distance"], "lighting": sample["lighting"], "background": sample["background"],
                "face_visible": sample["face_visible"], "glasses": sample["glasses"], "kind": sample["mode"],
                "sequence_id": sample.get("sequence_id"), "frame_index": sample.get("frame_index"),
                "crop": str(crop_path.relative_to(out)).replace("\\", "/"), "crop_size": list(crop.image.size),
                "landmarks": [list(p) for p in observation.landmarks] if observation else None,
                "handedness_raw": observation.handedness if observation else None,
                "handedness_score": observation.handedness_score if observation else None,
            })
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    per_split: dict[str, Counter[str]] = {s: Counter() for s in SPLITS}
    for row in rows:
        per_split[row["split"]][row["letter"]] += 1
    report = {
        "exported": len(rows),
        "skipped": dict(skipped),
        "per_split_total": {s: sum(c.values()) for s, c in per_split.items()},
        "per_split_per_letter": {s: {l: c.get(l, 0) for l in LETTERS} for s, c in per_split.items()},
        "missing_letters": {s: [l for l in LETTERS if per_split[s].get(l, 0) == 0] for s in SPLITS},
        "session_split": assignment,
        "rooms": {sid: sessions[sid]["room"] for sid in sessions},
        "leakage": leakage_report(assignment, sessions),
        "handedness_vs_mediapipe": {"agree": agree, "disagree": disagree,
                                    "note": "MediaPipe's label is mirrored-selfie based; actual_hand corrects it for unmirrored frames"},
        "notes": notes or [],
        "crop_padding": pad,
    }
    (out / "splits.json").write_text(json.dumps({"session_split": assignment, "rooms": report["rooms"]}, indent=2), encoding="utf-8")
    (out / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
