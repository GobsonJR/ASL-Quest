"""Read-only consistency check of dataset/webcam (stdlib only; never moves, edits, relabels or deletes anything).

    venv311\\Scripts\\python.exe scripts\\check_webcam_sessions.py
    venv311\\Scripts\\python.exe scripts\\check_webcam_sessions.py --src dataset\\webcam --letters F A C

Prints sample counts by session and letter (single / hold), then flags:
  ERROR    sample JSON session_id != its folder, label != letter folder, label_source not person_confirmed,
           missing/unreadable JSON, image file missing, image without metadata, duplicate sample_id,
           sample lighting/background different from its session.json, session.json missing/invalid
  WARNING  an empty session; a session whose samples were captured long after it was created while a NEWER session
           already existed (the "I was in another room but frames went to the old session" pattern);
           two sessions with the same room name; a >5 minute pause inside one session (possible change of room);
           samples captured before their session was created.
Exit code 1 only for ERRORs. Warnings are for you to judge; nothing is changed automatically.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAUSE_SECONDS = 300


def parse_time(text: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "dataset" / "webcam")
    ap.add_argument("--letters", nargs="*", default=None, help="only print these letters in the table (checks still cover all)")
    args = ap.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    folders = sorted(p for p in args.src.glob("s*") if p.is_dir()) if args.src.exists() else []
    if not folders:
        print(f"No sessions under {args.src}.")
        return 0

    sessions: dict[str, dict] = {}
    samples: dict[str, list[dict]] = {}
    seen_ids: dict[str, str] = {}
    for folder in folders:
        sid = folder.name
        meta_path = folder / "session.json"
        try:
            sessions[sid] = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{sid}: session.json missing or invalid ({exc})")
            sessions[sid] = {}
        if sessions[sid].get("session_id") not in (None, sid):
            errors.append(f"{sid}: session.json says session_id={sessions[sid].get('session_id')!r}")
        samples[sid] = []
        listed_images: set[Path] = set()
        for meta in sorted(folder.glob("*/*.json")):
            try:
                rec = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                errors.append(f"{sid}/{meta.parent.name}/{meta.name}: unreadable JSON ({exc})")
                continue
            where = f"{sid}/{meta.parent.name}/{meta.name}"
            rec["_where"] = where
            samples[sid].append(rec)
            if rec.get("session_id") != sid:
                errors.append(f"{where}: records session_id={rec.get('session_id')!r} but lives in {sid}")
            if rec.get("letter") != meta.parent.name:
                errors.append(f"{where}: label {rec.get('letter')!r} != folder {meta.parent.name!r}")
            if rec.get("label_source") != "person_confirmed":
                errors.append(f"{where}: label_source={rec.get('label_source')!r}")
            if rec.get("sample_id") in seen_ids:
                errors.append(f"{where}: duplicate sample_id also in {seen_ids[rec['sample_id']]}")
            seen_ids[rec.get("sample_id", where)] = where
            image = meta.parent / rec.get("image", "")
            listed_images.add(image)
            if not image.is_file():
                errors.append(f"{where}: image {rec.get('image')!r} missing")
            for key in ("lighting", "background"):
                if sessions[sid].get(key) is not None and rec.get(key) != sessions[sid].get(key):
                    errors.append(f"{where}: {key} {rec.get(key)!r} differs from session {sessions[sid].get(key)!r}")
        for image in sorted(p for p in folder.glob("*/*") if p.suffix.lower() in (".jpg", ".png") and p not in listed_images):
            errors.append(f"{sid}/{image.parent.name}/{image.name}: image has no metadata JSON")

    # ------------------------------------------------------------ table
    letters = sorted({r["letter"] for recs in samples.values() for r in recs if "letter" in r})
    shown = [l for l in letters if not args.letters or l in args.letters]
    print(f"{'session':<42}{'room':<16}{'created (UTC)':<21}{'total':>6}   per letter (single+hold)")
    for sid in samples:
        s = sessions[sid]
        counts: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for r in samples[sid]:
            counts[r.get("letter", "?")][r.get("mode") == "hold"] += 1
        cells = "  ".join(f"{l}:{counts[l][0]}+{counts[l][1]}" for l in shown if l in counts) or "-"
        created = (s.get("created_at") or "")[:19].replace("T", " ")
        print(f"{sid:<42}{str(s.get('room', ''))[:15]:<16}{created:<21}{len(samples[sid]):>6}   {cells}")
        print(f"{'':<42}{s.get('lighting', '?')} / {s.get('background', '?')} / glasses={s.get('glasses')} / face={s.get('face_visible')}")

    # ------------------------------------------------------------ warnings
    by_room: dict[str, list[str]] = collections.defaultdict(list)
    for sid, s in sessions.items():
        by_room[str(s.get("room", "")).strip().lower()].append(sid)
        if not samples[sid]:
            warnings.append(f"{sid}: empty session (no samples)")
    for room, sids in by_room.items():
        if room and len(sids) > 1:
            warnings.append(f"room name {room!r} is used by {len(sids)} sessions: {', '.join(sids)} (intended? the exporter treats them as ONE room)")
    created_at = {sid: parse_time(s.get("created_at")) for sid, s in sessions.items()}
    for sid, recs in samples.items():
        times = sorted((parse_time(r.get("captured_at")), r) for r in recs if parse_time(r.get("captured_at")))
        if not times:
            continue
        c = created_at.get(sid)
        if c and times[0][0] < c:
            warnings.append(f"{sid}: {sum(1 for t, _ in times if t < c)} samples captured before the session was created")
        newer = sorted((t, other) for other, t in created_at.items() if t and c and t > c and other != sid)
        for t_new, other in newer:
            after = [r for t, r in times if t > t_new]
            if after:
                letters_after = collections.Counter(r["letter"] for r in after)
                warnings.append(
                    f"{sid}: {len(after)} samples ({dict(letters_after)}) were saved AFTER the newer session {other} was created "
                    f"({t_new:%H:%M:%S} UTC). If you were recording in the new room, they were saved to the wrong session"
                )
        for (t1, _), (t2, r2) in zip(times, times[1:]):
            if (t2 - t1).total_seconds() > PAUSE_SECONDS:
                warnings.append(f"{sid}: pause of {(t2 - t1).total_seconds() / 60:.1f} min before {r2['_where']} (did you change room here?)")

    print()
    for w in warnings:
        print("WARNING ", w)
    for e in errors:
        print("ERROR   ", e)
    print(f"\n{sum(len(v) for v in samples.values())} samples in {len(sessions)} sessions: {len(errors)} errors, {len(warnings)} warnings (nothing was changed).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
