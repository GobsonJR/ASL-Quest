"""Pre-training audit of collected webcam data (stdlib only: no torch, no MediaPipe, safe on a low-memory machine).

    venv311\\Scripts\\python.exe scripts\\audit_webcam.py
    venv311\\Scripts\\python.exe scripts\\audit_webcam.py --focus C D W X --min-per-letter 40

Reports accepted samples per letter, refused save attempts (quality / near-duplicate), room and background
distribution, and warns about letters or rooms that are too thin to train or evaluate on. Exit code 1 = not ready.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LETTERS = [chr(65 + i) for i in range(26)]
PLAIN = {"plain_wall"}


def room_key(session: dict) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(session.get("room", "")).lower()).strip("-") or session["session_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "dataset" / "webcam")
    ap.add_argument("--focus", nargs="*", default=list("CDWX"), help="letters you are specifically trying to fix")
    ap.add_argument("--min-per-letter", type=int, default=40, help="accepted singles per focus letter before training")
    ap.add_argument("--min-per-room-letter", type=int, default=8, help="accepted singles per (room, focus letter)")
    ap.add_argument("--min-rooms", type=int, default=3, help="train + val + held-out test, each in its own room")
    args = ap.parse_args()

    sessions = sorted(p for p in args.src.glob("s*") if (p / "session.json").exists())
    if not sessions:
        print(f"No sessions under {args.src}. Collect some samples first.")
        return 1

    per_letter: Counter[str] = Counter()
    per_room_letter: dict[str, Counter[str]] = defaultdict(Counter)
    room_info: dict[str, dict] = {}
    rejected: Counter[tuple[str, str]] = Counter()
    rejected_reasons: Counter[str] = Counter()
    holds = singles = unconfirmed = 0
    hands: Counter[str] = Counter()
    distances: Counter[str] = Counter()
    for folder in sessions:
        session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        room = room_key(session)
        info = room_info.setdefault(room, {"sessions": 0, "samples": 0, "backgrounds": set(), "lighting": set()})
        info["sessions"] += 1
        info["backgrounds"].add(session["background"])
        info["lighting"].add(session["lighting"])
        for meta in folder.glob("*/*.json"):
            s = json.loads(meta.read_text(encoding="utf-8"))
            if s.get("label_source") != "person_confirmed":
                unconfirmed += 1
                continue
            info["samples"] += 1
            hands[s["hand"]] += 1
            distances[s["distance"]] += 1
            if s["mode"] == "hold":
                holds += 1
                continue
            singles += 1
            per_letter[s["letter"]] += 1
            per_room_letter[room][s["letter"]] += 1
        log = folder / "rejections.jsonl"
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                rejected[(r["letter"], r["kind"])] += 1
                for reason in r["reasons"]:
                    rejected_reasons[re.sub(r"\s*\(.*", "", reason)] += 1

    warnings: list[str] = []
    print(f"sessions: {len(sessions)}   rooms: {len(room_info)}   accepted singles: {singles}   hold frames: {holds}")
    if unconfirmed:
        warnings.append(f"{unconfirmed} samples lack label_source=person_confirmed (must not be trained on)")

    print("\nAccepted single frames per letter (focus letters marked *):")
    for letter in LETTERS:
        n = per_letter[letter]
        mark = "*" if letter in args.focus else " "
        dup = rejected[(letter, "near-duplicate")]
        q = rejected[(letter, "quality")]
        print(f"  {mark}{letter}: {n:4d}   refused: {q} quality, {dup} near-duplicate")
        if letter in args.focus and n < args.min_per_letter:
            warnings.append(f"letter {letter}: {n} accepted < {args.min_per_letter}")
    if not per_letter:
        warnings.append("no accepted single frames at all")

    print("\nRefused save attempts by reason:", dict(rejected_reasons) or "none logged (only attempts after this audit was added are counted)")

    print("\nRooms:")
    for room, info in room_info.items():
        kind = "plain" if info["backgrounds"] <= PLAIN else ("natural" if not info["backgrounds"] & PLAIN else "mixed")
        counts = per_room_letter[room]
        weak = {l: counts[l] for l in args.focus}
        print(f"  {room}: {info['samples']} samples, {info['sessions']} session(s), {kind} background "
              f"{sorted(info['backgrounds'])}, lighting {sorted(info['lighting'])}; focus letters {weak}")
        for l in args.focus:
            if counts[l] < args.min_per_room_letter:
                warnings.append(f"room {room!r}, letter {l}: {counts[l]} < {args.min_per_room_letter}")
    kinds = {("plain" if i["backgrounds"] <= PLAIN else "natural") for i in room_info.values()}
    if "natural" not in kinds:
        warnings.append("no natural-background room recorded: the real failure case is not represented")
    if "plain" not in kinds:
        warnings.append("no plain-background room recorded: no like-for-like baseline")
    if len(room_info) < args.min_rooms:
        warnings.append(f"only {len(room_info)} room(s); an honest unseen-room split needs >= {args.min_rooms} (train / val / held-out test)")

    print(f"\nHand: {dict(hands)}   Distance: {dict(distances)}")
    for h in ("left", "right"):
        if hands[h] == 0:
            warnings.append(f"no {h}-hand samples")

    print()
    if warnings:
        print("NOT READY TO TRAIN:")
        for w in warnings:
            print("  - " + w)
        return 1
    print("Data volume checks pass. Next: export with room-held-out splits and check leakage before training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
