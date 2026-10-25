"""Held-out WEBCAM evaluation: ResNet18 vs Landmark L1 vs Landmark L2 on exactly the same eligible frames.

READ-ONLY: it reads dataset/webcam/ (never writes, moves or deletes a session) and the model files, trains nothing, and writes
only to outputs/webcam_heldout_eval/. Methodology: WEBCAM_EVAL_PROTOCOL.md.

    venv311\\Scripts\\python.exe scripts\\evaluate_webcam_heldout.py coverage     # what has been collected vs what is required
    venv311\\Scripts\\python.exe scripts\\evaluate_webcam_heldout.py run          # detect + classify every eligible frame (heavy: torch + MediaPipe)
    venv311\\Scripts\\python.exe scripts\\evaluate_webcam_heldout.py report       # tables from the saved records (light)

Eligible frame = a saved sample whose SESSION and PERSON no model was trained on (a landmark model's training rooms come from its own
metadata; the ResNet checkpoint must declare dataset-only training). Positives carry a letter; negatives (no_hand / partial_hand) carry
none and measure false detections. The report states COMPLETE only when every coverage requirement is met by collected data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CONFIDENCE_THRESHOLD  # noqa: E402

LETTERS = [chr(65 + i) for i in range(26)]
NEGATIVE_KINDS = ("no_hand", "partial_hand")
WEBCAM_RAW = ROOT / "dataset" / "webcam"
OUT = ROOT / "outputs" / "webcam_heldout_eval"
DEFAULT_RESNET = ROOT / "models" / "best_model.pth"
DEFAULT_LANDMARKS = [ROOT / "models/experiments/landmark_L1_s0.joblib", ROOT / "models/experiments/landmark_L2_s0.joblib"]
SYSTEMS_MAIN = ["resnet_crop", "landmark_L1_s0", "landmark_L2_s0"]  # compared; resnet_full is a reference only
HOLD_KEEP = 3

# Minimum collected data before the evaluation may be called complete. Rationale in WEBCAM_EVAL_PROTOCOL.md: at ~30 frames per letter the
# 95% interval on a per-letter recall is still about +-15 points, so these are floors, not comfortable sample sizes.
REQUIREMENTS = {
    "min_per_letter": 30,        # eligible positive frames for EVERY letter, J and Z included
    "min_persons": 3,            # distinct named people
    "min_rooms": 3,              # distinct rooms
    "min_per_person": 100,
    "min_lighting_values": 3,    # distinct recorded lighting conditions ...
    "min_per_lighting": 100,     # ... each with this many frames
    "min_distance_values": 3,    # near / medium / far
    "min_per_distance": 100,
    "min_per_hand": 200,         # left AND right
    "min_left_per_letter": 5,
    "min_negatives_each": 100,   # no_hand and partial_hand
    "min_edge_positives": 50,    # labelled frames whose hand touches the frame edge
    "min_per_position_bin": 100, # derived after detection: each horizontal and vertical third of the frame
}


def load_compare():
    spec = importlib.util.spec_from_file_location("compare_resnet_landmarks", ROOT / "scripts" / "compare_resnet_landmarks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------------------ loading + eligibility
def room_slug(session_id: str) -> str:
    parts = session_id.split("_", 2)
    return parts[2] if len(parts) == 3 else session_id


def training_footprint(landmark_paths: list[Path]) -> dict[str, Any]:
    """Rooms (webcam sessions) any landmark model trained on, read from the model files' own metadata."""
    import joblib

    rooms: set[str] = set()
    for path in landmark_paths:
        for room in (joblib.load(path).get("trained_on") or {}).get("rooms") or []:
            if not str(room).startswith("original-"):  # 'original-blockN' = the Kaggle dataset's TRAIN split, not webcam sessions
                rooms.add(str(room))
    return {"rooms": sorted(rooms)}


def thin_hold(rows: list[dict]) -> list[dict]:
    singles, seqs = [], defaultdict(list)
    for r in rows:
        (seqs[r["sequence_id"]] if r.get("mode") == "hold" else singles).append(r)
    for seq in seqs.values():
        seq.sort(key=lambda r: r.get("frame_index") or 0)
        idx = sorted(set(np.linspace(0, len(seq) - 1, min(HOLD_KEEP, len(seq))).round().astype(int).tolist()))
        singles.extend(seq[i] for i in idx)
    return singles


def build_items(root: Path, training_rooms: set[str], extra_excluded_sessions: set[str] = frozenset(), declared_training_persons: set[str] = frozenset()) -> tuple[list[dict], dict[str, Any]]:
    """Every eligible frame as an item for compare_resnet_landmarks.run_set, plus an audit of what was excluded and why."""
    sessions = []
    for sj in sorted(root.glob("s*/session.json")):
        sessions.append((sj.parent, json.loads(sj.read_text(encoding="utf-8"))))
    # people recorded in training sessions, plus anyone the operator declares (older sessions carry no person field)
    train_persons = {s.get("person", "") for folder, s in sessions if room_slug(folder.name) in training_rooms and s.get("person")} | set(declared_training_persons)
    train_sessions_without_person = [f.name for f, s in sessions if room_slug(f.name) in training_rooms and not s.get("person")]
    items: list[dict] = []
    audit: dict[str, Any] = {"excluded_sessions": {}, "included_sessions": [], "training_persons": sorted(train_persons),
                             "training_sessions_without_person": train_sessions_without_person}
    for folder, session in sessions:
        sid = folder.name
        person = session.get("person", "")
        reason = None
        if room_slug(sid) in training_rooms:
            reason = "a model trained on this room"
        elif sid in extra_excluded_sessions:
            reason = "excluded by --exclude-session"
        elif person and person in train_persons:
            reason = f"person {person!r} appears in a training session"
        if reason:
            audit["excluded_sessions"][sid] = reason
            continue
        audit["included_sessions"].append(sid)
        base = {"session_id": sid, "room": room_slug(sid), "person": person, "lighting": session.get("lighting"),
                "background": session.get("background"), "glasses": session.get("glasses"), "face_visible": session.get("face_visible")}
        positives = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*/*.json")) if p.parent.name != "negatives"]
        for rec in thin_hold(positives):
            image = folder / rec["letter"] / rec["image"]
            if not image.exists():
                continue
            warnings = (rec.get("quality") or {}).get("warnings") or []
            items.append({"id": f"heldout/{sid}/{rec['sample_id']}", "path": str(image), "letter": rec["letter"],
                          "meta": {**base, "kind": rec.get("mode", "single"), "hand": rec.get("hand"), "distance": rec.get("distance"),
                                   "touches_edge": any("frame edge" in w for w in warnings)}})
        for kind in NEGATIVE_KINDS:
            for meta in sorted((folder / "negatives" / kind).glob("*.json")):
                rec = json.loads(meta.read_text(encoding="utf-8"))
                image = meta.parent / rec["image"]
                if image.exists():
                    items.append({"id": f"heldout/{sid}/{rec['sample_id']}", "path": str(image), "letter": "NONE",
                                  "meta": {**base, "kind": kind, "hand": None, "distance": rec.get("distance"), "touches_edge": bool(rec.get("touches_edge"))}})
    return items, audit


def coverage(items: list[dict], req: dict[str, int] = REQUIREMENTS) -> dict[str, Any]:
    """Collected-data requirements (the position-bin requirement needs detection, so it is checked in the report)."""
    pos = [i for i in items if i["letter"] != "NONE"]
    neg = [i for i in items if i["letter"] == "NONE"]
    per_letter = Counter(i["letter"] for i in pos)
    persons = Counter(i["meta"]["person"] for i in pos if i["meta"]["person"])
    rooms = Counter(i["meta"]["room"] for i in pos)
    lighting = Counter(i["meta"]["lighting"] for i in pos if i["meta"]["lighting"])
    distance = Counter(i["meta"]["distance"] for i in pos if i["meta"]["distance"])
    hands = Counter(i["meta"]["hand"] for i in pos if i["meta"]["hand"])
    left_per_letter = Counter(i["letter"] for i in pos if i["meta"]["hand"] == "left")
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    short = {l: per_letter.get(l, 0) for l in LETTERS if per_letter.get(l, 0) < req["min_per_letter"]}
    add(f"every letter >= {req['min_per_letter']} frames (J and Z included)", not short, "short: " + (", ".join(f"{l}={n}" for l, n in short.items()) or "none"))
    add(f">= {req['min_persons']} named people", len(persons) >= req["min_persons"], f"{len(persons)} named: {dict(persons)}; frames with no person recorded: {sum(1 for i in pos if not i['meta']['person'])}")
    add(f"each person >= {req['min_per_person']} frames", bool(persons) and min(persons.values()) >= req["min_per_person"], str(dict(persons)))
    add(f">= {req['min_rooms']} rooms", len(rooms) >= req["min_rooms"], str(dict(rooms)))
    add(f">= {req['min_lighting_values']} lighting conditions, each >= {req['min_per_lighting']} frames",
        sum(1 for n in lighting.values() if n >= req["min_per_lighting"]) >= req["min_lighting_values"], str(dict(lighting)))
    add(f"{req['min_distance_values']} distances, each >= {req['min_per_distance']} frames",
        sum(1 for n in distance.values() if n >= req["min_per_distance"]) >= req["min_distance_values"], str(dict(distance)))
    add(f"left and right hand each >= {req['min_per_hand']} frames", hands.get("left", 0) >= req["min_per_hand"] and hands.get("right", 0) >= req["min_per_hand"], str(dict(hands)))
    short_left = [l for l in LETTERS if left_per_letter.get(l, 0) < req["min_left_per_letter"]]
    add(f"every letter >= {req['min_left_per_letter']} LEFT-hand frames", not short_left, f"short: {', '.join(short_left) if short_left else 'none'}")
    for kind in NEGATIVE_KINDS:
        n = sum(1 for i in neg if i["meta"]["kind"] == kind)
        add(f"{kind} negatives >= {req['min_negatives_each']}", n >= req["min_negatives_each"], f"{n}")
    edge = sum(1 for i in pos if i["meta"]["touches_edge"])
    add(f"labelled frames with the hand touching the frame edge >= {req['min_edge_positives']}", edge >= req["min_edge_positives"], f"{edge}")
    return {"checks": [{"requirement": n, "met": ok, "have": d} for n, ok, d in checks], "met_all": all(ok for _, ok, _ in checks),
            "n_positive": len(pos), "n_negative": len(neg)}


# ------------------------------------------------------------------------------------------------ metrics (pure functions)
def answer(rec: dict, system: str) -> tuple[str, float] | None:
    out = rec.get(system)
    if not out or "top" not in out:
        return None
    return out["top"][0][0], float(out["top"][0][1])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return centre - half, centre + half


def fmt(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% ({100 * lo:.0f}-{100 * hi:.0f})"


def metrics(records: list[dict], system: str, threshold: float = CONFIDENCE_THRESHOLD) -> dict[str, Any]:
    pos = [r for r in records if r["letter"] != "NONE"]
    neg = {k: [r for r in records if r["letter"] == "NONE" and r["meta"]["kind"] == k] for k in NEGATIVE_KINDS}
    ans = [(r, answer(r, system)) for r in pos]
    correct = sum(1 for r, a in ans if a and a[0] == r["letter"])
    answered = [(r, a) for r, a in ans if a]
    accepted = [(r, a) for r, a in answered if a[1] >= threshold]
    cm = {l: Counter() for l in LETTERS}
    for r, a in ans:
        cm[r["letter"]][a[0] if a else "-"] += 1
    per_letter = {}
    for l in LETTERS:
        support = sum(cm[l].values())
        tp = cm[l][l]
        predicted = sum(cm[t][l] for t in LETTERS)  # answers "l" on positives of any letter
        per_letter[l] = {"support": support, "recall": tp / support if support else None, "precision": tp / predicted if predicted else None,
                         "answered": predicted}
    neg_out = {}
    for kind, rs in neg.items():
        a = [answer(r, system) for r in rs]
        neg_out[kind] = {"n": len(rs), "answered": sum(1 for x in a if x), "accepted": sum(1 for x in a if x and x[1] >= threshold),
                         "detected": sum(1 for r in rs if r["detected"])}
    return {
        "n_positive": len(pos), "detected": sum(1 for r in pos if r["detected"]), "answered": len(answered), "correct": correct,
        "accepted": len(accepted), "accepted_correct": sum(1 for r, a in accepted if a[0] == r["letter"]),
        "accepted_wrong": sum(1 for r, a in accepted if a[0] != r["letter"]),
        "rejected_but_correct": sum(1 for r, a in answered if a[1] < threshold and a[0] == r["letter"]),
        "per_letter": per_letter, "negatives": neg_out,
    }


def threshold_sweep(records: list[dict], system: str, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9)) -> list[dict]:
    rows = []
    for t in thresholds:
        m = metrics(records, system, t)
        n = m["n_positive"]
        neg_n = sum(v["n"] for v in m["negatives"].values())
        rows.append({"threshold": t, "accepted_share": m["accepted"] / n if n else None,
                     "accuracy_among_accepted": m["accepted_correct"] / m["accepted"] if m["accepted"] else None,
                     "wrong_accepted_rate": m["accepted_wrong"] / n if n else None,
                     "negative_false_accept": sum(v["accepted"] for v in m["negatives"].values()) / neg_n if neg_n else None})
    return rows


def same_frames(records_by_system: dict[str, list[str]]) -> bool:
    ids = [tuple(v) for v in records_by_system.values()]
    return len(set(ids)) <= 1


# ------------------------------------------------------------------------------------------------ run
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def cmd_coverage(args) -> None:
    footprint = training_footprint(args.landmarks)
    items, audit = build_items(args.root, set(footprint["rooms"]), set(args.exclude_session or []), set(args.training_person or []))
    cov = coverage(items)
    print(json.dumps({"training_rooms": footprint["rooms"], **{k: audit[k] for k in ("excluded_sessions", "included_sessions", "training_persons", "training_sessions_without_person")}}, indent=2))
    print(f"\neligible frames: {cov['n_positive']} labelled, {cov['n_negative']} negative\n")
    for c in cov["checks"]:
        print(f"[{'OK ' if c['met'] else 'MISSING'}] {c['requirement']}\n        have: {c['have']}")
    print("\nCOVERAGE " + ("MET" if cov["met_all"] else "NOT MET: the evaluation cannot be called complete"))


def cmd_run(args) -> None:
    footprint = training_footprint(args.landmarks)
    items, audit = build_items(args.root, set(footprint["rooms"]), set(args.exclude_session or []), set(args.training_person or []))
    cov = coverage(items)
    if args.require_complete and not cov["met_all"]:
        sys.exit("coverage not met; run `coverage` for the list (--require-complete was given)")
    if not items:
        sys.exit("no eligible frames")
    import torch  # noqa: F401
    from src.utils.checkpoint import load_checkpoint

    payload = load_checkpoint(args.resnet, map_location="cpu")
    policy = str(payload.get("training_policy", "")) + str(payload.get("init_from", ""))
    if "webcam" in policy.lower():
        sys.exit(f"the ResNet checkpoint declares webcam training ({policy!r}); its footprint must be added before evaluating")
    compare = load_compare()
    compare.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"git_head": git_head(), "frames": [{"id": i["id"], "letter": i["letter"], "sha256": sha256(Path(i["path"]))} for i in items],
                "audit": audit, "training_rooms": footprint["rooms"], "coverage": cov,
                "models": {str(p.name): sha256(p) for p in [args.resnet, *args.landmarks]}}
    manifest["frames_digest"] = hashlib.sha256("".join(f["sha256"] for f in manifest["frames"]).encode()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    compare.run_set("heldout", items, args.resnet, args.landmarks, args.device, resume=False)


# ------------------------------------------------------------------------------------------------ report
def lat(values: list[float]) -> str:
    return f"{np.median(values):.1f} / {np.quantile(values, 0.95):.1f}" if values else "n/a"


def cmd_report(args) -> None:
    from scipy.stats import binomtest

    path = OUT / "records_heldout.jsonl"
    if not path.exists():
        sys.exit("no records: run `run` first")
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    systems = SYSTEMS_MAIN + ["resnet_full"]
    systems = [s for s in systems if any(s in r for r in records)]
    pos = [r for r in records if r["letter"] != "NONE"]
    cov = manifest["coverage"]
    # derived coverage: detected positives per position bin
    det = [r for r in pos if r["detected"]]
    bins = {"horizontal": Counter("left" if r["hand"]["cx"] < 1 / 3 else "right" if r["hand"]["cx"] > 2 / 3 else "centre" for r in det),
            "vertical": Counter("top" if r["hand"]["cy"] < 1 / 3 else "bottom" if r["hand"]["cy"] > 2 / 3 else "middle" for r in det)}
    need = REQUIREMENTS["min_per_position_bin"]
    pos_ok = all(bins[a].get(b, 0) >= need for a, names in (("horizontal", ("left", "centre", "right")), ("vertical", ("top", "middle", "bottom"))) for b in names)
    complete = cov["met_all"] and pos_ok
    L = ["# Held-out webcam evaluation: ResNet18 vs Landmark L1 vs Landmark L2", "",
         f"## STATUS: {'COMPLETE (all coverage requirements met by collected data)' if complete else 'INCOMPLETE - the requirements below are NOT met; treat every number as a pilot, not a verdict'}", "",
         f"- git HEAD at run time: `{manifest['git_head']}`; eligible-frame digest `{manifest['frames_digest'][:16]}...`; {len(pos)} labelled + {len(records) - len(pos)} negative frames",
         f"- models (sha256 prefix): " + ", ".join(f"{k} `{v[:10]}`" for k, v in manifest["models"].items()),
         f"- training footprint excluded: rooms {manifest['training_rooms']}; persons {manifest['audit']['training_persons']}; "
         f"training sessions with NO person recorded (person-disjointness unprovable for them): {manifest['audit']['training_sessions_without_person']}",
         f"- excluded sessions: {manifest['audit']['excluded_sessions']}", "", "### Coverage requirements", "", "| requirement | met | have |", "|---|---|---|"]
    for c in cov["checks"]:
        L.append(f"| {c['requirement']} | {'yes' if c['met'] else '**NO**'} | {c['have']} |")
    L.append(f"| each horizontal and vertical third of the frame >= {need} detected labelled frames (derived) | {'yes' if pos_ok else '**NO**'} | {dict(bins['horizontal'])} {dict(bins['vertical'])} |")
    L += ["", "All systems were scored on the identical frame list (one record per frame carries every system's answer).", ""]

    # ---- detection
    neg_by_kind = {k: [r for r in records if r["letter"] == "NONE" and r["meta"]["kind"] == k] for k in NEGATIVE_KINDS}
    L += ["### Hand detection (MediaPipe; shared by ResNet-crop and both landmark models)", "",
          f"- recall on labelled frames (hand present): {fmt(len(det), len(pos))}"]
    for kind, rs in neg_by_kind.items():
        L.append(f"- {kind} negatives where a hand was nevertheless 'found': {fmt(sum(r['detected'] for r in rs), len(rs))}"
                 + (" (these are false detections)" if kind == "no_hand" else " (a hand cut by the frame edge; detection here is not an error by itself)"))
    L.append("")

    # ---- headline
    L += ["### Accuracy", "", f"Threshold = the app's {CONFIDENCE_THRESHOLD:.2f}. End-to-end counts a missed hand as wrong; 'when answered' is conditional on detection.", "",
          "| system | end-to-end accuracy | accuracy when answered | answered | accepted | accuracy among accepted | wrong & accepted | correct but rejected |", "|---|---|---|---|---|---|---|---|"]
    ms = {s: metrics(records, s) for s in systems}
    for s in systems:
        m = ms[s]
        n = m["n_positive"]
        L.append(f"| {s}{' (reference)' if s == 'resnet_full' else ''} | {fmt(m['correct'], n)} | {fmt(m['correct'], m['answered'])} | {m['answered']}/{n} | {m['accepted']}/{n} | "
                 f"{fmt(m['accepted_correct'], m['accepted'])} | {m['accepted_wrong']}/{n} | {m['rejected_but_correct']} |")
    L += ["", "Paired vs `resnet_crop` (labelled frames, end-to-end): only-ResNet-right / only-other-right / exact binomial p", ""]
    base = {r["id"]: (answer(r, "resnet_crop") or ("-", 0))[0] == r["letter"] for r in pos}
    for s in systems:
        if s in ("resnet_crop",):
            continue
        a = sum(1 for r in pos if base[r["id"]] and not ((answer(r, s) or ("-", 0))[0] == r["letter"]))
        b = sum(1 for r in pos if not base[r["id"]] and ((answer(r, s) or ("-", 0))[0] == r["letter"]))
        L.append(f"- {s}: {a} / {b} / p={binomtest(a, a + b, 0.5).pvalue if a + b else 1.0:.2g}")
    L += ["", "Frames within a session are correlated, so these intervals understate the real uncertainty; the per-person tables below show the spread.", ""]

    # ---- negatives + threshold sweep
    L += ["### False answers on frames with no usable hand", "", "| system | kind | n | answered | accepted at threshold |", "|---|---|---|---|---|"]
    for s in SYSTEMS_MAIN:
        if s in ms:
            for kind, v in ms[s]["negatives"].items():
                L.append(f"| {s} | {kind} | {v['n']} | {v['answered']} | {v['accepted']} |")
    L += ["", "### Confidence-threshold sweep (coverage vs error)", "", "| system | threshold | accepted share | accuracy among accepted | wrong-accept rate | accepted on negatives |", "|---|---|---|---|---|---|"]
    pc = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    for s in SYSTEMS_MAIN:
        if s in ms:
            for row in threshold_sweep(records, s):
                L.append(f"| {s} | {row['threshold']:.2f} | {pc(row['accepted_share'])} | {pc(row['accuracy_among_accepted'])} | {pc(row['wrong_accepted_rate'])} | {pc(row['negative_false_accept'])} |")

    # ---- per letter
    L += ["", "### Per-letter recall / precision (labelled frames; a missed hand counts against recall; precision = correct among answers of that letter)", "",
          "| letter | n | detected | " + " | ".join(f"{s} R / P" for s in SYSTEMS_MAIN if s in ms) + " |", "|---|---|---|" + "---|" * len([s for s in SYSTEMS_MAIN if s in ms])]
    def pct0(x):
        return "-" if x is None else f"{100 * x:.0f}%"

    for l in LETTERS:
        sub = [r for r in pos if r["letter"] == l]
        flag = " **(motion letter, still pose only)**" if l in "JZ" else ""
        cells = [pct0(ms[s]["per_letter"][l]["recall"]) + " / " + pct0(ms[s]["per_letter"][l]["precision"]) for s in SYSTEMS_MAIN if s in ms]
        if not sub:
            L.append(f"| {l}{flag} | 0 (!) | - | " + " | ".join(cells) + " |")
            continue
        short = " (!)" if len(sub) < REQUIREMENTS["min_per_letter"] else ""
        L.append(f"| {l}{flag} | {len(sub)}{short} | {100 * sum(r['detected'] for r in sub) / len(sub):.0f}% | " + " | ".join(cells) + " |")
    L.append("")
    L.append("(!) = below the per-letter requirement.")

    # ---- breakdowns
    def breakdown(title: str, key, only_detected: bool = False) -> None:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in pos:
            if only_detected and not r["detected"]:
                continue
            g = key(r)
            if g is not None:
                groups[str(g)].append(r)
        if len(groups) < 1:
            return
        L.extend(["", f"### By {title}" + (" (detected frames only)" if only_detected else ""), "", "| group | n | detected | " + " | ".join(SYSTEMS_MAIN) + " |", "|---|---|---|" + "---|" * len(SYSTEMS_MAIN)])
        for g in sorted(groups):
            rs = groups[g]
            cells = [fmt(sum(1 for r in rs if (answer(r, s) or ("-", 0))[0] == r["letter"]), len(rs)) for s in SYSTEMS_MAIN]
            L.append(f"| {g}{' (n<20)' if len(rs) < 20 else ''} | {len(rs)} | {100 * sum(r['detected'] for r in rs) / len(rs):.0f}% | " + " | ".join(cells) + " |")

    for title, k in (("person", "person"), ("room", "room"), ("recorded lighting", "lighting"), ("background", "background"), ("distance", "distance"), ("hand used", "hand"), ("glasses", "glasses"), ("face visible", "face_visible")):
        breakdown(title, lambda r, k=k: r["meta"].get(k))
    breakdown("hand touching the frame edge", lambda r: "touches edge" if r["meta"].get("touches_edge") else "inside frame")
    b = [r["brightness"] for r in pos]
    breakdown("measured brightness (absolute bands)", lambda r: "dark (<90)" if r["brightness"] < 90 else "bright (>180)" if r["brightness"] > 180 else "normal (90-180)")
    breakdown("hand position, horizontal thirds", lambda r: "left" if r["hand"]["cx"] < 1 / 3 else "right" if r["hand"]["cx"] > 2 / 3 else "centre", True)
    breakdown("hand position, vertical thirds", lambda r: "top" if r["hand"]["cy"] < 1 / 3 else "bottom" if r["hand"]["cy"] > 2 / 3 else "middle", True)
    breakdown("hand size (fraction of the longer frame side)", lambda r: "small (<0.25)" if r["hand"]["size"] < 0.25 else "large (>0.40)" if r["hand"]["size"] > 0.40 else "medium", True)

    # ---- latency
    L += ["", "### Latency (ms, median / p95, batch 1, this machine)", "",
          f"- MediaPipe detection (all systems need it): {lat([r['detect_ms'] for r in records])}",
          f"- crop: {lat([r['crop_ms'] for r in records if 'crop_ms' in r])}",
          f"- ResNet18 on the GPU: {lat([r['resnet_crop']['ms'] for r in records if r.get('resnet_crop')])}"]
    cpu = OUT / "cpu_latency_heldout.json"
    if cpu.exists():
        L.append(f"- ResNet18 on the CPU: {lat(json.loads(cpu.read_text())['ms'])}")
    for s in SYSTEMS_MAIN:
        if s.startswith("landmark_"):
            L.append(f"- {s} classifier: {lat([r[s]['ms'] for r in records if r.get(s) and 'ms' in r[s]])}")
    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("coverage", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=WEBCAM_RAW)
        p.add_argument("--landmarks", nargs="+", type=Path, default=DEFAULT_LANDMARKS)
        p.add_argument("--exclude-session", action="append", help="extra session id to leave out (repeatable)")
        p.add_argument("--training-person", action="append", help="person slug who appears in a model's TRAINING data (repeatable); all their sessions are excluded")
        if name == "run":
            p.add_argument("--resnet", type=Path, default=DEFAULT_RESNET)
            p.add_argument("--device", default="cuda")
            p.add_argument("--require-complete", action="store_true", help="refuse to run until every coverage requirement is met")
    args = ap.parse_args()
    {"coverage": cmd_coverage, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
