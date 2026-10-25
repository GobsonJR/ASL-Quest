"""READ-ONLY investigation of the MediaPipe "Left" hand accuracy drop (landmark models, dataset test frames).

Nothing is trained or written outside outputs/recognition_investigation/. It measures, it does not change behaviour.

Questions answered with data:
  Q1  Is the drop explained by WHICH LETTERS are labelled Left (a confound)?            -> per-letter Left vs Right accuracy
  Q2  Is MediaPipe's Left/Right label consistent with the hand's geometric chirality?   -> sign of the wrist/index-MCP/pinky-MCP triangle
  Q3  Which hand-convention makes the Left frames classify correctly?                   -> as-deployed vs never-flip vs always-flip
  Q4  Is the label stable? Mirror the image: does the label swap, and does the answer survive?
  Q5  On real webcam frames, how does the raw label relate to the hand the person said they used?

    venv311\\Scripts\\python.exe scripts\\investigate_handedness.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.settings import HAND_DETECTION_CONFIDENCE  # noqa: E402
from src.config import DATASET_DIR  # noqa: E402
from src.inference.hands import HandDetector  # noqa: E402
from src.landmarks.classifier import MEDIAPIPE_TO_ACTUAL_HAND, LandmarkClassifier  # noqa: E402

LETTERS = [chr(65 + i) for i in range(26)]
OUT = ROOT / "outputs" / "recognition_investigation"
RECORDS = ROOT / "outputs" / "compare_resnet_landmarks" / "records_dataset.jsonl"
MODELS = {"L1": ROOT / "models/experiments/landmark_L1_s0.joblib", "L2": ROOT / "models/experiments/landmark_L2_s0.joblib"}
WEBCAM_RAW = ROOT / "dataset" / "webcam"


def chirality(landmarks) -> int:
    """+1/-1: orientation of the (wrist, index MCP, pinky MCP) triangle in the image plane. Same view of the same
    hand gives the same sign; a mirrored hand (or the other hand seen the same way) gives the opposite sign."""
    p = np.asarray(landmarks)
    a, b = p[5, :2] - p[0, :2], p[17, :2] - p[0, :2]
    return 1 if a[0] * b[1] - a[1] * b[0] > 0 else -1


def predict(clf: LandmarkClassifier, landmarks, hand: str | None) -> str:
    return clf.predict_points(landmarks, hand, top_k=1)["prediction"]


def convention_hand(label: str | None, convention: str) -> str | None:
    if convention == "as_deployed":
        return MEDIAPIPE_TO_ACTUAL_HAND.get(label or "")
    if convention == "never_flip":
        return "right"
    if convention == "always_flip":
        return "left"
    if convention == "inverted_mapping":  # the opposite of the deployed mapping
        return {"Left": "left", "Right": "right"}.get(label or "")
    raise ValueError(convention)


def acc(pairs: list[tuple[str, str]]) -> str:
    return f"{100 * sum(t == p for t, p in pairs) / len(pairs):.1f}% (n={len(pairs)})" if pairs else "n/a"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    detector = HandDetector(static_image_mode=True, detection_confidence=HAND_DETECTION_CONFIDENCE)
    clfs = {k: LandmarkClassifier(v) for k, v in MODELS.items()}
    conventions = ["as_deployed", "never_flip", "always_flip"]
    lines = ["# Handedness investigation (measured)", ""]

    # ------------------------------------------------------------------ dataset: detected test frames
    ids = [json.loads(l)["id"] for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["detected"]]
    rows = []
    for n, fid in enumerate(ids, 1):
        _, letter, name = fid.split("/")
        rgb = np.array(Image.open(DATASET_DIR / letter / name).convert("RGB"))
        obs = detector.observe(rgb)
        flipped = detector.observe(np.ascontiguousarray(rgb[:, ::-1]))
        if obs is None:
            continue
        rows.append({"letter": letter, "label": obs.handedness, "score": obs.handedness_score, "lm": obs.landmarks, "chir": chirality(obs.landmarks),
                     "flip_label": flipped.handedness if flipped else None, "flip_lm": flipped.landmarks if flipped else None,
                     "flip_score": flipped.handedness_score if flipped else None})
        if n % 1000 == 0:
            print(f"dataset {n}/{len(ids)}", flush=True)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    lines += [f"## Dataset test frames where a hand was found: {len(rows)}", "",
              "MediaPipe raw label counts: " + str({k: len(v) for k, v in by_label.items()}), ""]

    # Q1 letter confound
    lines += ["### Q1 - is it a letter confound? (L2, as deployed; per-letter accuracy on Left vs Right frames)", "",
              "| letter | Left n | Left acc | Right n | Right acc |", "|---|---|---|---|---|"]
    gap_letters = []
    for letter in LETTERS:
        cells = {}
        for lab in ("Left", "Right"):
            sub = [r for r in by_label[lab] if r["letter"] == letter]
            cells[lab] = (len(sub), [(letter, predict(clfs["L2"], r["lm"], convention_hand(lab, "as_deployed"))) for r in sub])
        if cells["Left"][0] >= 10 and cells["Right"][0] >= 10:
            la = np.mean([t == p for t, p in cells["Left"][1]]); ra = np.mean([t == p for t, p in cells["Right"][1]])
            gap_letters.append((letter, la, ra))
        lines.append(f"| {letter} | {cells['Left'][0]} | {acc(cells['Left'][1])} | {cells['Right'][0]} | {acc(cells['Right'][1])} |")
    if gap_letters:
        lines += ["", f"Letters with >=10 frames in both groups: {len(gap_letters)}. Mean Left acc {100 * np.mean([g[1] for g in gap_letters]):.1f}% vs mean Right acc {100 * np.mean([g[2] for g in gap_letters]):.1f}% "
                  "(if the gap persists within letters, it is not just a different letter mix)."]

    # Q2 chirality vs label
    lines += ["", "### Q2 - does the raw label track geometry? (share of frames with chirality sign +1, by raw label)", ""]
    for lab in ("Left", "Right"):
        sub = by_label[lab]
        lines.append(f"- raw {lab}: n={len(sub)}, chirality +1 in {100 * np.mean([r['chir'] == 1 for r in sub]):.1f}%, median label score {np.median([r['score'] for r in sub]):.2f}")
    lines.append("If real left and right hands were being separated, the two groups would have OPPOSITE chirality signs; the same sign in both means the label is not a function of geometry for these close-ups (for one signer showing the same view).")
    lines.append("Within-label chirality by letter (share +1): " + str({l: (round(float(np.mean([r['chir'] == 1 for r in by_label['Left'] if r['letter'] == l])), 2) if sum(1 for r in by_label['Left'] if r['letter'] == l) >= 10 else None) for l in LETTERS if sum(1 for r in by_label['Left'] if r['letter'] == l) >= 10}))

    # Q3 conventions
    lines += ["", "### Q3 - which hand convention classifies each group correctly? (accuracy on detected frames)", "",
              "| model | convention | raw Left frames | raw Right frames | all |", "|---|---|---|---|---|"]
    for mname, clf in clfs.items():
        for conv in conventions:
            cells = {}
            for lab in ("Left", "Right"):
                cells[lab] = [(r["letter"], predict(clf, r["lm"], convention_hand(lab, conv))) for r in by_label[lab]]
            lines.append(f"| {mname} | {conv} | {acc(cells['Left'])} | {acc(cells['Right'])} | {acc(cells['Left'] + cells['Right'])} |")
    lines.append("")
    lines.append("`as_deployed`: Left->right (no mirror), Right->left (mirror). `never_flip`: treat every hand as right. `always_flip`: mirror every hand.")

    # Q4 mirror test
    lines += ["", "### Q4 - mirror the image: does the label swap, and does the answer survive? (L2, as deployed)", ""]
    sw = Counter()
    agree = Counter()
    for r in rows:
        if r["flip_lm"] is None:
            sw[(r["label"], "flipped image: no hand found")] += 1
            continue
        sw[(r["label"], r["flip_label"])] += 1
        a = predict(clfs["L2"], r["lm"], convention_hand(r["label"], "as_deployed"))
        b = predict(clfs["L2"], r["flip_lm"], convention_hand(r["flip_label"], "as_deployed"))
        agree[(r["label"], "same answer")] += a == b
        agree[(r["label"], "n")] += 1
    for (a, b), c in sorted(sw.items(), key=lambda kv: str(kv[0])):
        lines.append(f"- original {a} -> mirrored image label {b}: {c}")
    for lab in ("Left", "Right"):
        if agree[(lab, "n")]:
            lines.append(f"- original {lab}: the answer is unchanged after mirroring the image in {100 * agree[(lab, 'same answer')] / agree[(lab, 'n')]:.1f}% (n={agree[(lab, 'n')]})")
    lines.append("A hand-invariant pipeline would give the same answer for a hand and its mirror image; disagreement means the label-driven flip is not doing its job.")

    # Q7 hypothesis test: when the label is unreliable, are the LANDMARKS themselves wrong (not just the flip)?
    # Re-detect on the mirrored image (a hand-invariant pipeline must give the same letter) and use those landmarks.
    lines += ["", "### Q7 - accuracy when the landmarks come from the MIRRORED image instead (L2; the mirrored-image label drives the flip)", "",
              "| original raw label -> mirrored-image label | n | acc from original image | acc from mirrored image |", "|---|---|---|---|"]
    groups = defaultdict(list)
    for r in rows:
        if r["flip_lm"] is not None:
            groups[(r["label"], r["flip_label"])].append(r)
    for key in sorted(groups):
        g = groups[key]
        a = [(r["letter"], predict(clfs["L2"], r["lm"], convention_hand(r["label"], "as_deployed"))) for r in g]
        b = [(r["letter"], predict(clfs["L2"], r["flip_lm"], convention_hand(r["flip_label"], "as_deployed"))) for r in g]
        lines.append(f"| {key[0]} -> {key[1]} | {len(g)} | {acc(a)} | {acc(b)} |")
    both = [r for r in rows if r["flip_lm"] is not None]
    pick = [(r["letter"], predict(clfs["L2"], *((r["lm"], convention_hand(r["label"], "as_deployed")) if (r["score"] or 0) >= (r["flip_score"] or 0) else (r["flip_lm"], convention_hand(r["flip_label"], "as_deployed"))))) for r in both]
    orig = [(r["letter"], predict(clfs["L2"], r["lm"], convention_hand(r["label"], "as_deployed"))) for r in both]
    lines += ["", f"Over the {len(both)} frames where both orientations were detected: original image {acc(orig)}; using whichever orientation MediaPipe was more confident about {acc(pick)}.",
              "This is a diagnostic on already-trained models, not a proposed change: it shows whether the failure sits in the landmark layout. Any such change must be re-measured on webcam data first."]

    # Q6 alternative explanations: in-plane orientation, label confidence, hand size (both models are rotate=False)
    lines += ["", "### Q6 - alternative explanations (L2 as deployed; accuracy by condition, Left vs Right raw label)", "",
              "Models rotate flags: " + str({k: v.rotate for k, v in clfs.items()}), ""]

    def tilt(r) -> float:  # degrees the wrist->middle-MCP axis is rotated away from pointing straight up (image y down)
        v = np.asarray(r["lm"])[9, :2] - np.asarray(r["lm"])[0, :2]
        return float(np.degrees(np.arctan2(v[0], -v[1])))

    def size(r) -> float:
        pts = np.asarray(r["lm"])[:, :2]
        return float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))

    def table(title, key, bins):
        lines.extend(["", f"**{title}**", "", "| bin | Left n | Left acc | Right n | Right acc |", "|---|---|---|---|---|"])
        for name, test in bins:
            cells = {}
            for lab in ("Left", "Right"):
                sub = [r for r in by_label[lab] if test(key(r))]
                cells[lab] = [(r["letter"], predict(clfs["L2"], r["lm"], convention_hand(lab, "as_deployed"))) for r in sub]
            lines.append(f"| {name} | {len(cells['Left'])} | {acc(cells['Left'])} | {len(cells['Right'])} | {acc(cells['Right'])} |")

    table("hand tilt (0 = fingers up; negative = leaning left in the image)", tilt,
          [("< -60", lambda t: t < -60), ("-60..-20", lambda t: -60 <= t < -20), ("-20..20", lambda t: -20 <= t <= 20), ("20..60", lambda t: 20 < t <= 60), ("> 60", lambda t: t > 60)])
    table("MediaPipe handedness score", lambda r: r["score"],
          [("< 0.70", lambda v: v < 0.70), ("0.70-0.90", lambda v: 0.70 <= v < 0.90), ("0.90-0.99", lambda v: 0.90 <= v < 0.99), (">= 0.99", lambda v: v >= 0.99)])
    table("hand size (landmark box / image side)", size, [("< 0.35", lambda v: v < 0.35), ("0.35-0.50", lambda v: 0.35 <= v < 0.50), (">= 0.50", lambda v: v >= 0.50)])

    # ------------------------------------------------------------------ webcam: raw label vs what the person said
    lines += ["", "## Real webcam frames (all collected sessions): MediaPipe raw label vs the hand the person said they used", ""]
    cross = Counter()
    for sj in sorted(WEBCAM_RAW.glob("s*/session.json")):
        for meta in sorted(sj.parent.glob("*/*.json")):
            rec = json.loads(meta.read_text(encoding="utf-8"))
            img = sj.parent / rec["letter"] / rec["image"]
            if not img.exists():
                continue
            obs = detector.observe(np.array(Image.open(img).convert("RGB")))
            cross[(sj.parent.name, rec["hand"], obs.handedness if obs else "no hand")] += 1
    lines += ["| session | person said | MediaPipe raw | frames |", "|---|---|---|---|"]
    for (s, said, raw), c in sorted(cross.items()):
        lines.append(f"| {s} | {said} | {raw} | {c} |")
    total = sum(c for (_, _, raw), c in cross.items() if raw != "no hand")
    same = sum(c for (_, said, raw), c in cross.items() if raw != "no hand" and raw.lower() == said)
    swapped = sum(c for (_, said, raw), c in cross.items() if raw != "no hand" and {"Left": "right", "Right": "left"}[raw] == said)
    lines += ["", f"Raw label equals the person's stated hand in {same}/{total} frames; the deployed mapping (Left->right) equals it in {swapped}/{total}.",
              "The person-supplied label is a human input and could itself be wrong; this table says what the two sources do, not which is right."]

    (OUT / "handedness.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
