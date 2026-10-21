"""Phase 4: why is F read as B (and what else is confused)? Compares predictions, crops, landmark geometry and confidence.

    venv311\\Scripts\\python.exe scripts\\diag_errors.py [--landmark-model models\\experiments\\landmark_L1_s0.joblib]

Read-only. Needs the backend running (production ResNet over HTTP) and, optionally, an experimental landmark model.
Writes outputs/diagnostics/phase4_error_analysis.json and phase4_*.png contact sheets. Nothing is relabelled or deleted:
"suspect" lists are for you to review by eye.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from _diag_common import LETTERS, OUT, ROOT, load_webcam_rows, predict, sheet, top1
from src.landmarks.features import canonicalize

GEOMETRY_NAMES = ["pinch(thumb-index tip)", "index_ext", "middle_ext", "ring_ext", "pinky_ext", "thumb_tip_to_middle_tip"]


def geometry(points, hand) -> list[float]:
    p = canonicalize(points, hand, rotate=True)
    ext = lambda tip, mcp: float(np.linalg.norm(p[tip]) / (np.linalg.norm(p[mcp]) + 1e-9))  # noqa: E731 (>1.8 = finger extended)
    return [float(np.linalg.norm(p[4] - p[8])), ext(8, 5), ext(12, 9), ext(16, 13), ext(20, 17), float(np.linalg.norm(p[4] - p[12]))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmark-model", type=Path, default=ROOT / "models" / "experiments" / "landmark_L1_s0.joblib")
    ap.add_argument("--npz", type=Path, default=ROOT / "outputs" / "diagnostics" / "landmarks" / "dataset_landmarks.npz")
    args = ap.parse_args()
    rows = load_webcam_rows(thin=True)
    crops = [Image.open(r["crop_path"]).convert("RGB") for r in rows]
    mp_hand = {"Left": "right", "Right": "left"}  # the deployed mapping (src/landmarks/classifier.py)

    resnet = [top1(predict(c, top_k=10)) for c in crops]
    lm = []
    if args.landmark_model.exists():
        from src.landmarks.classifier import LandmarkClassifier

        clf = LandmarkClassifier(args.landmark_model)
        for r in rows:
            reply = clf.predict_points(r["landmarks"], mp_hand.get(r["handedness_raw"]), top_k=3)
            t = reply["top_predictions"][0]
            lm.append((t["label"], float(t["confidence"])))
    report: dict = {"n_frames": len(rows), "landmark_model": str(args.landmark_model) if lm else None, "per_letter": {}}

    by_letter = defaultdict(list)
    for i, r in enumerate(rows):
        by_letter[r["letter"]].append(i)
    for letter, idx in sorted(by_letter.items()):
        entry = {"n": len(idx), "resnet_correct": sum(resnet[i][0] == letter for i in idx), "resnet_top1": dict(Counter(resnet[i][0] for i in idx).most_common(4)),
                 "resnet_mean_conf_when_wrong": round(float(np.mean([resnet[i][1] for i in idx if resnet[i][0] != letter] or [0])), 2),
                 "resnet_wrong_but_confident_(>=0.70)": sum(resnet[i][0] != letter and resnet[i][1] >= 0.70 for i in idx)}
        if lm:
            entry.update(landmark_correct=sum(lm[i][0] == letter for i in idx), landmark_top1=dict(Counter(lm[i][0] for i in idx).most_common(4)))
        report["per_letter"][letter] = entry
        print(letter, entry)

    # ------------------------------------------------ geometry: is the webcam F shaped like the dataset's F or its B?
    data = np.load(args.npz, allow_pickle=False)
    kag = defaultdict(list)
    for pts, hand, letter in zip(data["landmarks"], data["hands"], data["letters"]):
        if str(letter) in ("F", "B", "W", "V", "X"):
            kag[str(letter)].append(geometry(pts, str(hand)))
    web = defaultdict(list)
    for r in rows:
        if r["letter"] in ("F", "B", "W", "V", "X"):
            web[r["letter"]].append(geometry(r["landmarks"], mp_hand.get(r["handedness_raw"])))
    geo = {}
    for letter in ("F", "B", "W", "V", "X"):
        geo[letter] = {"original_dataset": {"n": len(kag[letter]), **{n: round(float(np.mean([g[k] for g in kag[letter]])), 2) for k, n in enumerate(GEOMETRY_NAMES)}} if kag[letter] else None,
                       "webcam": {"n": len(web[letter]), **{n: round(float(np.mean([g[k] for g in web[letter]])), 2) for k, n in enumerate(GEOMETRY_NAMES)}} if web[letter] else None}
    report["geometry_mean_per_letter"] = geo
    for letter, g in geo.items():
        print("geometry", letter, "original:", g["original_dataset"], "| webcam:", g["webcam"])
    # nearest class centroid (original dataset geometry) for each webcam F / B frame
    cent = {l: np.mean(v, axis=0) for l, v in kag.items() if v}
    near = {}
    for letter in ("F", "B"):
        near[letter] = dict(Counter(min(cent, key=lambda c: np.linalg.norm(np.array(g) - cent[c])) for g in web[letter]).most_common())
    report["nearest_original_dataset_centroid_of_webcam_frames_(F,B,W,V,X only)"] = near
    print("nearest centroid among F/B/W/V/X for webcam F:", near["F"], " webcam B:", near["B"])

    # ------------------------------------------------ suspects (for review only)
    sharp = [(r["sample_id"], r["room"], r["letter"], json.loads(next(Path(ROOT / "dataset" / "webcam").glob(f"s*/{r['letter']}/{r['sample_id']}.json")).read_text())["quality"]["metrics"].get("sharpness")) for r in rows]
    report["lowest_sharpness_frames"] = sorted(sharp, key=lambda t: t[3] or 1e9)[:12]
    suspects = []
    if lm:
        for i, r in enumerate(rows):
            if lm[i][0] != r["letter"] and lm[i][1] >= 0.9 and resnet[i][0] != r["letter"]:
                suspects.append({"sample_id": r["sample_id"], "labelled": r["letter"], "landmark": lm[i], "resnet": resnet[i]})
    report["possible_label_or_shape_problems_(both_models_disagree_with_label_and_landmark_conf>=0.9)"] = suspects
    print(len(suspects), "frames where BOTH models disagree with the label (landmark conf >= 0.9)")

    # ------------------------------------------------ contact sheets
    OUT.mkdir(parents=True, exist_ok=True)
    cap = lambda i: f"{rows[i]['letter']}>{resnet[i][0]} {resnet[i][1]:.2f}" + (f" | lm {lm[i][0]} {lm[i][1]:.2f}" if lm else "")  # noqa: E731
    for letter in ("F", "B", "C", "V"):
        idx = by_letter[letter][:24]
        sheet([crops[i] for i in idx], OUT / f"phase4_webcam_{letter}_resnet_predictions.png", cols=6, size=160, labels=[cap(i) for i in idx])
    ok = [i for i in range(len(rows)) if resnet[i][0] == rows[i]["letter"]]
    rng = random.Random(0)
    pick = rng.sample(ok, min(12, len(ok)))
    sheet([crops[i] for i in pick], OUT / "phase4_webcam_correct_examples.png", cols=6, size=160, labels=[cap(i) for i in pick])
    rng2 = random.Random(1)
    for letter in ("F", "B"):
        files = sorted((ROOT / "dataset" / "asl_alphabet_train" / letter).glob("*.jpg"))
        imgs = [Image.open(p).convert("RGB") for p in rng2.sample(files, 18)]
        sheet(imgs, OUT / f"phase4_original_dataset_{letter}.png", cols=6, size=160)
    (OUT / "phase4_error_analysis.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("saved", OUT / "phase4_error_analysis.json", "and phase4_*.png")


if __name__ == "__main__":
    main()
