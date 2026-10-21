"""Phase 2: how much does the production model rely on the BACKGROUND rather than the hand?

    venv311\\Scripts\\python.exe scripts\\diag_background.py     (backend must be running; read-only)

For each de-duplicated collected frame the live crop is taken (padding 0.30), a hand mask is built, and the model is queried on:
  original         the live crop unchanged
  bg_gray_hull     everything outside a dilated hull of the 21 landmarks -> neutral grey (ImageNet mean: 0 after normalisation)
  bg_gray_grabcut  everything outside a GrabCut mask (seeded from landmark skeleton + palm + forearm) -> neutral grey
  hand_removed     the GrabCut hand (+forearm) -> grey, background kept (inverse test: is the answer still the same without a hand?)
Hand shape, orientation and scale are untouched (the pixels inside the mask are the original pixels).
There is no dedicated hand-segmentation model in the project, so the masks are approximations: inspect
outputs/diagnostics/phase2_masks_*.png. This is a diagnostic of sensitivity, not independent accuracy evidence.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image

from _diag_common import OUT, load_webcam_rows, predict, sheet, top1
from src.inference.hands import square_window

GRAY = (124, 116, 104)
FOCUS = ["F", "A", "C", "B", "V", "W", "X"]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
PALM = [0, 1, 2, 5, 9, 13, 17]


def hull_mask(pts: np.ndarray, side: int) -> np.ndarray:
    mask = np.zeros((side, side), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts.astype(np.int32)), 255)
    k = max(3, int(0.05 * side)) | 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def grabcut_mask(rgb: np.ndarray, pts: np.ndarray) -> np.ndarray:
    side = rgb.shape[0]
    base =np.zeros((side, side), np.uint8)
    cv2.fillConvexPoly(base, cv2.convexHull(pts.astype(np.int32)), 255)
    # forearm: a band from the wrist away from the middle-finger knuckle (the model's training images show the forearm)
    wrist, mid = pts[0], pts[9]
    d = (wrist - mid) / (np.linalg.norm(wrist - mid) + 1e-9)
    palm_len = float(np.linalg.norm(wrist - mid))
    width = float(np.linalg.norm(pts[5] - pts[17])) * 0.55
    nrm = np.array([-d[1], d[0]])
    quad = np.array([wrist + nrm * width, wrist - nrm * width, wrist - nrm * width + d * palm_len * 2.0, wrist + nrm * width + d * palm_len * 2.0]).astype(np.int32)
    fore = np.zeros((side, side), np.uint8)
    cv2.fillConvexPoly(fore, quad, 255)
    ell = lambda f: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(f * side)) | 1,) * 2)  # noqa: E731
    probable_fg = cv2.dilate(base, ell(0.07)) | fore
    probable_bg = cv2.dilate(base | fore, ell(0.20))
    mask = np.full((side, side), cv2.GC_BGD, np.uint8)
    mask[probable_bg > 0] = cv2.GC_PR_BGD
    mask[probable_fg > 0] = cv2.GC_PR_FGD
    sure = np.zeros((side, side), np.uint8)
    for a, b in BONES:
        cv2.line(sure, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), 255, max(2, int(0.035 * side)))
    cv2.fillConvexPoly(sure, cv2.convexHull(pts[PALM].astype(np.int32)), 255)
    mask[sure > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return hull_mask(pts, side)
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:  # keep the component that holds the palm
        keep = lab[int(pts[9][1]), int(pts[9][0])] or (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
        fg = (lab == keep).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ell(0.02))
    return cv2.dilate(fg, ell(0.012)) > 0  # a thin fringe so original edge pixels are not cut


def paint(rgb: np.ndarray, keep: np.ndarray) -> Image.Image:
    out = np.empty_like(rgb)
    out[:] = GRAY
    out[keep] = rgb[keep]
    return Image.fromarray(out)


def main() -> None:
    rows = load_webcam_rows(thin=True)
    conditions = ["original", "bg_gray_hull", "bg_gray_grabcut", "hand_removed"]
    results: dict[str, list] = {c: [] for c in conditions}
    kept_rows, skipped, sheets = [], 0, defaultdict(list)
    for r in rows:
        crop = Image.open(r["crop_path"]).convert("RGB")
        rgb = np.array(crop)
        side = rgb.shape[0]  # crops are square
        # The landmarks stored at export time are relative to the FULL frame; map them into the crop with the same window
        # the live crop used (re-detecting on the cropped image fails for ~30% of frames).
        if not r.get("landmarks"):
            skipped += 1
            continue
        with Image.open(r["frame_path"]) as frame:
            fw, fh = frame.size
        full = np.array([[p[0] * fw, p[1] * fh] for p in r["landmarks"]], dtype=np.float64)
        left, top, right, _ = square_window((full[:, 0].min(), full[:, 1].min(), full[:, 0].max(), full[:, 1].max()), 0.30)
        pts = np.clip((full - [left, top]) * (side / (right - left)), 1, side - 2)
        hm, gm = hull_mask(pts, side), grabcut_mask(rgb, pts)
        images = {"original": crop, "bg_gray_hull": paint(rgb, hm), "bg_gray_grabcut": paint(rgb, gm), "hand_removed": paint(rgb, ~gm)}
        for name, image in images.items():
            results[name].append(top1(predict(image, top_k=10)))
        kept_rows.append(r)
        if len(sheets[r["letter"]]) < 2 * 4:
            sheets[r["letter"]].extend(images[c] for c in conditions)
        r["_mask_fraction"] = float(gm.mean())
    print(f"{len(kept_rows)} frames used, {skipped} skipped (no landmarks on the crop)")

    report = {"frames_used": len(kept_rows), "skipped_no_landmarks": skipped, "mean_grabcut_mask_fraction_of_crop": round(float(np.mean([r['_mask_fraction'] for r in kept_rows])), 3), "per_letter": {}, "per_room": {}}
    for scope, key in (("per_letter", "letter"), ("per_room", "room")):
        groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(kept_rows):
            groups[r[key]].append(i)
        for g, idx in sorted(groups.items()):
            if scope == "per_letter" and g not in FOCUS:
                continue
            entry = {"n": len(idx)}
            for c in conditions:
                preds = [results[c][i] for i in idx]
                entry[c] = {
                    "correct": sum(p == kept_rows[i]["letter"] for p, i in zip((p[0] for p in preds), idx)) if key == "letter" else sum(results[c][i][0] == kept_rows[i]["letter"] for i in idx),
                    "mean_top1_confidence": round(float(np.mean([p[1] for p in preds])), 2),
                    "top1": dict(Counter(p[0] for p in preds).most_common(4)),
                    "changed_vs_original": sum(results[c][i][0] != results["original"][i][0] for i in idx) if c != "original" else 0,
                    "confident_(>=0.70)_answers": sum(p[1] >= 0.70 for p in preds),
                }
            report[scope][g] = entry
    for g, e in report["per_letter"].items():
        print(f"{g} n={e['n']:>2} | " + " | ".join(f"{c}: {e[c]['correct']} ok, {e[c]['top1']}" for c in conditions))

    OUT.mkdir(parents=True, exist_ok=True)
    for letter in FOCUS:
        if sheets[letter]:
            sheet(sheets[letter], OUT / f"phase2_masks_{letter}.png", cols=4, size=170)
    (OUT / "phase2_background.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("saved", OUT / "phase2_background.json", "and phase2_masks_<letter>.png (columns: original | hull | grabcut | hand removed)")


if __name__ == "__main__":
    main()
