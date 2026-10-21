"""Phase 1: does the live webcam path feed the model the same kind of input it was trained on?

    venv311\\Scripts\\python.exe scripts\\diag_preprocessing.py            (backend must be running; read-only)

Tests (all on the collected webcam frames, de-duplicated to <=3 frames per hold sequence; DIAGNOSTIC, not accuracy evidence):
  T1  the saved export crop equals what /predict makes from the full frame (same top-1 and confidence)
  T2  hand SCALE and MediaPipe handedness of the original training images vs the live crop
  T3  horizontal flip of the live crop (orientation / chirality sensitivity)
  T4  crop padding sweep 0.0 ... 0.8 (the live app uses 0.30)
  T5  JPEG quality 0.72 (live camera upload) vs 0.92 (what the Collect page stored)
  T6  resolution/interpolation: crop shrunk to 200x200 (the training images' size) before upload
Writes outputs/diagnostics/phase1_preprocessing.json
"""

from __future__ import annotations

import io
import json
import random
import statistics
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from _diag_common import LETTERS, OUT, ROOT, load_webcam_rows, make_detector, predict, top1


def accuracy(rows, preds):
    by = defaultdict(lambda: [0, 0])
    for r, (p, _) in zip(rows, preds):
        by[r["letter"]][1] += 1
        by[r["letter"]][0] += p == r["letter"]
    total = sum(v[1] for v in by.values())
    return {
        "accuracy": round(sum(v[0] for v in by.values()) / total, 3),
        "per_letter_correct": {l: f"{c}/{n}" for l, (c, n) in sorted(by.items())},
        "predicted": dict(Counter(p for p, _ in preds).most_common(6)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_webcam_rows(thin=True)
    det = make_detector()
    report: dict = {"n_frames": len(rows), "letters": sorted({r["letter"] for r in rows})}

    # ---------------------------------------------------------------- T1
    same = conf_diff = 0
    live_preds, crop_preds = [], []
    for r in rows:
        live = predict(r["frame_path"].read_bytes(), require_hand=True)          # what the app does with a camera frame
        crop = predict(Image.open(r["crop_path"]).convert("RGB"), require_hand=False)  # the saved training/eval crop
        a, b = top1(live), top1(crop)
        same += a[0] == b[0]
        conf_diff = max(conf_diff, abs(a[1] - b[1]))
        live_preds.append(a)
        crop_preds.append(b)
    report["T1_export_crop_equals_live_path"] = {"same_top1": f"{same}/{len(rows)}", "max_confidence_difference": round(conf_diff, 4), "live_path_accuracy": accuracy(rows, live_preds)}
    print("T1", report["T1_export_crop_equals_live_path"]["same_top1"], "identical top-1; max conf diff", round(conf_diff, 4))

    # ---------------------------------------------------------------- T2
    rng = random.Random(0)
    kag = {"n": 0, "detected": 0, "box_fraction": [], "handedness": Counter(), "per_letter_handed": defaultdict(Counter), "sizes": Counter()}
    for letter in LETTERS:
        files = sorted((ROOT / "dataset" / "asl_alphabet_train" / letter).glob("*.jpg"))
        for path in rng.sample(files, 15):
            with Image.open(path) as im:
                rgb = np.array(im.convert("RGB"))
            kag["n"] += 1
            kag["sizes"][rgb.shape[:2]] += 1
            obs = det.observe(rgb)
            if obs is None:
                continue
            kag["detected"] += 1
            pts = np.array(obs.landmarks)
            h, w = rgb.shape[:2]
            side = max((pts[:, 0].max() - pts[:, 0].min()) * w, (pts[:, 1].max() - pts[:, 1].min()) * h)
            kag["box_fraction"].append(side / max(h, w))
            kag["handedness"][obs.handedness] += 1
            kag["per_letter_handed"][letter][obs.handedness] += 1
    web_fraction, web_hand = [], Counter()
    for r in rows:
        pts = np.array(r["landmarks"])
        with Image.open(r["frame_path"]) as im:
            w, h = im.size
        side = max((pts[:, 0].max() - pts[:, 0].min()) * w, (pts[:, 1].max() - pts[:, 1].min()) * h)
        web_fraction.append(side / r["crop_size"][0])
        web_hand[r["handedness_raw"]] += 1
    q = lambda v: [round(float(x), 2) for x in np.percentile(v, [10, 50, 90])]
    report["T2_scale_and_handedness"] = {
        "training_images": {"sampled": kag["n"], "hand_detected": kag["detected"], "image_sizes": {f"{h}x{w}": n for (h, w), n in kag["sizes"].items()},
                            "hand_box_side_as_fraction_of_image_p10_p50_p90": q(kag["box_fraction"]), "raw_mediapipe_handedness": dict(kag["handedness"])},
        "live_webcam_crops": {"frames": len(rows), "hand_box_side_as_fraction_of_crop_p10_p50_p90": q(web_fraction), "raw_mediapipe_handedness": dict(web_hand), "padding": 0.30,
                              "note": "fraction is 1/(1+2*0.30)=0.625 by construction unless the minimum crop side applies"},
    }
    print("T2 training-image hand fraction p10/50/90", q(kag["box_fraction"]), "| live crop", q(web_fraction))
    print("   handedness training", dict(kag["handedness"]), "| webcam", dict(web_hand))

    # ---------------------------------------------------------------- T3 / T4 / T5 / T6
    crops = [Image.open(r["crop_path"]).convert("RGB") for r in rows]
    report["T3_hflip"] = {
        "original": accuracy(rows, [top1(predict(c)) for c in crops]),
        "flipped": accuracy(rows, [top1(predict(c.transpose(Image.Transpose.FLIP_LEFT_RIGHT))) for c in crops]),
    }
    print("T3 original", report["T3_hflip"]["original"]["accuracy"], "flipped", report["T3_hflip"]["flipped"]["accuracy"])

    sweep = {}
    for pad in (0.0, 0.15, 0.30, 0.50, 0.80):
        preds = []
        for r in rows:
            frame = np.array(Image.open(r["frame_path"]).convert("RGB"))
            crop = det.crop_rgb(frame, padding=pad)
            preds.append(top1(predict(crop.image)) if crop.detected else (None, 0.0))
        sweep[str(pad)] = accuracy(rows, preds)
        print("T4 padding", pad, sweep[str(pad)]["accuracy"])
    report["T4_padding_sweep"] = sweep

    jpeg = {}
    for quality in (92, 72, 50):
        preds = []
        for r in rows:
            buf = io.BytesIO()
            Image.open(r["frame_path"]).convert("RGB").save(buf, "JPEG", quality=quality)
            preds.append(top1(predict(buf.getvalue(), require_hand=True)))
        jpeg[f"q{quality}"] = {**accuracy(rows, preds), "agrees_with_q92": sum(a[0] == b[0] for a, b in zip(preds, live_preds)), "mean_abs_conf_change_vs_stored": round(statistics.mean(abs(a[1] - b[1]) for a, b in zip(preds, live_preds)), 3)}
        print("T5 jpeg", quality, jpeg[f"q{quality}"]["accuracy"], "top-1 agrees with stored frame:", jpeg[f"q{quality}"]["agrees_with_q92"], "/", len(rows))
    report["T5_jpeg_quality"] = jpeg

    small = [c.resize((200, 200), Image.Resampling.LANCZOS) for c in crops]
    report["T6_resolution_200px"] = {"crop_size_p50": int(statistics.median(r["crop_size"][0] for r in rows)), "at_200px": accuracy(rows, [top1(predict(c)) for c in small])}
    print("T6 crop median size", report["T6_resolution_200px"]["crop_size_p50"], "px; at 200px accuracy", report["T6_resolution_200px"]["at_200px"]["accuracy"])

    (OUT / "phase1_preprocessing.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("saved", OUT / "phase1_preprocessing.json")


if __name__ == "__main__":
    main()
