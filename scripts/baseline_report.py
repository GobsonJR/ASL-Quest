"""Measured baseline for the A-Z recognizer, in three independent sections (run them one at a time):

  dataset    held-out dataset test split: accuracy, macro-F1, per-class P/R/F1, confusion matrix, model latency
  detection  MediaPipe hand-detection rate per letter on held-out dataset photos, and detection latency on real
             960x720 webcam frames when captured bundles exist
  webcam     real webcam captures (live_diagnostics bundles) that carry a label file: per-request correctness and
             how stable consecutive predictions are

Each section merges into outputs/baseline/baseline.json and rewrites outputs/baseline/baseline.md. Torch and MediaPipe
are never loaded in the same process (together they exhaust memory on small machines).

    venv311\\Scripts\\python.exe scripts\\baseline_report.py --section dataset
    venv311\\Scripts\\python.exe scripts\\baseline_report.py --section detection
    venv311\\Scripts\\python.exe scripts\\baseline_report.py --section webcam --bundles outputs\\diagnostics\\live_requests_20261019_c_session

Webcam labels: a `labels.json` in the bundle folder, {"<bundle id>": "<true letter or null>"}. Only requests with a
confirmed label are scored; everything else is reported as unlabelled, never guessed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BEST_MODEL_PATH, DATASET_DIR, OUTPUTS_DIR, SPLITS_PATH  # noqa: E402

LETTERS = [chr(65 + i) for i in range(26)]
OUT = OUTPUTS_DIR / "baseline"


def merge(section: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "baseline.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report[section] = {"measured_at": datetime.now(timezone.utc).isoformat(), **data}
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report)
    print(f"\nsaved section '{section}' -> {path}")


def write_markdown(report: dict) -> None:
    lines = ["# Recognition baseline (measured)", ""]
    if "dataset" in report:
        d = report["dataset"]
        lines += [
            "## Held-out dataset test split",
            f"- checkpoint: `{d['checkpoint']}`",
            f"- images: {d['n_images']}, accuracy **{100 * d['accuracy']:.2f}%**, macro-F1 **{100 * d['macro_f1']:.2f}%**",
            f"- model latency (batch 1, {d['latency']['device']}): median {d['latency']['median_ms']:.1f} ms, p95 {d['latency']['p95_ms']:.1f} ms",
            "",
            "| letter | precision | recall | F1 |",
            "|---|---|---|---|",
        ]
        lines += [f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |" for c, v in d["per_class"].items()]
        lines += ["", "This split comes from the same recording setup as the training photos: it does NOT measure webcam accuracy.", ""]
    if "detection" in report:
        d = report["detection"]
        lines += ["## Hand detection", f"- dataset photos: detected {d['dataset_detected']}/{d['dataset_total']} ({100 * d['dataset_rate']:.0f}%); these are extreme close-ups, so this is NOT the webcam failure rate"]
        if d.get("webcam_frames"):
            lines.append(f"- real webcam frames ({d['webcam_frames']}): detection median {d['webcam_latency_median_ms']:.0f} ms, p95 {d['webcam_latency_p95_ms']:.0f} ms (only frames that were detected were captured, so no failure rate is available from them)")
        lines.append("")
    if "webcam" in report:
        w = report["webcam"]
        lines += ["## Real webcam captures", f"- bundles: {w['bundles']}, labelled: {w['labelled']}, unlabelled (not scored): {w['unlabelled']}"]
        for letter, s in w["per_letter"].items():
            lines.append(f"- **{letter}**: top-1 correct {s['correct']}/{s['n']} ({100 * s['rate']:.0f}%), accepted (confidence >= threshold) {s['accepted']}, wrong predictions {s['wrong_top']}")
        lines.append(f"- consecutive requests that kept the same top-1 label (same labelled pose): {w['stability']['same_top1_pairs']}/{w['stability']['pairs']}")
        lines.append("")
    (OUT / "baseline.md").write_text("\n".join(lines), encoding="utf-8")


def split_items(split: str = "test") -> list[tuple[Path, int]]:
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))["splits"][split]
    samples: list[Path] = []
    for cls in sorted(os.listdir(DATASET_DIR)):
        samples.extend(DATASET_DIR / cls / f for f in sorted(os.listdir(DATASET_DIR / cls)))
    items = [(samples[i], LETTERS.index(samples[i].parent.name)) for i in splits]
    assert all(p.name.upper().startswith(LETTERS[k]) for p, k in items[:200]), "split indices do not line up with files"
    return items


def section_dataset(args) -> None:
    import torch

    from src.data.preprocessing import eval_transforms
    from src.models.resnet18 import build_resnet18
    from src.utils.checkpoint import load_checkpoint

    device = torch.device(args.device)
    payload = load_checkpoint(args.model, map_location=device)
    assert payload["class_names"] == LETTERS
    net = build_resnet18(num_classes=26, pretrained=False)
    net.load_state_dict(payload["model_state_dict"])
    net.to(device).eval()
    transform = eval_transforms()
    items = split_items("test")
    y_true, y_pred = [], []
    with torch.no_grad():
        for s in range(0, len(items), 16):
            chunk = items[s : s + 16]
            batch = torch.stack([transform(Image.open(p).convert("RGB")) for p, _ in chunk]).to(device)
            y_pred.extend(net(batch).argmax(1).cpu().tolist())
            y_true.extend(k for _, k in chunk)
        # latency: single image, includes transform + forward, after warmup
        sample = Image.open(items[0][0]).convert("RGB")
        timings = []
        for i in range(120):
            t0 = time.perf_counter()
            x = transform(sample).unsqueeze(0).to(device)
            net(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 20:
                timings.append((time.perf_counter() - t0) * 1000)
    cm = np.zeros((26, 26), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(26), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(26), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(26), where=(precision + recall) > 0)
    merge("dataset", {
        "checkpoint": str(args.model.resolve()),
        "n_images": len(items),
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])} for i, c in enumerate(LETTERS)},
        "confusion_matrix": cm.tolist(),
        "latency": {"device": str(device), "median_ms": statistics.median(timings), "p95_ms": float(np.percentile(timings, 95))},
    })


def section_detection(args) -> None:
    from src.inference.hands import HandDetector

    detector = HandDetector(static_image_mode=True, detection_confidence=0.35)
    per_letter, total, found = {}, 0, 0
    items = split_items("test")
    by_letter: dict[int, list[Path]] = {}
    for path, k in items:
        by_letter.setdefault(k, []).append(path)
    for k, paths in sorted(by_letter.items()):
        chosen = paths[:: max(1, len(paths) // args.per_class)][: args.per_class]
        hits = sum(detector.landmark_box(np.array(Image.open(p).convert("RGB"))) is not None for p in chosen)
        per_letter[LETTERS[k]] = {"n": len(chosen), "detected": hits}
        total += len(chosen)
        found += hits
    webcam_ms: list[float] = []
    if args.bundles and args.bundles.exists():
        for folder in sorted(args.bundles.glob("*"))[:25]:
            frames = list(folder.glob("1_original_frame.*"))
            if not frames:
                continue
            rgb = np.array(Image.open(frames[0]).convert("RGB"))
            t0 = time.perf_counter()
            detector.landmark_box(rgb)
            webcam_ms.append((time.perf_counter() - t0) * 1000)
    data = {"dataset_total": total, "dataset_detected": found, "dataset_rate": found / total, "per_letter": per_letter, "webcam_frames": len(webcam_ms)}
    if webcam_ms:
        data["webcam_latency_median_ms"] = statistics.median(webcam_ms)
        data["webcam_latency_p95_ms"] = float(np.percentile(webcam_ms, 95))
    merge("detection", data)


def section_webcam(args) -> None:
    labels_path = args.bundles / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    records = []
    for folder in sorted(p for p in args.bundles.glob("*") if (p / "result.json").exists()):
        records.append((folder.name, json.loads((folder / "result.json").read_text(encoding="utf-8"))))
    per_letter: dict[str, dict] = {}
    scored = []
    for name, rec in records:
        truth = labels.get(name)
        if not truth:
            continue
        top = rec["top_predictions"][0]["label"]
        s = per_letter.setdefault(truth, {"n": 0, "correct": 0, "accepted": 0, "wrong": Counter()})
        s["n"] += 1
        s["correct"] += top == truth
        s["accepted"] += (not rec["low_confidence"]) and top == truth
        if top != truth:
            s["wrong"][top] += 1
        scored.append((name, truth, top))
    out = {}
    for letter, s in per_letter.items():
        out[letter] = {"n": s["n"], "correct": s["correct"], "rate": s["correct"] / s["n"], "accepted": s["accepted"], "wrong_top": dict(s["wrong"].most_common(5))}
    pairs = same = 0
    for (n1, t1, p1), (n2, t2, p2) in zip(scored, scored[1:]):
        if t1 == t2:
            pairs += 1
            same += p1 == p2
    merge("webcam", {
        "bundles": len(records), "labelled": len(scored), "unlabelled": len(records) - len(scored),
        "labels_file": str(labels_path) if labels_path.exists() else None,
        "per_letter": out, "stability": {"pairs": pairs, "same_top1_pairs": same},
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", choices=["dataset", "detection", "webcam"], required=True)
    ap.add_argument("--model", type=Path, default=BEST_MODEL_PATH)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--bundles", type=Path, default=OUTPUTS_DIR / "diagnostics" / "live_requests_20261019_c_session")
    args = ap.parse_args()
    {"dataset": section_dataset, "detection": section_detection, "webcam": section_webcam}[args.section](args)


if __name__ == "__main__":
    main()
