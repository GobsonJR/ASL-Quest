"""READ-ONLY head-to-head: production ResNet18 vs the experimental MediaPipe-landmark classifiers, A-Z incl. J and Z.

Nothing is trained, written to models/ or promoted. Every frame goes through the SAME detector the backend builds
(HandDetector(static_image_mode=True, detection_confidence=HAND_DETECTION_CONFIDENCE)) once, and each recognizer is scored
on the identical frame, so all comparisons are paired.

Systems scored per frame
  resnet_crop   what the live app does today: MediaPipe finds the hand -> square crop (padding 0.30) -> ResNet18
                (CameraPractice sends require_hand=true). No hand found => no answer.
  resnet_full   reference only: ResNet18 on the whole frame with no MediaPipe (how the checkpoint was validated)
  <landmarks>   MediaPipe 21 landmarks -> geometric features -> sklearn model (one per --landmarks file)

Held-out sets (never used to train, select or early-stop ANY of the scored models)
  dataset  the 11,700-image test split of the original Kaggle photos (last 15% of every class, by numeric filename).
           The only set with J and Z. Same signer/setup as training, so it flatters both models.
  webcam   the 'test' sessions of dataset/webcam_export_holdout (rooms the models never saw): real webcam frames, but
           only the letters recorded so far (no J/Z).

    venv311\\Scripts\\python.exe scripts\\compare_resnet_landmarks.py run --sets dataset webcam
    venv311\\Scripts\\python.exe scripts\\compare_resnet_landmarks.py report
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CONFIDENCE_THRESHOLD, DATASET_DIR, SPLITS_PATH  # noqa: E402

LETTERS = [chr(65 + i) for i in range(26)]
OUT = ROOT / "outputs" / "compare_resnet_landmarks"
WEBCAM_EXPORT = ROOT / "dataset" / "webcam_export_holdout"
WEBCAM_RAW = ROOT / "dataset" / "webcam"
DEFAULT_LANDMARKS = ["models/experiments/landmark_L1_s0.joblib", "models/experiments/landmark_L2_s0.joblib"]


# ----------------------------------------------------------------------------------------------- test sets
def numeric_key(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else -1


def dataset_items(limit_per_class: int | None) -> list[dict]:
    files: list[Path] = []
    for cls in sorted(os.listdir(DATASET_DIR)):
        files.extend(DATASET_DIR / cls / f for f in sorted(os.listdir(DATASET_DIR / cls)))
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))["splits"]
    assert not (set(splits["test"]) & (set(splits["train"]) | set(splits["val"]))), "test overlaps train/val"
    by_letter: dict[str, list[Path]] = defaultdict(list)
    for i in splits["test"]:
        by_letter[files[i].parent.name].append(files[i])
    items = []
    for letter in LETTERS:
        paths = sorted(by_letter[letter], key=lambda p: numeric_key(p.name))
        all_numbers = sorted(numeric_key(p.name) for p in (DATASET_DIR / letter).iterdir())
        # the split promises "last 15% by numeric filename": prove the mapping from indices to files really is that
        assert [numeric_key(p.name) for p in paths] == all_numbers[-len(paths):], f"{letter}: split does not match files"
        if limit_per_class:
            paths = paths[:: max(1, len(paths) // limit_per_class)][:limit_per_class]
        items += [{"id": f"dataset/{p.parent.name}/{p.name}", "path": str(p), "letter": letter, "meta": {}} for p in paths]
    return items


def webcam_items() -> list[dict]:
    rows = [json.loads(l) for l in (WEBCAM_EXPORT / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == "test"]
    singles, seqs = [], defaultdict(list)
    for r in rows:
        (seqs[r["sequence_id"]] if r["kind"] == "hold" else singles).append(r)
    for seq in seqs.values():  # a few evenly spaced frames per hold so near-identical frames don't dominate
        seq.sort(key=lambda r: r.get("frame_index") or 0)
        idx = sorted(set(np.linspace(0, len(seq) - 1, min(3, len(seq))).round().astype(int).tolist()))
        singles.extend(seq[i] for i in idx)
    items = []
    for r in singles:
        folder = WEBCAM_RAW / r["session_id"] / r["letter"]
        raw = next((p for ext in ("jpg", "png") if (p := folder / f"{r['sample_id']}.{ext}").exists()), None)
        if raw is None:
            continue
        meta = {k: r.get(k) for k in ("session_id", "room", "lighting", "background", "distance", "hand", "face_visible", "glasses", "kind")}
        items.append({"id": f"webcam/{r['session_id']}/{r['sample_id']}", "path": str(raw), "letter": r["letter"], "meta": meta})
    return items


# ----------------------------------------------------------------------------------------------- scoring one frame
def top_of(probs: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    order = np.argsort(probs)[::-1][:k]
    return [(LETTERS[int(i)], float(probs[i])) for i in order]


def run_set(name: str, items: list[dict], resnet_path: Path, landmark_paths: list[Path], device: str, resume: bool) -> None:
    import torch
    from src.config import HAND_LANDMARKER_PATH  # noqa: F401
    from backend.settings import HAND_DETECTION_CONFIDENCE
    from src.data.preprocessing import eval_transforms
    from src.inference.hands import HandDetector
    from src.landmarks.classifier import LandmarkClassifier
    from src.landmarks.features import LandmarkError
    from src.models.resnet18 import build_resnet18
    from src.utils.checkpoint import load_checkpoint

    payload = load_checkpoint(resnet_path, map_location=device)
    assert payload["class_names"] == LETTERS
    net = build_resnet18(num_classes=26, pretrained=False)
    net.load_state_dict(payload["model_state_dict"])
    net.to(device).eval()
    transform = eval_transforms()
    detector = HandDetector(static_image_mode=True, detection_confidence=HAND_DETECTION_CONFIDENCE)
    if not detector.available:
        sys.exit(f"hand detector unavailable: {detector.error}")
    clfs = {p.stem: LandmarkClassifier(p) for p in landmark_paths}

    def resnet(image: Image.Image) -> tuple[list[tuple[str, float]], float]:
        t0 = time.perf_counter()
        x = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(net(x), dim=1)[0].cpu().numpy()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        return top_of(probs), (time.perf_counter() - t0) * 1000

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"records_{name}.jsonl"
    done = set()
    if resume and out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    cpu_crops: list[Image.Image] = []
    t_start = time.time()
    with out_path.open("a" if resume else "w", encoding="utf-8") as handle:
        for n, item in enumerate(items, 1):
            if item["id"] in done:
                continue
            rgb = np.array(Image.open(item["path"]).convert("RGB"))
            h, w = rgb.shape[:2]
            rec = {"id": item["id"], "letter": item["letter"], "meta": item["meta"], "size": [w, h],
                   "brightness": float(rgb.mean(axis=2).mean())}
            t0 = time.perf_counter()
            try:
                obs = detector.observe(rgb)
                rec["detect_error"] = None
            except Exception as exc:  # noqa: BLE001 - a detector crash is a detection failure, and is counted as one
                obs, rec["detect_error"] = None, repr(exc)
            rec["detect_ms"] = (time.perf_counter() - t0) * 1000
            rec["detected"] = obs is not None
            full, ms = resnet(Image.fromarray(rgb))
            rec["resnet_full"] = {"top": full, "ms": ms}
            if obs is not None:
                xs = [p[0] * w for p in obs.landmarks]
                ys = [p[1] * h for p in obs.landmarks]
                rec["hand"] = {"cx": (min(xs) + max(xs)) / 2 / w, "cy": (min(ys) + max(ys)) / 2 / h,
                               "size": max(max(xs) - min(xs), max(ys) - min(ys)) / max(w, h),
                               "handedness": obs.handedness, "handedness_score": obs.handedness_score}
                t0 = time.perf_counter()
                crop = detector.crop_rgb(rgb, observation=obs)
                rec["crop_ms"] = (time.perf_counter() - t0) * 1000
                if crop.detected:
                    top, ms = resnet(crop.image)
                    rec["resnet_crop"] = {"top": top, "ms": ms}
                    if len(cpu_crops) < 200:
                        cpu_crops.append(crop.image)
                else:
                    rec["resnet_crop"] = None  # box fell outside the frame: the app treats this as "no hand"
                for key, clf in clfs.items():
                    t0 = time.perf_counter()
                    try:
                        reply = clf.predict_observation(obs, top_k=3)
                        rec[key] = {"top": [(t["label"], t["confidence"]) for t in reply["top_predictions"]],
                                    "ms": (time.perf_counter() - t0) * 1000}
                    except LandmarkError:
                        rec[key] = {"unusable": True}
            else:
                rec["resnet_crop"] = None
                for key in clfs:
                    rec[key] = None
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            if n % 250 == 0:
                print(f"[{name}] {n}/{len(items)}  {time.time() - t_start:.0f}s", flush=True)

    # ResNet on CPU (the likely deployed hardware: a small server without a GPU), same crops, batch 1
    cpu_net = copy.deepcopy(net).cpu().eval()
    times = []
    with torch.no_grad():
        for i, image in enumerate(cpu_crops):
            t0 = time.perf_counter()
            torch.softmax(cpu_net(transform(image.convert("RGB")).unsqueeze(0)), dim=1)
            if i >= 5:
                times.append((time.perf_counter() - t0) * 1000)
    (OUT / f"cpu_latency_{name}.json").write_text(json.dumps({"n": len(times), "ms": times}), encoding="utf-8")
    print(f"[{name}] done: {len(items)} frames in {time.time() - t_start:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------- report
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return centre - half, centre + half


def pct(k: int, n: int, ci: bool = True) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% ({100 * lo:.0f}-{100 * hi:.0f})" if ci else f"{100 * k / n:.1f}%"


def answer(rec: dict, system: str):
    """(label, confidence) or None when the system gave no answer for this frame."""
    out = rec.get(system)
    if not out or "top" not in out:
        return None
    return out["top"][0]


def quant(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else float("nan")


def lat(values: list[float]) -> str:
    return f"{statistics.median(values):.1f} / {quant(values, 0.95):.1f}" if values else "n/a"


def report() -> None:
    from scipy.stats import binomtest

    lines: list[str] = ["# ResNet18 vs MediaPipe-landmark recognizers - held-out comparison", ""]
    summary: dict = {}
    for path in sorted(OUT.glob("records_*.jsonl")):
        name = path.stem.removeprefix("records_")
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not recs:
            continue
        systems = ["resnet_crop", "resnet_full"] + sorted({k for r in recs for k in r if k.startswith("landmark_")})
        n = len(recs)
        detected = [r for r in recs if r["detected"]]
        lines += [f"## Set: {name}  ({n} frames, {len({r['letter'] for r in recs})} letters)", ""]
        fails = n - len(detected)
        lo, hi = wilson(fails, n)
        lines += [f"**Landmark/hand detection failure rate: {fails}/{n} = {100 * fails / n:.1f}%** (95% CI {100 * lo:.0f}-{100 * hi:.0f}%). "
                  "Both `resnet_crop` and every landmark model produce NO answer on these frames.", ""]
        unusable = sum(1 for r in recs for k in systems if (r.get(k) or {}).get("unusable"))
        if unusable:
            lines += [f"Detected but landmarks unusable (degenerate/non-finite): {unusable} (counted as no answer).", ""]

        # ---- headline table
        lines += ["| system | end-to-end accuracy (all frames) | accuracy when it answered | answered | accepted (conf>=%.2f) | accuracy among accepted | wrong & accepted |" % CONFIDENCE_THRESHOLD,
                  "|---|---|---|---|---|---|---|"]
        correct: dict[str, list[bool]] = {}
        for s in systems:
            ans = [(r, answer(r, s)) for r in recs]
            ok = [bool(a and a[0] == r["letter"]) for r, a in ans]
            correct[s] = ok
            answered = [(r, a) for r, a in ans if a]
            acc = [(r, a) for r, a in answered if a[1] >= CONFIDENCE_THRESHOLD]
            wrong_acc = sum(1 for r, a in acc if a[0] != r["letter"])
            lines.append(f"| {s} | {pct(sum(ok), n)} | {pct(sum(a[0] == r['letter'] for r, a in answered), len(answered))} | {len(answered)}/{n} | "
                         f"{len(acc)}/{n} | {pct(sum(a[0] == r['letter'] for r, a in acc), len(acc))} | {wrong_acc}/{n} ({100 * wrong_acc / n:.1f}%) |")
        lines += ["", "95% Wilson CIs in parentheses. Frames within one class are consecutive captures of one person/session, so the real uncertainty is wider than the CI.", ""]

        # ---- paired comparison vs the production path
        lines += ["Paired, end-to-end, vs `resnet_crop` (same frames): `only ResNet right` / `only other right` / exact two-sided binomial p", ""]
        for s in systems:
            if s == "resnet_crop":
                continue
            a_only = sum(1 for x, y in zip(correct["resnet_crop"], correct[s]) if x and not y)
            b_only = sum(1 for x, y in zip(correct["resnet_crop"], correct[s]) if y and not x)
            p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
            lines.append(f"- {s}: {a_only} / {b_only} / p={p:.2g}")
        lines.append("")

        # ---- per letter (J and Z called out)
        lines += ["### Per-letter recall (end-to-end; a missed hand counts as wrong) and detection rate", "",
                  "| letter | n | detected | " + " | ".join(systems) + " |", "|---|---|---|" + "---|" * len(systems)]
        for letter in LETTERS:
            sub = [i for i, r in enumerate(recs) if r["letter"] == letter]
            if not sub:
                continue
            det = sum(recs[i]["detected"] for i in sub)
            mark = " **(motion letter)**" if letter in "JZ" else ""
            lines.append(f"| {letter}{mark} | {len(sub)} | {100 * det / len(sub):.0f}% | " +
                         " | ".join(f"{100 * sum(correct[s][i] for i in sub) / len(sub):.0f}%" for s in systems) + " |")
        present = [l for l in LETTERS if any(r["letter"] == l for r in recs)]
        lines += ["", "Macro recall over letters present: " + ", ".join(
            f"{s} {100 * np.mean([np.mean([correct[s][i] for i, r in enumerate(recs) if r['letter'] == l]) for l in present]):.1f}%" for s in systems), ""]
        for letter in ("J", "Z"):
            if letter in present:
                for s in systems:
                    wrong = Counter(answer(r, s)[0] for r in recs if r["letter"] == letter and answer(r, s) and answer(r, s)[0] != letter)
                    lines.append(f"- {letter} / {s}: most common wrong answers {dict(wrong.most_common(3))}")
        lines.append("")

        # ---- conditions
        def breakdown(title: str, key) -> None:
            groups: dict[str, list[int]] = defaultdict(list)
            for i, r in enumerate(recs):
                g = key(r)
                if g is not None:
                    groups[g].append(i)
            if len(groups) < 2:
                return
            lines.extend(["", f"### By {title}", "", "| group | n | detected | " + " | ".join(systems) + " |", "|---|---|---|" + "---|" * len(systems)])
            for g in sorted(groups):
                idx = groups[g]
                lines.append(f"| {g}{' (n<20, indicative only)' if len(idx) < 20 else ''} | {len(idx)} | {100 * sum(recs[i]['detected'] for i in idx) / len(idx):.0f}% | " +
                             " | ".join(pct(sum(correct[s][i] for i in idx), len(idx), ci=False) for s in systems) + " |")

        b = [r["brightness"] for r in recs]
        band = lambda r: "dark (<90)" if r["brightness"] < 90 else "bright (>180)" if r["brightness"] > 180 else "normal (90-180)"
        lines += ["", f"_Measured frame brightness (0-255): min {min(b):.0f}, median {statistics.median(b):.0f}, max {max(b):.0f}. Bands are absolute, not quantiles; a set with one band has NO lighting variation to learn from._"]
        breakdown("measured lighting (absolute brightness band)", band)
        for k in ("lighting", "background", "distance", "hand", "room"):
            breakdown(f"recorded {k}", lambda r, k=k: (r["meta"].get(k) if r["meta"] else None))
        sizes = [r["hand"]["size"] for r in detected]
        if sizes:
            s1, s2 = quant(sizes, 1 / 3), quant(sizes, 2 / 3)
            lines += ["", "_Hand-position breakdowns exist only for frames where a hand was found (position comes from the landmarks), so they exclude the detection failures counted above._"]
            tri = lambda v, a, b_: "low" if v < a else "mid" if v < b_ else "high"
            sub = [r for r in recs if r["detected"]]
            idx_of = {id(r): i for i, r in enumerate(recs)}
            def breakdown_det(title, key):
                groups = defaultdict(list)
                for r in sub:
                    groups[key(r)].append(idx_of[id(r)])
                if len(groups) < 2:
                    return
                lines.extend(["", f"### By {title} (detected frames only)", "", "| group | n | " + " | ".join(systems) + " |", "|---|---|" + "---|" * len(systems)])
                for g in sorted(groups):
                    idx = groups[g]
                    lines.append(f"| {g}{' (n<20, indicative only)' if len(idx) < 20 else ''} | {len(idx)} | " + " | ".join(pct(sum(correct[s][i] for i in idx), len(idx), ci=False) for s in systems) + " |")
            breakdown_det("hand position, horizontal (image left/centre/right thirds)", lambda r: "left" if r["hand"]["cx"] < 1 / 3 else "right" if r["hand"]["cx"] > 2 / 3 else "centre")
            breakdown_det("hand position, vertical (top/middle/bottom thirds)", lambda r: "top" if r["hand"]["cy"] < 1 / 3 else "bottom" if r["hand"]["cy"] > 2 / 3 else "middle")
            breakdown_det(f"hand size in frame (terciles: small <{s1:.2f}, medium <{s2:.2f}, large)", lambda r: {"low": "small", "mid": "medium", "high": "large"}[tri(r["hand"]["size"], s1, s2)])
            breakdown_det("MediaPipe handedness (raw label)", lambda r: r["hand"]["handedness"] or "unknown")

        # ---- detection failures by condition
        lines += ["", "### Where detection fails", ""]
        by_letter = {l: sum(1 for r in recs if r["letter"] == l and not r["detected"]) for l in present}
        worst = sorted(by_letter.items(), key=lambda kv: -kv[1])[:8]
        lines.append("Most-missed letters (frames with no hand found): " + ", ".join(f"{l} {c}/{sum(1 for r in recs if r['letter'] == l)}" for l, c in worst))

        # ---- latency
        lines += ["", "### Latency (ms, median / p95, batch 1, this machine)", "",
                  f"- MediaPipe detection (needed by resnet_crop AND all landmark models): {lat([r['detect_ms'] for r in recs])}",
                  f"- crop step: {lat([r['crop_ms'] for r in recs if 'crop_ms' in r])}",
                  f"- ResNet18 forward on GPU ({'cuda' if True else ''}): {lat([r['resnet_crop']['ms'] for r in recs if r.get('resnet_crop')])}"]
        cpu_file = OUT / f"cpu_latency_{name}.json"
        if cpu_file.exists():
            cpu = json.loads(cpu_file.read_text())["ms"]
            lines.append(f"- ResNet18 forward on CPU (n={len(cpu)}): {lat(cpu)}")
        for s in systems:
            if s.startswith("landmark_"):
                lines.append(f"- {s} classify (features + sklearn): {lat([r[s]['ms'] for r in recs if r.get(s) and 'ms' in r[s]])}")
        det_ms = statistics.median([r["detect_ms"] for r in recs])
        lines.append(f"- so end-to-end median ~ detection {det_ms:.0f} ms + model; detection dominates both pipelines.")
        lines.append("")
        summary[name] = {"n": n, "detection_failure_rate": fails / n,
                         "accuracy_end_to_end": {s: sum(correct[s]) / n for s in systems}}
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--sets", nargs="+", default=["dataset", "webcam"], choices=["dataset", "webcam"])
    run.add_argument("--resnet", type=Path, default=ROOT / "models" / "best_model.pth")
    run.add_argument("--landmarks", nargs="+", type=Path, default=[ROOT / p for p in DEFAULT_LANDMARKS])
    run.add_argument("--limit-per-class", type=int, default=None, help="dataset set only: evenly sample N test images per class (default: all 450)")
    run.add_argument("--device", default="cuda")
    run.add_argument("--resume", action="store_true")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "report":
        report()
        return
    for name in args.sets:
        items = dataset_items(args.limit_per_class) if name == "dataset" else webcam_items()
        print(f"[{name}] {len(items)} frames; letters: {''.join(sorted({i['letter'] for i in items}))}", flush=True)
        run_set(name, items, args.resnet, args.landmarks, args.device, args.resume)


if __name__ == "__main__":
    main()
