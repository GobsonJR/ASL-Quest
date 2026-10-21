"""Evaluate a recognizer on REAL webcam crops from rooms it never saw (the exported `test` split).

Two approaches use the same rows and the same metrics, so they can be compared fairly:

  --approach resnet     the image classifier (a .pth checkpoint) on the exported hand crops
  --approach landmarks  the landmark classifier (.joblib) on the stored MediaPipe landmarks

    venv311\\Scripts\\python.exe scripts\\evaluate_webcam.py --approach resnet --checkpoint models\\best_model.pth --name resnet_current
    venv311\\Scripts\\python.exe scripts\\evaluate_webcam.py --approach landmarks --model models\\landmark_classifier.joblib --name landmarks
    venv311\\Scripts\\python.exe scripts\\compare_approaches.py outputs\\evaluation_webcam\\resnet_current.json outputs\\evaluation_webcam\\landmarks.json

Reported: accuracy and macro-F1, per-letter precision/recall/F1, the confusion matrix, how often a prediction is rejected as
uncertain (confidence below the app's threshold) and the accuracy among the accepted ones, accuracy per room / background /
hand / distance / lighting / face visible, prediction stability on recorded hold sequences (how soon the app's own
confirmation rule fires and whether it ever confirms a WRONG letter), and latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIDENCE_THRESHOLD  # noqa: E402
from src.inference.stability import LOW_CONFIDENCE, RECOGNIZED, StableLetterTracker  # noqa: E402

LETTERS = [chr(65 + i) for i in range(26)]
ROOT = Path(__file__).resolve().parent.parent
HOLD_FRAME_SECONDS = 0.3  # spacing used by the collector's hold capture


def thin_hold(rows: list[dict], keep: int = 3) -> list[dict]:
    """Singles + a few evenly spaced frames per hold sequence, so near-identical frames don't dominate accuracy."""
    out, seqs = [], defaultdict(list)
    for r in rows:
        (seqs[r["sequence_id"]] if r["kind"] == "hold" else out).append(r)
    for seq in seqs.values():
        seq.sort(key=lambda r: r.get("frame_index") or 0)
        idx = sorted(set(np.linspace(0, len(seq) - 1, min(keep, len(seq))).round().astype(int).tolist()))
        out.extend(seq[i] for i in idx)
    return out


def predict_resnet(rows: list[dict], export: Path, checkpoint: Path, device: str):
    import torch
    from PIL import Image

    from src.data.preprocessing import eval_transforms
    from src.models.resnet18 import build_resnet18
    from src.utils.checkpoint import load_checkpoint

    payload = load_checkpoint(checkpoint, map_location=device)
    assert payload["class_names"] == LETTERS
    net = build_resnet18(num_classes=26, pretrained=False)
    net.load_state_dict(payload["model_state_dict"])
    net.to(device).eval()
    transform = eval_transforms()
    preds, latencies = [], []
    with torch.no_grad():
        for i, row in enumerate(rows):
            t0 = time.perf_counter()
            x = transform(Image.open(export / row["crop"]).convert("RGB")).unsqueeze(0).to(device)
            probs = torch.softmax(net(x), dim=1)[0].cpu()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            if i >= 5:
                latencies.append((time.perf_counter() - t0) * 1000)
            conf, idx = probs.topk(5)
            preds.append([(LETTERS[j], float(c)) for j, c in zip(idx.tolist(), conf.tolist())])
    return preds, latencies


def predict_landmarks(rows: list[dict], model_path: Path, hand_source: str):
    from src.landmarks.classifier import LandmarkClassifier

    clf = LandmarkClassifier(model_path)
    from src.landmarks.classifier import MEDIAPIPE_TO_ACTUAL_HAND as mp_hand  # the same mapping training and the live app use
    preds, latencies = [], []
    for row in rows:
        if not row.get("landmarks"):
            preds.append(None)
            continue
        hand = row["hand"] if hand_source == "label" else mp_hand.get(row.get("handedness_raw") or "", None)
        t0 = time.perf_counter()
        reply = clf.predict_points(row["landmarks"], hand, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000)
        preds.append([(t["label"], t["confidence"]) for t in reply["top_predictions"]])
    return preds, latencies


def metrics_for(rows: list[dict], preds: list) -> dict:
    n = len(LETTERS)
    cm = np.zeros((n, n), dtype=int)
    accepted = accepted_correct = rejected = no_prediction = 0
    wrong: dict[str, Counter] = defaultdict(Counter)
    for row, pred in zip(rows, preds):
        t = LETTERS.index(row["letter"])
        if pred is None:
            no_prediction += 1
            continue
        label, conf = pred[0]
        cm[t, LETTERS.index(label)] += 1
        if conf >= CONFIDENCE_THRESHOLD:
            accepted += 1
            accepted_correct += label == row["letter"]
        else:
            rejected += 1
        if label != row["letter"]:
            wrong[row["letter"]][label] += 1
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(n), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(n), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) > 0)
    present = [i for i in range(n) if cm[i].sum() > 0]
    scored = int(cm.sum())
    return {
        "n_rows": len(rows), "n_scored": scored, "no_prediction": no_prediction,
        "accuracy": float(tp.sum() / scored) if scored else None,
        "macro_f1_over_present_letters": float(np.mean([f1[i] for i in present])) if present else None,
        "letters_present": len(present),
        "accepted": accepted, "rejected_uncertain": rejected,
        "reject_rate": rejected / scored if scored else None,
        "accuracy_among_accepted": accepted_correct / accepted if accepted else None,
        "per_letter": {LETTERS[i]: {"support": int(cm[i].sum()), "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                                    "top_wrong": dict(wrong[LETTERS[i]].most_common(3))} for i in range(n)},
        "confusion_matrix": cm.tolist(),
    }


def grouped_accuracy(rows: list[dict], preds: list, key: str) -> dict:
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row, pred in zip(rows, preds):
        g = str(row.get(key))
        groups[g][1] += 1
        if pred is not None and pred[0][0] == row["letter"]:
            groups[g][0] += 1
    return {g: {"accuracy": c / n, "n": n} for g, (c, n) in sorted(groups.items())}


def stability(rows: list[dict], preds: list) -> dict:
    """Run the app's own confirmation rule over each recorded hold sequence."""
    seqs: dict[str, list[tuple[dict, list]]] = defaultdict(list)
    for row, pred in zip(rows, preds):
        if row["kind"] == "hold":
            seqs[row["sequence_id"]].append((row, pred))
    results = []
    for frames in seqs.values():
        frames.sort(key=lambda fp: fp[0].get("frame_index") or 0)
        tracker, confirmed, at = StableLetterTracker(), None, None
        for i, (row, pred) in enumerate(frames):
            state = LOW_CONFIDENCE if pred is None or pred[0][1] < CONFIDENCE_THRESHOLD else RECOGNIZED
            result = tracker.update(state, pred[0][0] if pred else None)
            if result.stable:
                confirmed, at = result.stable, i + 1
                break
        results.append((frames[0][0]["letter"], confirmed, at, len(frames)))
    n = len(results)
    if not n:
        return {"sequences": 0}
    correct = [r for r in results if r[1] == r[0]]
    false = [r for r in results if r[1] and r[1] != r[0]]
    never = [r for r in results if r[1] is None]
    return {
        "sequences": n,
        "confirmed_correct": len(correct) / n, "confirmed_wrong": len(false) / n, "never_confirmed": len(never) / n,
        "median_frames_to_confirm": statistics.median(r[2] for r in correct) if correct else None,
        "median_seconds_to_confirm": statistics.median(r[2] for r in correct) * HOLD_FRAME_SECONDS if correct else None,
        "wrong_confirmations": {f"{truth}->{shown}": n for (truth, shown), n in Counter((r[0], r[1]) for r in false).most_common(5)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, default=ROOT / "dataset" / "webcam_export")
    ap.add_argument("--split", default="test")
    ap.add_argument("--approach", choices=["resnet", "landmarks"], required=True)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "best_model.pth")
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "landmark_classifier.joblib")
    ap.add_argument("--hand-source", choices=["mediapipe", "label"], default="mediapipe", help="landmarks: use MediaPipe's handedness (as deployed) or the person's label")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "evaluation_webcam")
    args = ap.parse_args()

    manifest = [json.loads(line) for line in (args.export / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in manifest if r["split"] == args.split]
    if not rows:
        sys.exit(f"no rows in split {args.split!r}")
    if args.split != "test":
        print(f"NOTE: evaluating the {args.split!r} split, not the held-out test rooms")
    if args.approach == "resnet":
        preds, latencies = predict_resnet(rows, args.export, args.checkpoint, args.device)
        source = str(args.checkpoint.resolve())
    else:
        preds, latencies = predict_landmarks(rows, args.model, args.hand_source)
        source = str(args.model.resolve())

    keep_ids = {id(r) for r in thin_hold(rows)}
    scored_idx = [i for i, r in enumerate(rows) if id(r) in keep_ids]
    sel_rows = [rows[i] for i in scored_idx]
    sel_preds = [preds[i] for i in scored_idx]
    result = {
        "name": args.name or args.approach, "approach": args.approach, "source": source, "split": args.split,
        "rooms": sorted({r["room"] for r in rows}), "threshold": CONFIDENCE_THRESHOLD,
        "overall": metrics_for(sel_rows, sel_preds),
        "by_room": grouped_accuracy(sel_rows, sel_preds, "room"),
        "by_background": grouped_accuracy(sel_rows, sel_preds, "background"),
        "by_hand": grouped_accuracy(sel_rows, sel_preds, "hand"),
        "by_distance": grouped_accuracy(sel_rows, sel_preds, "distance"),
        "by_lighting": grouped_accuracy(sel_rows, sel_preds, "lighting"),
        "by_face_visible": grouped_accuracy(sel_rows, sel_preds, "face_visible"),
        "stability": stability(rows, preds),
        "latency_ms": {"median": statistics.median(latencies), "p95": float(np.percentile(latencies, 95)), "device": args.device if args.approach == "resnet" else "cpu"} if latencies else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{result['name']}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")

    o = result["overall"]
    print(f"{result['name']}  ({args.approach}) on {len(rows)} crops from rooms {result['rooms']} ({o['n_scored']} scored frames, {o['letters_present']} letters)")
    print(f"accuracy {100 * o['accuracy']:.1f}%   macro-F1 {100 * o['macro_f1_over_present_letters']:.1f}%   rejected as uncertain {100 * o['reject_rate']:.0f}%   accuracy among accepted {100 * (o['accuracy_among_accepted'] or 0):.1f}%")
    print("per letter recall:", " ".join(f"{l}:{100 * v['recall']:.0f}" for l, v in o["per_letter"].items() if v["support"]))
    print("stability:", result["stability"])
    if result["latency_ms"]:
        print("latency:", result["latency_ms"])
    print(f"saved {path}")


if __name__ == "__main__":
    main()
