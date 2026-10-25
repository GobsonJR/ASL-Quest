"""READ-ONLY investigation of the landmark recognizer's M / T -> Z confusion (model landmark_L2_s0.joblib).

Nothing is trained, saved into models/, or promoted. Outputs go to outputs/mtz_investigation/.

    venv311\\Scripts\\python.exe scripts\\investigate_mtz.py

Questions answered
  A  what the model was trained on (per-letter counts, hand labels, how few webcam rooms/signers/letters)
  B  held-out confusion matrix + per-class precision/recall (Kaggle test split, the only held-out set containing M, T, Z)
  C  is the confidence calibrated (reliability table, and how often a confident "Z" is right)
  D  identical landmarks with hand = left / right / unknown: does the answer change?
  E  how sensitive M / T / Z are to in-plane rotation, scale and the z coordinate (the model was fitted with rotate=False)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CONFIDENCE_THRESHOLD  # noqa: E402
from src.landmarks.classifier import LETTERS, LandmarkClassifier, make_xy  # noqa: E402
from src.landmarks.features import feature_vector  # noqa: E402

OUT = ROOT / "outputs" / "mtz_investigation"
MODEL = ROOT / "models" / "experiments" / "landmark_L2_s0.joblib"
DATASET_NPZ = ROOT / "outputs" / "diagnostics" / "landmarks" / "dataset_landmarks.npz"
RECORDS = ROOT / "outputs" / "compare_resnet_landmarks" / "records_dataset.jsonl"
KEY = ("M", "T", "Z")


def predict(clf: LandmarkClassifier, pts: np.ndarray, hand: str | None) -> tuple[str, float, np.ndarray]:
    probs = clf.pipeline.predict_proba(feature_vector(pts, hand, rotate=clf.rotate)[None, :])[0]
    i = int(np.argmax(probs))
    return LETTERS[int(clf.pipeline.classes_[i])], float(probs[i]), probs


def rotate_xy(pts: np.ndarray, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    out = pts.copy()
    rel = pts[:, :2] - pts[0, :2]
    out[:, 0] = pts[0, 0] + c * rel[:, 0] - s * rel[:, 1]
    out[:, 1] = pts[0, 1] + s * rel[:, 0] + c * rel[:, 1]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clf = LandmarkClassifier(MODEL)
    lines: list[str] = ["# M / T -> Z investigation (landmark_L2_s0, read-only)", ""]
    result: dict = {}

    # ---------------------------------------------------------------- A: training data
    data = np.load(DATASET_NPZ, allow_pickle=False)
    letters, hands, lms = data["letters"], data["hands"], data["landmarks"]
    counts = Counter(letters.tolist())
    lines += ["## A. Training data", "",
              "L2 was fitted on 9,878 rows: 9,458 landmark sets extracted from the ORIGINAL Kaggle photos (one signer, 200x200 close-ups, "
              "MediaPipe found a hand in only ~1/3 of them) plus 420 real webcam frames. Webcam letters present in training: "
              "A, B, C, F, U, V, W, X, Y. **M, N, T and Z (and 13 other letters) have ZERO webcam training examples.**", "",
              "| letter | Kaggle landmark rows | share labelled hand=left | webcam rows |", "|---|---|---|---|"]
    webcam_letters = {"A", "B", "C", "F", "U", "V", "W", "X", "Y"}
    for l in KEY + ("N", "S", "E"):
        m = letters == l
        lines.append(f"| {l} | {counts[l]} | {100 * np.mean(hands[m] == 'left'):.0f}% | {'yes' if l in webcam_letters else '**0**'} |")
    lines += ["", "Model selection used leave-block-out CV (macro-F1 0.94), but the 'rooms' were `original-block0..4`: five chunks of the "
              "same signer and setup. That score measures the same person in the same photo studio, not new signers/rooms.", ""]
    result["training_counts"] = {l: counts[l] for l in LETTERS}

    # ---------------------------------------------------------------- B: held-out confusion matrix
    recs = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    key = "landmark_L2_s0"
    det = [r for r in recs if r.get(key) and "top" in r[key]]
    idx = {l: i for i, l in enumerate(LETTERS)}
    cm = np.zeros((26, 26), dtype=int)
    for r in det:
        cm[idx[r["letter"]], idx[r[key]["top"][0][0]]] += 1
    n_all = Counter(r["letter"] for r in recs)
    lines += ["## B. Held-out confusion (Kaggle test split, last 15% of each class; detected frames only)", "",
              f"{len(det)} of {len(recs)} test images had a detectable hand. **Only these frames reach the classifier**, so M/T are judged on the "
              "minority of fist-like images MediaPipe could find.", "",
              "| letter | test images | hand found | recall | precision | most common wrong answers |", "|---|---|---|---|---|---|"]
    prf = {}
    for l in LETTERS:
        i = idx[l]
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        rec = tp / support if support else float("nan")
        prec = tp / predicted if predicted else float("nan")
        wrong = {LETTERS[j]: int(cm[i, j]) for j in np.argsort(cm[i])[::-1][:3] if j != i and cm[i, j]}
        prf[l] = {"support": int(support), "recall": rec, "precision": prec}
        if l in KEY or l in ("N", "S", "E"):
            lines.append(f"| {l} | {n_all[l]} | {support} | {100 * rec:.1f}% | {100 * prec:.1f}% | {wrong} |")
    sub = [l for l in KEY]
    lines += ["", "Confusion sub-matrix (rows = true, columns = predicted; `other` = any other letter):", "",
              "| true \\ pred | M | T | Z | other |", "|---|---|---|---|---|"]
    for l in sub:
        i = idx[l]
        other = int(cm[i].sum() - sum(cm[i, idx[p]] for p in sub))
        lines.append(f"| {l} | " + " | ".join(str(int(cm[i, idx[p]])) for p in sub) + f" | {other} |")
    lines += ["", "Frames of OTHER letters that were answered as M / T / Z: " +
              ", ".join(f"{p}: {int(cm[:, idx[p]].sum() - cm[idx[p], idx[p]])}" for p in sub), ""]
    result["per_class"] = prf
    np.save(OUT / "confusion_L2_dataset_test.npy", cm)

    # ---------------------------------------------------------------- C: calibration
    conf = np.array([r[key]["top"][0][1] for r in det])
    ok = np.array([r[key]["top"][0][0] == r["letter"] for r in det])
    edges = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
    lines += ["## C. Is the confidence calibrated?", "",
              "LogisticRegression.predict_proba is a softmax of linear scores fitted on the TRAINING distribution; it says how far a sample is "
              "from the decision boundary, not how often such answers are right on new people. Held-out Kaggle frames (same signer, "
              "so the most favourable case):", "",
              "| confidence | n | actual accuracy |", "|---|---|---|"]
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            lines.append(f"| {lo:.2f}-{min(hi, 1):.2f} | {int(m.sum())} | {100 * ok[m].mean():.1f}% |")
            ece += m.sum() / len(conf) * abs(ok[m].mean() - conf[m].mean())
    lines += ["", f"Expected calibration error on this set: {ece:.3f}. Threshold used by the app: {CONFIDENCE_THRESHOLD:.2f}.", ""]
    z_conf = [(c, r["letter"]) for r, c in zip(det, conf) if r[key]["top"][0][0] == "Z" and c >= 0.8]
    lines += [f"Answers 'Z' with confidence >= 0.80 on held-out frames: {len(z_conf)}; truly Z: {sum(1 for _, t in z_conf if t == 'Z')}; "
              f"truly M: {sum(1 for _, t in z_conf if t == 'M')}; truly T: {sum(1 for _, t in z_conf if t == 'T')}.", ""]
    result["ece_dataset"] = float(ece)

    # ---------------------------------------------------------------- D: Left vs Right with identical coordinates
    lines += ["## D. Identical landmarks, different handedness label", "",
              "Training landmark sets of M / T / Z (not held out; this isolates the label's effect), each classified with hand=left, right, and unknown:", "",
              "| letter | n | acc (left) | acc (right) | acc (unknown) | prediction changes left vs right | Z answers (left / right) |", "|---|---|---|---|---|---|---|"]
    side = {}
    for l in KEY:
        pts_l = lms[letters == l]
        preds = {h: [predict(clf, p, h)[0] for p in pts_l] for h in ("left", "right", None)}
        acc = {h: float(np.mean([x == l for x in v])) for h, v in preds.items()}
        changed = float(np.mean([a != b for a, b in zip(preds["left"], preds["right"])]))
        side[l] = {"acc": {str(k): v for k, v in acc.items()}, "changed": changed}
        lines.append(f"| {l} | {len(pts_l)} | {100 * acc['left']:.1f}% | {100 * acc['right']:.1f}% | {100 * acc[None]:.1f}% | {100 * changed:.1f}% | "
                     f"{sum(x == 'Z' for x in preds['left'])} / {sum(x == 'Z' for x in preds['right'])} |")
    lines += ["", "Hand label is used for one thing only: `hand == 'left'` mirrors x. So a label flip is the same as mirroring the whole hand; "
              "it changes the answer exactly when the classifier is not mirror-symmetric for that pose.", ""]
    result["handedness"] = side

    # ---------------------------------------------------------------- E: sensitivity to rotation / scale / z
    lines += ["## E. Sensitivity (training M/T/Z landmark sets, hand label as extracted)", "",
              "Accuracy after turning the hand in the image plane about the wrist (the model was fitted with rotate=False, i.e. it expects the "
              "photo-studio orientation), and after dropping depth:", "",
              "| letter | 0 deg | +15 | -15 | +30 | -30 | +45 | -45 | z = 0 | answered Z at +30 |", "|---|---|---|---|---|---|---|---|---|---|"]
    sens = {}
    for l in KEY:
        m = letters == l
        pts_l, hs = lms[m], hands[m]
        row = {}
        for name, fn in [("0", lambda p: p)] + [(str(d), (lambda p, d=d: rotate_xy(p, d))) for d in (15, -15, 30, -30, 45, -45)] + \
                        [("z0", lambda p: np.column_stack([p[:, :2], np.zeros(21)]))]:
            row[name] = float(np.mean([predict(clf, fn(p), h)[0] == l for p, h in zip(pts_l, hs)]))
        zrate = float(np.mean([predict(clf, rotate_xy(p, 30), h)[0] == "Z" for p, h in zip(pts_l, hs)]))
        sens[l] = {**row, "z_at_30": zrate}
        lines.append(f"| {l} | " + " | ".join(f"{100 * row[k]:.0f}%" for k in ("0", "15", "-15", "30", "-30", "45", "-45", "z0")) + f" | {100 * zrate:.0f}% |")
    result["sensitivity"] = sens

    # what the classifier sees: where do M/T training samples sit relative to Z?
    x, y, _ = make_xy([{"landmarks": p.tolist(), "hand": h, "letter": l, "room": "o"} for p, h, l in zip(lms, hands, letters)], rotate=clf.rotate)
    scaled = clf.pipeline[0].transform(x)
    cent = {l: scaled[y == idx[l]].mean(axis=0) for l in KEY}
    lines += ["", "Distance between class centroids in the model's standardised feature space: "
              f"M-Z {np.linalg.norm(cent['M'] - cent['Z']):.2f}, T-Z {np.linalg.norm(cent['T'] - cent['Z']):.2f}, M-T {np.linalg.norm(cent['M'] - cent['T']):.2f}.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
