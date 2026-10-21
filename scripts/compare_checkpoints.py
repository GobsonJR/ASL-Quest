"""Compare two A-Z checkpoints on EXACTLY the same held-out images.

Reports, per checkpoint: overall accuracy, macro precision/recall/F1, per-class P/R/F1, the
confusion row and column of one letter (default C), and accuracy under realistic input changes
(mirror, rotation, hand size, JPEG, lighting, blur, noise). Both models see identical images and
identical perturbations. Nothing here replaces the production checkpoint.

    venv311\\Scripts\\python.exe scripts\\compare_checkpoints.py ^
        --old models\\best_model.pth --new models\\experiments\\az_robust_best.pth --letter C
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_c_experiments import LETTERS, PERTURBATIONS, test_images  # noqa: E402
from src.config import DATASET_DIR, OUTPUTS_DIR, SPLITS_PATH  # noqa: E402
from src.data.preprocessing import eval_transforms  # noqa: E402
from src.models.resnet18 import build_resnet18  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402


class Net:
    def __init__(self, path: Path, device: str):
        payload = load_checkpoint(path, map_location=device)
        self.classes = payload["class_names"]
        assert self.classes == LETTERS, f"{path.name}: class order differs from A-Z"
        self.net = build_resnet18(num_classes=len(self.classes), pretrained=False)
        self.net.load_state_dict(payload["model_state_dict"])
        self.net.to(device).eval()
        self.device, self.transform = device, eval_transforms()
        self.meta = {k: payload.get(k) for k in ("epoch", "val_accuracy", "augment", "seed", "training_policy")}

    @torch.no_grad()
    def predict(self, images: list[Image.Image], batch: int = 32) -> np.ndarray:
        out = []
        for s in range(0, len(images), batch):
            x = torch.stack([self.transform(im.convert("RGB")) for im in images[s : s + batch]]).to(self.device)
            out.append(self.net(x).argmax(1).cpu())
        return torch.cat(out).numpy()


def all_test_paths() -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for k, letter in enumerate(LETTERS):
        items.extend((p, k) for p in test_images(letter, None))
    return items


def prf_from_cm(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return precision, recall, f1


def evaluate(net: Net, test: list[tuple[Path, int]], perturb_n: int, letter: str) -> dict:
    y_true = np.array([k for _, k in test])
    y_pred = np.concatenate([net.predict([Image.open(p) for p, _ in test[s : s + 256]]) for s in range(0, len(test), 256)])
    cm = np.zeros((26, 26), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    precision, recall, f1 = prf_from_cm(cm)
    k = LETTERS.index(letter)
    result = {
        "meta": net.meta,
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])} for i, c in enumerate(LETTERS)},
        "letter_row": {LETTERS[j]: int(cm[k, j]) for j in range(26) if j != k and cm[k, j]},
        "letter_column": {LETTERS[i]: int(cm[i, k]) for i in range(26) if i != k and cm[i, k]},
        "confusion_matrix": cm.tolist(),
        "perturbations": {},
    }
    # fixed subset: the first `perturb_n` held-out images of each letter (identical for every model)
    subset = [(Image.open(p).convert("RGB"), k_) for k_, c in enumerate(LETTERS) for p in test_images(c, perturb_n)]
    truth = np.array([k_ for _, k_ in subset])
    for name, fn in PERTURBATIONS.items():
        pred = net.predict([fn(im) for im, _ in subset])
        per_letter = {c: float((pred[truth == i] == i).mean()) for i, c in enumerate(LETTERS)}
        result["perturbations"][name] = {"mean": float(np.mean(list(per_letter.values()))), "per_letter": per_letter}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--letter", default="C")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--perturb-n", type=int, default=20)
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "diagnostics" / "checkpoint_comparison.json")
    args = ap.parse_args()
    letter = args.letter.upper()

    test = all_test_paths()
    print(f"held-out test images: {len(test)} (identical for both checkpoints), device={args.device}")
    results = {}
    for tag, path in (("old", args.old), ("new", args.new)):
        print(f"evaluating {tag}: {path}", flush=True)
        results[tag] = evaluate(Net(path, args.device), test, args.perturb_n, letter)
        results[tag]["checkpoint"] = str(path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")

    o, n = results["old"], results["new"]
    print(f"\n{'':34s} {'old':>9s} {'new':>9s} {'change':>9s}")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"{key:34s} {100 * o[key]:8.2f}% {100 * n[key]:8.2f}% {100 * (n[key] - o[key]):+8.2f}")
    print(f"\nper-class (precision / recall / f1), letters whose recall changed or that matter most:")
    interesting = sorted({letter, "W", "O", "G", "S", "X", "V", "Y"} | {c for c in LETTERS if abs(n['per_class'][c]['recall'] - o['per_class'][c]['recall']) > 0.02})
    for c in interesting:
        po, pn = o["per_class"][c], n["per_class"][c]
        print(f"{c}  old {po['precision']:.3f}/{po['recall']:.3f}/{po['f1']:.3f}   new {pn['precision']:.3f}/{pn['recall']:.3f}/{pn['f1']:.3f}")
    print(f"\n{letter} confusions (true {letter} -> predicted): old {o['letter_row'] or 'none'}  new {n['letter_row'] or 'none'}")
    print(f"{letter} column (other true -> predicted {letter}): old {o['letter_column'] or 'none'}  new {n['letter_column'] or 'none'}")
    print(f"\nrobustness, mean accuracy over letters ({args.perturb_n} held-out images per letter, same images for both):")
    print(f"{'perturbation':14s} {'old':>8s} {'new':>8s}   {letter} old -> new")
    for name in PERTURBATIONS:
        po, pn = o["perturbations"][name], n["perturbations"][name]
        print(f"{name:14s} {100 * po['mean']:7.0f}% {100 * pn['mean']:7.0f}%   {100 * po['per_letter'][letter]:4.0f}% -> {100 * pn['per_letter'][letter]:4.0f}%")
    print(f"\nfull results: {args.out}")


if __name__ == "__main__":
    main()
