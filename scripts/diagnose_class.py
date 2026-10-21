"""Class-specific diagnosis of the deployed checkpoint on the controlled held-out split.

Answers, for one letter (default C): how many examples exist per split, precision/recall/F1,
its confusion-matrix row and column, what it is confused with, whether labels/duplicates/leakage
look wrong, and how it compares with similar letters. Everything is computed from the real
checkpoint and dataset (no numbers are hand-entered).

    venv311\\Scripts\\python.exe scripts\\diagnose_class.py --letter C --compare W O G S X V Y
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BEST_MODEL_PATH, DATASET_DIR, OUTPUTS_DIR, SPLITS_PATH  # noqa: E402
from src.data.dataset import load_image_folder, load_split_indices  # noqa: E402
from src.data.preprocessing import eval_transforms  # noqa: E402
from src.models.resnet18 import build_resnet18  # noqa: E402
from src.utils.checkpoint import load_checkpoint  # noqa: E402
from src.utils.device import get_device  # noqa: E402


class PathDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), label


def load_model(path: Path, device):
    payload = load_checkpoint(path, map_location=device)
    model = build_resnet18(num_classes=len(payload["class_names"]), pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), payload["class_names"]


@torch.no_grad()
def predict_split(model, samples, device, batch_size=16):
    loader = DataLoader(PathDataset(samples, eval_transforms()), batch_size=batch_size, num_workers=0)
    probs = []
    for images, _ in loader:
        probs.append(torch.softmax(model(images.to(device)), dim=1).cpu())
    return torch.cat(probs).numpy()


def prf(cm: np.ndarray, k: int) -> tuple[float, float, float]:
    tp = cm[k, k]
    precision = tp / cm[:, k].sum() if cm[:, k].sum() else 0.0
    recall = tp / cm[k, :].sum() if cm[k, :].sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--letter", default="C")
    ap.add_argument("--compare", nargs="*", default=["W", "O", "G", "S", "X", "V", "Y"])
    ap.add_argument("--model", type=Path, default=BEST_MODEL_PATH)
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "diagnostics" / "class_analysis")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto). Use cpu while another process holds the GPU.")
    args = ap.parse_args()
    letter = args.letter.upper()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else get_device()
    model, classes = load_model(args.model, device)
    k = classes.index(letter)
    folder = load_image_folder(DATASET_DIR, transform=None)
    assert folder.classes == classes, "dataset folder order differs from checkpoint class order"
    splits = load_split_indices(SPLITS_PATH)
    print(f"checkpoint={args.model.name}  device={device}  classes==folders: True")

    # ---- 1. counts per split --------------------------------------------------------------
    counts = {name: Counter(folder.samples[i][1] for i in idx) for name, idx in splits.items()}
    print(f"\n[1] {letter} examples per split:",
          {name: counts[name][k] for name in splits},
          "| all-class min/max in train:", min(counts["train"].values()), max(counts["train"].values()))

    # ---- 2. metrics on the full held-out test split ------------------------------------------
    test = [folder.samples[i] for i in splits["test"]]
    probs = predict_split(model, test, device)
    y_true = np.array([t for _, t in test])
    y_pred = probs.argmax(1)
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    overall = float((y_true == y_pred).mean())
    print(f"\n[2] held-out test images={len(test)} overall accuracy={overall:.4f}")
    p_, r_, f_ = prf(cm, k)
    print(f"{letter}: precision={p_:.4f} recall={r_:.4f} f1={f_:.4f} (support={cm[k].sum()})")
    row = {classes[j]: int(cm[k, j]) for j in range(n) if cm[k, j] and j != k}
    col = {classes[i]: int(cm[i, k]) for i in range(n) if cm[i, k] and i != k}
    print(f"{letter} row (true {letter}, predicted as other):", row or "none")
    print(f"{letter} column (other true classes predicted as {letter}):", col or "none")
    print("\nper-class on the held-out test split:")
    print(f"{'cls':3s} {'prec':>6s} {'rec':>6s} {'f1':>6s}  top confusions (true->pred)")
    for name in [letter] + args.compare:
        j = classes.index(name)
        pj, rj, fj = prf(cm, j)
        wrong = sorted(((cm[j, c], classes[c]) for c in range(n) if c != j and cm[j, c]), reverse=True)[:3]
        print(f"{name:3s} {pj:6.3f} {rj:6.3f} {fj:6.3f}  {', '.join(f'{name}->{c}:{v}' for v, c in wrong) or '-'}")

    # ---- 3. label sanity: filename prefix vs folder, sizes -----------------------------------
    mismatched = [p for p, t in folder.samples if not Path(p).name.upper().startswith(classes[t])]
    print(f"\n[3] files whose name prefix != folder letter (whole dataset): {len(mismatched)}", mismatched[:3])
    nums = defaultdict(list)
    for p, t in folder.samples:
        m = re.search(r"(\d+)", Path(p).stem)
        nums[t].append(int(m.group(1)) if m else -1)
    c_nums = sorted(nums[k])
    print(f"{letter} filename numbers: min={c_nums[0]} max={c_nums[-1]} count={len(c_nums)} unique={len(set(c_nums))}")

    # ---- 4. exact duplicates and cross-split leakage for this letter ------------------------------
    split_of = {}
    for name, idx in splits.items():
        for i in idx:
            split_of[i] = name
    digest_to_items = defaultdict(list)
    for i, (p, t) in enumerate(folder.samples):
        if t == k:
            digest_to_items[hashlib.md5(Path(p).read_bytes()).hexdigest()].append((Path(p).name, split_of[i]))
    dups = {d: v for d, v in digest_to_items.items() if len(v) > 1}
    cross = {d: v for d, v in dups.items() if len({s for _, s in v}) > 1}
    print(f"\n[4] {letter} exact-duplicate groups: {len(dups)} (files involved: {sum(len(v) for v in dups.values())}); "
          f"groups spanning train/val/test (leakage): {len(cross)}")
    if cross:
        print("   e.g.", list(cross.values())[0])

    # ---- 5. a cross-class look: do other classes' images look like this letter's? --------------
    sizes = Counter(Image.open(folder.samples[i][0]).size for i in splits["test"][:2000])
    print(f"\n[5] image sizes (first 2000 test images): {dict(sizes.most_common(3))}")

    # ---- save raw results + contact sheets ------------------------------------------------------
    result = {
        "checkpoint": str(args.model), "letter": letter, "overall_accuracy": overall,
        "counts": {name: int(counts[name][k]) for name in splits},
        "precision": p_, "recall": r_, "f1": f_, "row": row, "column": col,
        "confusion_matrix": cm.tolist(), "classes": classes,
    }
    (args.out / f"{letter}_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def sheet(items, name, cols=8, size=140):
        rows = (len(items) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * size, rows * size), "white")
        for n_, path in enumerate(items):
            with Image.open(path) as im:
                canvas.paste(im.convert("RGB").resize((size, size)), ((n_ % cols) * size, (n_ // cols) * size))
        canvas.save(args.out / name)

    rng = np.random.default_rng(0)
    for name in ("train", "test"):
        idx = [i for i in splits[name] if folder.samples[i][1] == k]
        sheet([folder.samples[i][0] for i in rng.choice(idx, 32, replace=False)], f"{letter}_{name}_sample.jpg")
    print(f"\ncontact sheets + JSON written to {args.out}")


if __name__ == "__main__":
    main()
