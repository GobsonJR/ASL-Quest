"""Read-only audit of a downloaded alphabet dataset (a ZIP is read in place, never extracted).

    venv311\\Scripts\\python.exe scripts\\audit_external_dataset.py --zip dataset\\archive.zip
    venv311\\Scripts\\python.exe scripts\\audit_external_dataset.py --folder dataset\\asl_alphabet_train   # for comparison

Reports image sizes/modes, classes and counts, duplicate files (zip CRC), background statistics (corner/edge colours and
whether images look segmented: transparent or flat-black surroundings) and writes contact sheets for the weak letters.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FOCUS = list("CDWXJTM")


def background_stats(img: Image.Image) -> dict:
    rgb = np.asarray(img.convert("RGB").resize((96, 96)), dtype=np.float32)
    edge = np.concatenate([rgb[:6].reshape(-1, 3), rgb[-6:].reshape(-1, 3), rgb[:, :6].reshape(-1, 3), rgb[:, -6:].reshape(-1, 3)])
    return {
        "edge_mean": edge.mean(0).round(0).tolist(),
        "edge_std": float(edge.std(0).mean()),
        "edge_dark_fraction": float((edge.max(1) < 20).mean()),
    }


def sheet(images: list[Image.Image], path: Path, cols: int = 8, size: int = 160) -> None:
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * size, rows * size), "white")
    for i, im in enumerate(images):
        canvas.paste(im.convert("RGB").resize((size, size)), ((i % cols) * size, (i // cols) * size))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path)
    ap.add_argument("--folder", type=Path)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "dataset_audit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    entries: dict[str, list[tuple[str, int, object]]] = collections.defaultdict(list)  # "split/class" -> [(name, crc, handle)]
    if args.zip:
        z = zipfile.ZipFile(args.zip)
        for info in z.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            entries["/".join(info.filename.split("/")[:-1]) or "(root)"].append((info.filename, info.CRC, info))
        opener = lambda e: Image.open(io.BytesIO(z.read(e[2]))); tag = args.zip.stem
    else:
        files = [p for p in args.folder.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        for p in files:
            entries[str(p.parent.relative_to(args.folder))].append((str(p), 0, p))
        opener = lambda e: Image.open(e[2]); tag = args.folder.name

    report: dict = {"source": str(args.zip or args.folder), "groups": {g: len(v) for g, v in sorted(entries.items())}}
    crcs = collections.Counter(c for v in entries.values() for _, c, _ in v if c)
    report["duplicate_files_by_crc"] = sum(n - 1 for n in crcs.values() if n > 1)
    # train/test overlap by CRC (zip only)
    by_split = collections.defaultdict(set)
    for g, v in entries.items():
        by_split[g.split("/")[0]].update(c for _, c, _ in v if c)
    splits = [s for s in by_split if s]
    if len(splits) >= 2:
        report["cross_split_identical_files"] = len(by_split[splits[0]] & by_split[splits[1]])

    sizes, modes, bg = collections.Counter(), collections.Counter(), []
    per_class_bg: dict[str, list[dict]] = {}
    for group, items in sorted(entries.items()):
        picked = rng.sample(items, min(args.per_class, len(items)))
        stats = []
        for e in picked:
            with opener(e) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
                s = background_stats(im)
                if im.mode in ("RGBA", "LA") or "transparency" in im.info:
                    a = np.asarray(im.convert("RGBA"))[..., 3]
                    s["transparent_fraction"] = float((a < 250).mean())
                stats.append(s)
        per_class_bg[group] = stats
        bg.extend(stats)
    report["sample_image_sizes"] = {f"{w}x{h}": n for (w, h), n in sizes.most_common(8)}
    report["sample_modes"] = dict(modes)
    means = np.array([s["edge_mean"] for s in bg])
    report["background"] = {
        "edge_colour_spread_across_images(std of per-image edge mean)": means.std(0).round(1).tolist(),
        "median_edge_std_within_image": round(float(np.median([s["edge_std"] for s in bg])), 1),
        "images_with_mostly_black_edges": round(float(np.mean([s["edge_dark_fraction"] > 0.8 for s in bg])), 3),
        "images_with_transparency": round(float(np.mean([s.get("transparent_fraction", 0) > 0.05 for s in bg])), 3),
    }
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    for group, items in entries.items():
        letter = group.split("/")[-1]
        if letter in FOCUS:
            ims = []
            for e in rng.sample(items, min(16, len(items))):
                with opener(e) as im:
                    ims.append(im.copy())
            sheet(ims, out / f"{group.replace('/', '_')}.png")
    (out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "groups"}, indent=2))
    print("groups:", len(report["groups"]), "total images:", sum(report["groups"].values()))
    print("contact sheets in", out)


if __name__ == "__main__":
    main()
