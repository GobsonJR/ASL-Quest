from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchvision import datasets

from src.config import DATASET_DIR, OUTPUTS_DIR
from src.data.dataset import sequential_split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check exact duplicate leakage across controlled splits")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "evaluation" / "duplicate_leakage.json")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quicker exploratory checks")
    return parser.parse_args()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset = datasets.ImageFolder(str(args.data_dir))
    split_indices = dict(zip(("train", "val", "test"), sequential_split_indices(dataset.samples)))
    seen: dict[str, list[dict[str, str]]] = defaultdict(list)
    checked = 0

    for split_name, indices in split_indices.items():
        for idx in indices:
            if args.max_files is not None and checked >= args.max_files:
                break
            path = Path(dataset.samples[idx][0])
            seen[file_md5(path)].append(
                {
                    "split": split_name,
                    "class": dataset.classes[dataset.samples[idx][1]],
                    "path": str(path),
                }
            )
            checked += 1
        if args.max_files is not None and checked >= args.max_files:
            break

    duplicate_groups = []
    leakage_groups = []
    for items in seen.values():
        if len(items) <= 1:
            continue
        duplicate_groups.append(items)
        if len({item["split"] for item in items}) > 1:
            leakage_groups.append(items)

    result = {
        "data_dir": str(args.data_dir),
        "split_policy": "sequential per-class split by numeric filename",
        "checked_files": checked,
        "complete_scan": args.max_files is None,
        "duplicate_groups": len(duplicate_groups),
        "leakage_groups": len(leakage_groups),
        "examples": leakage_groups[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Exact Duplicate Split Leakage")
    print("-----------------------------")
    print(f"Checked files: {checked}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Cross-split leakage groups: {len(leakage_groups)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
