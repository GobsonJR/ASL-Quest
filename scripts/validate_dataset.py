from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, UnidentifiedImageError

from src.config import DATASET_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the ASL image dataset")
    parser.add_argument("--data-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--check-duplicates", action="store_true")
    parser.add_argument("--max-hash", type=int, default=None, help="Optional cap for duplicate hashing")
    return parser.parse_args()


def iter_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return files


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.data_dir
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]
    expected = [chr(ord("A") + i) for i in range(26)]
    missing = [c for c in expected if c not in class_names]

    counts: dict[str, int] = {}
    valid = 0
    corrupted: list[str] = []
    empty: list[str] = []
    tiny: list[str] = []
    unsupported: list[str] = []
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    hashed = 0

    all_files = list(root.rglob("*"))
    extra_files = [
        p for p in all_files if p.is_file() and p.suffix.lower() not in IMAGE_EXTS and not p.name.startswith(".")
    ]
    for path in extra_files:
        unsupported.append(str(path.relative_to(root)))

    for class_dir in class_dirs:
        files = [p for p in class_dir.iterdir() if p.is_file()]
        n_ok = 0
        for path in files:
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            if path.stat().st_size == 0:
                empty.append(str(path.relative_to(root)))
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    sizes[image.size] += 1
                    modes[image.mode] += 1
                    if min(image.size) < MIN_SIZE:
                        tiny.append(str(path.relative_to(root)))
                        continue
            except (UnidentifiedImageError, OSError):
                corrupted.append(str(path.relative_to(root)))
                continue
            n_ok += 1
            valid += 1
            if args.check_duplicates and (args.max_hash is None or hashed < args.max_hash):
                hashes[file_md5(path)].append(str(path.relative_to(root)))
                hashed += 1
        counts[class_dir.name] = n_ok

    duplicate_groups = {k: v for k, v in hashes.items() if len(v) > 1}
    duplicate_count = sum(len(v) - 1 for v in duplicate_groups.values())

    print("Dataset Validation")
    print("------------------")
    print(f"Root: {root}")
    print(f"Total classes: {len(class_dirs)}")
    print(f"Valid images: {valid}")
    print(f"Corrupted images: {len(corrupted)}")
    print(f"Empty files: {len(empty)}")
    print(f"Tiny images (<{MIN_SIZE}px): {len(tiny)}")
    print(f"Unsupported files: {len(unsupported)}")
    print(f"Duplicate images: {duplicate_count if args.check_duplicates else 'not scanned'}")
    print(f"Missing classes: {', '.join(missing) if missing else 'none'}")
    print("\nPer-class counts")
    for name in class_names:
        print(f"{name}: {counts.get(name, 0)}")
    if sizes:
        common_size, n = sizes.most_common(1)[0]
        print(f"\nMost common size: {common_size[0]}x{common_size[1]} ({n} images)")
        print(f"Unique sizes: {len(sizes)}")
    if modes:
        print("Channels/modes:", dict(modes))
    if corrupted[:10]:
        print("\nCorrupted samples:")
        for item in corrupted[:10]:
            print(f"  {item}")
    if args.check_duplicates and duplicate_groups:
        print("\nDuplicate groups (first 5):")
        for i, paths in enumerate(list(duplicate_groups.values())[:5], start=1):
            print(f"  {i}. {paths[:3]}")


if __name__ == "__main__":
    main()
