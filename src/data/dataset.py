from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from src.config import BATCH_SIZE, NUM_WORKERS, SPLITS_PATH, TEST_RATIO, TRAIN_RATIO, VAL_RATIO
from src.data.preprocessing import eval_transforms, train_transforms


_NUM_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def numeric_name_key(path: str) -> tuple[int, str]:
    name = Path(path).name
    match = _NUM_RE.search(name)
    return (int(match.group(1)) if match else 10**12, name)


def load_image_folder(data_dir: Path, transform) -> datasets.ImageFolder:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    return datasets.ImageFolder(root=str(data_dir), transform=transform)


def sequential_split_indices(
    samples: list[tuple[str, int]],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[list[int], list[int], list[int]]:
    """Split each class by filename order to reduce near-duplicate leakage."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.")

    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, (path, label) in enumerate(samples):
        by_class[label].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for label in sorted(by_class):
        ordered = sorted(by_class[label], key=lambda i: numeric_name_key(samples[i][0]))
        n = len(ordered)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(ordered[:n_train])
        val_idx.extend(ordered[n_train : n_train + n_val])
        test_idx.extend(ordered[n_train + n_val :])

    return train_idx, val_idx, test_idx


def save_split(indices: dict[str, list[int]], path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = dict(indices)
    if extra:
        payload = {"splits": indices, **extra}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_indices(path: Path) -> dict[str, list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "splits" in data and isinstance(data["splits"], dict):
        data = data["splits"]
    return {name: list(data[name]) for name in ("train", "val", "test") if name in data}


def full_sequential_splits(samples: list[tuple[str, int]]) -> dict[str, list[int]]:
    train_idx, val_idx, test_idx = sequential_split_indices(samples)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_dataloaders(
    data_dir: Path,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    max_per_class: int | None = None,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], dict[str, list[int]]]:
    train_base = load_image_folder(data_dir, train_transforms(image_size))
    eval_base = load_image_folder(data_dir, eval_transforms(image_size))

    samples = train_base.samples
    train_idx, val_idx, test_idx = sequential_split_indices(samples)

    if max_per_class is not None:
        train_idx = _limit_per_class(train_idx, samples, max_per_class)
        val_idx = _limit_per_class(val_idx, samples, max(1, max_per_class // 4))
        test_idx = _limit_per_class(test_idx, samples, max(1, max_per_class // 4))

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    split_path = SPLITS_PATH if max_per_class is None else SPLITS_PATH.parent / "smoke_test" / "splits.json"
    save_split(
        splits,
        split_path,
        extra={
            "policy": "sequential per-class split by numeric filename",
            "limited": max_per_class is not None,
            "max_per_class": max_per_class,
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
        },
    )

    pin = True
    persistent = num_workers > 0
    train_loader = DataLoader(
        Subset(train_base, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        Subset(eval_base, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        Subset(eval_base, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader, list(train_base.classes), splits


def _limit_per_class(
    indices: list[int],
    samples: list[tuple[str, int]],
    limit: int,
) -> list[int]:
    counts: dict[int, int] = defaultdict(int)
    kept: list[int] = []
    for idx in indices:
        label = samples[idx][1]
        if counts[label] >= limit:
            continue
        counts[label] += 1
        kept.append(idx)
    return kept


def subset_targets(dataset: Dataset, indices: list[int]) -> list[int]:
    if hasattr(dataset, "targets"):
        targets = dataset.targets  # type: ignore[attr-defined]
        return [int(targets[i]) for i in indices]
    if hasattr(dataset, "samples"):
        samples = dataset.samples  # type: ignore[attr-defined]
        return [int(samples[i][1]) for i in indices]
    raise TypeError("Dataset does not expose targets or samples.")
