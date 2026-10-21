"""Torch datasets/loaders for the exported webcam crops (see src/data/webcam_export.py)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

LETTERS = [chr(65 + i) for i in range(26)]


def read_manifest(root: Path, split: str, kinds: tuple[str, ...] = ("single", "hold")) -> list[dict]:
    path = root / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run scripts/export_webcam_dataset.py first")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["split"] == split and r["kind"] in kinds]


class WebcamCropDataset(Dataset):
    """The hand crops exactly as the live app produces them, with person-confirmed labels."""

    def __init__(self, root: Path, split: str, transform: Callable, kinds: tuple[str, ...] = ("single", "hold")) -> None:
        self.root, self.transform = Path(root), transform
        self.rows = read_manifest(self.root, split, kinds)
        self.labels = [LETTERS.index(r["letter"]) for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / row["crop"]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]


def mixed_train_loader(
    original,
    webcam: WebcamCropDataset,
    *,
    batch_size: int,
    webcam_share: float,
    epoch_samples: int,
    num_workers: int = 0,
) -> DataLoader:
    """Each epoch draws `epoch_samples` images, `webcam_share` of them from webcam crops.

    Webcam draws are balanced across the letters that have webcam data (one weak letter must not be drowned by a
    common one); the original dataset is already balanced. Sampling is with replacement, so a small webcam set is
    revisited several times per epoch under different random augmentation.
    """
    if not 0.0 < webcam_share < 1.0:
        raise ValueError("webcam_share must be between 0 and 1 (exclusive)")
    if len(webcam) == 0:
        raise ValueError("no webcam training crops: nothing to mix in")
    per_letter = Counter(webcam.labels)
    weights = [(1.0 - webcam_share) / len(original)] * len(original)
    letters_present = len(per_letter)
    weights += [webcam_share / letters_present / per_letter[label] for label in webcam.labels]
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=epoch_samples, replacement=True)
    return DataLoader(
        ConcatDataset([original, webcam]),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
