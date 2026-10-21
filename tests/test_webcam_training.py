"""Tests for the webcam dataset and the mixed original+webcam sampler (src/data/webcam.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import eval_transforms  # noqa: E402
from src.data.webcam import WebcamCropDataset, mixed_train_loader, read_manifest  # noqa: E402


def make_export(root: Path, rows: list[tuple[str, str, str]]) -> None:
    """rows: (split, letter, kind)"""
    lines = []
    for i, (split, letter, kind) in enumerate(rows):
        folder = root / split / letter
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{letter}_{i}.png"
        Image.fromarray(np.full((40, 40, 3), (i * 7) % 255, dtype=np.uint8)).save(path)
        lines.append(json.dumps({"split": split, "letter": letter, "kind": kind, "room": f"room-{split}", "crop": f"{split}/{letter}/{path.name}"}))
    (root / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class WebcamDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        make_export(self.root, [("train", "C", "single")] * 10 + [("train", "W", "single")] * 2 + [("val", "C", "single"), ("test", "C", "hold")])

    def test_reads_only_the_requested_split_and_maps_letters_to_the_model_indices(self) -> None:
        ds = WebcamCropDataset(self.root, "train", eval_transforms())
        self.assertEqual(len(ds), 12)
        self.assertEqual(Counter(ds.labels), {2: 10, 22: 2})  # C is index 2, W is index 22 (A-Z order)
        tensor, label = ds[0]
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(label, 2)

    def test_kind_filter(self) -> None:
        self.assertEqual(len(WebcamCropDataset(self.root, "test", eval_transforms(), kinds=("single",))), 0)
        self.assertEqual(len(WebcamCropDataset(self.root, "test", eval_transforms(), kinds=("hold",))), 1)

    def test_missing_manifest_gives_a_clear_error(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            read_manifest(self.root / "nowhere", "train")
        self.assertIn("export_webcam_dataset", str(ctx.exception))


class MixedSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        make_export(self.root, [("train", "C", "single")] * 10 + [("train", "W", "single")] * 2)
        self.webcam = WebcamCropDataset(self.root, "train", eval_transforms())
        self.original = torch.utils.data.TensorDataset(torch.zeros(500, 3, 224, 224), torch.zeros(500, dtype=torch.long))

    def draws(self, share: float, n: int = 20000):
        torch.manual_seed(0)
        loader = mixed_train_loader(self.original, self.webcam, batch_size=8, webcam_share=share, epoch_samples=n)
        return list(loader.sampler), len(self.original)

    def test_webcam_share_is_respected(self) -> None:
        for share in (0.2, 0.5):
            indices, n_orig = self.draws(share)
            fraction = sum(i >= n_orig for i in indices) / len(indices)
            self.assertAlmostEqual(fraction, share, delta=0.02)

    def test_webcam_draws_are_balanced_across_letters(self) -> None:
        indices, n_orig = self.draws(0.5)
        labels = Counter(self.webcam.labels[i - n_orig] for i in indices if i >= n_orig)
        total = sum(labels.values())
        self.assertAlmostEqual(labels[2] / total, 0.5, delta=0.03)   # 10 C crops vs 2 W crops, yet ~equal draws
        self.assertAlmostEqual(labels[22] / total, 0.5, delta=0.03)

    def test_epoch_size_and_validation(self) -> None:
        indices, _ = self.draws(0.3, n=1234)
        self.assertEqual(len(indices), 1234)
        with self.assertRaises(ValueError):
            mixed_train_loader(self.original, self.webcam, batch_size=8, webcam_share=0.0, epoch_samples=10)
        with self.assertRaises(ValueError):
            mixed_train_loader(self.original, self.webcam, batch_size=8, webcam_share=1.0, epoch_samples=10)
        empty = WebcamCropDataset(self.root, "val", eval_transforms())
        with self.assertRaises(ValueError):
            mixed_train_loader(self.original, empty, batch_size=8, webcam_share=0.3, epoch_samples=10)


if __name__ == "__main__":
    unittest.main()
