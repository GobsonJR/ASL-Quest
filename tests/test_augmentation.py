"""Augmentation policy tests: the default policy is unchanged, the robust one is opt-in, and
augmentation can never reach the validation/test subsets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASET_DIR, IMAGE_SIZE  # noqa: E402
from src.data.dataset import build_dataloaders  # noqa: E402
from src.data.preprocessing import (  # noqa: E402
    AddGaussianNoise,
    RandomJpeg,
    eval_transforms,
    robust_train_transforms,
    train_transforms,
    train_transforms_for,
)


def names(compose) -> list[str]:
    return [step.__class__.__name__ for step in compose.transforms]


class PolicyTests(unittest.TestCase):
    def test_default_policy_is_the_original_no_flip_pipeline(self) -> None:
        self.assertEqual(names(train_transforms_for("default")), names(train_transforms()))
        self.assertNotIn("RandomHorizontalFlip", names(train_transforms_for("default")))

    def test_robust_policy_adds_mirroring_scale_and_lighting(self) -> None:
        robust = names(train_transforms_for("robust"))
        for expected in ("RandomHorizontalFlip", "RandomAffine", "ColorJitter", "RandomJpeg", "AddGaussianNoise", "RandomErasing"):
            self.assertIn(expected, robust)

    def test_unknown_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            train_transforms_for("everything")

    def test_evaluation_transform_never_contains_augmentation(self) -> None:
        evaluation = names(eval_transforms())
        for random_step in ("RandomHorizontalFlip", "RandomAffine", "ColorJitter", "RandomErasing", "RandomCrop", "RandomJpeg", "AddGaussianNoise"):
            self.assertNotIn(random_step, evaluation)

    def test_robust_output_shape_dtype_and_range(self) -> None:
        torch.manual_seed(0)
        tensor = robust_train_transforms()(Image.new("RGB", (200, 200), (180, 120, 90)))
        self.assertEqual(tuple(tensor.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())

    def test_noise_stays_in_range_and_is_strength_limited(self) -> None:
        torch.manual_seed(3)
        noise = AddGaussianNoise(p=1.0, max_sigma=0.08)
        clean = torch.full((3, 32, 32), 0.5)
        noisy = noise(clean)
        self.assertGreaterEqual(float(noisy.min()), 0.0)
        self.assertLessEqual(float(noisy.max()), 1.0)
        self.assertLess(float((noisy - clean).abs().max()), 0.5)
        self.assertTrue(torch.equal(AddGaussianNoise(p=0.0)(clean), clean))  # p=0 never touches the image

    def test_random_jpeg_returns_same_size_rgb(self) -> None:
        out = RandomJpeg(p=1.0)(Image.new("RGB", (64, 48), (200, 50, 50)))
        self.assertEqual(out.size, (64, 48))
        self.assertEqual(out.mode, "RGB")

    def test_robust_policy_is_label_preserving_in_shape_only(self) -> None:
        # Mirroring happens, but only as one of the random steps: it does not change image size.
        torch.manual_seed(1)
        sizes = {tuple(robust_train_transforms()(Image.new("RGB", (200, 200))).shape) for _ in range(5)}
        self.assertEqual(sizes, {(3, IMAGE_SIZE, IMAGE_SIZE)})


@unittest.skipUnless(DATASET_DIR.exists(), "dataset not present")
class NoLeakageTests(unittest.TestCase):
    def test_augmentation_is_attached_to_the_train_subset_only(self) -> None:
        train, val, test, classes, splits = build_dataloaders(
            DATASET_DIR, batch_size=4, num_workers=0, max_per_class=8, augment="robust"
        )
        self.assertIn("RandomHorizontalFlip", names(train.dataset.dataset.transform))
        for loader in (val, test):
            self.assertNotIn("RandomHorizontalFlip", names(loader.dataset.dataset.transform))
            self.assertNotIn("RandomAffine", names(loader.dataset.dataset.transform))
        # the splits themselves are disjoint index sets
        self.assertFalse(set(splits["train"]) & set(splits["val"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        self.assertFalse(set(splits["val"]) & set(splits["test"]))
        self.assertEqual(classes, [chr(ord("A") + i) for i in range(26)])


if __name__ == "__main__":
    unittest.main()
