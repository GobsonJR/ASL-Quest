from __future__ import annotations

from torchvision import transforms

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Training augmentations that do not reverse left/right sign orientation."""
    return transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class RandomJpeg:
    """Re-encode as JPEG at a random quality, like the browser's live frames (quality 0.72)."""

    def __init__(self, p: float = 0.4, quality: tuple[int, int] = (35, 90)) -> None:
        self.p, self.quality = p, quality

    def __call__(self, image):
        import io
        import random

        if random.random() >= self.p:
            return image
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=random.randint(*self.quality))
        buffer.seek(0)
        from PIL import Image

        return Image.open(buffer).convert("RGB")


class AddGaussianNoise:
    """Sensor-style noise on a [0, 1] image tensor, with a random strength per image."""

    def __init__(self, p: float = 0.5, max_sigma: float = 0.08) -> None:
        self.p, self.max_sigma = p, max_sigma

    def __call__(self, tensor):
        import torch

        if torch.rand(()) >= self.p:
            return tensor
        sigma = float(torch.rand(())) * self.max_sigma
        return (tensor + torch.randn_like(tensor) * sigma).clamp(0.0, 1.0)


def robust_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Opt-in training augmentation for webcam robustness (``train.py --augment robust``).

    Added because measurements on the held-out set showed the retrained checkpoint is brittle
    outside the recording setup it was trained on (scripts/diagnose_c_experiments.py): a mirrored
    hand (left-handed signer) scored 0% for C and 37% on average over all letters, and hands that
    appear smaller or at a different angle lost accuracy. Noise and JPEG artifacts are included
    because a first candidate trained without them became noticeably more noise-sensitive than the
    original (C: 100% -> 20% under Gaussian noise). Mirroring is label-preserving for ASL
    letters (a left-handed signer forms the mirror image of the same letter).

    Applied to the TRAINING subset only. Validation/test always use :func:`eval_transforms`.
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=25, translate=(0.12, 0.12), scale=(0.55, 1.15)),
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.25),
            transforms.RandomGrayscale(p=0.05),
            RandomJpeg(p=0.4),
            transforms.ToTensor(),
            AddGaussianNoise(p=0.5, max_sigma=0.08),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )


def train_transforms_for(policy: str, image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """``"default"`` (the original no-flip policy) or ``"robust"``."""
    if policy == "default":
        return train_transforms(image_size)
    if policy == "robust":
        return robust_train_transforms(image_size)
    raise ValueError(f"Unknown augmentation policy: {policy!r} (expected 'default' or 'robust')")
