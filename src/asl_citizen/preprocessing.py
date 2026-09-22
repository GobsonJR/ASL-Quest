"""Spatial transforms and temporal frame sampling for ASL Citizen videos."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


def train_spatial_transforms(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> transforms.Compose:
    """Conservative training transforms — no horizontal flip (ASL handedness matters)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


def eval_spatial_transforms(
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


def apply_clip_color_jitter(
    images: list[Image.Image],
    color_jitter: "transforms.ColorJitter",
) -> list[Image.Image]:
    """Apply ONE sampled brightness/contrast/saturation/hue adjustment to every frame.

    Mirrors torchvision.ColorJitter.forward's per-call param sampling, but samples the
    factors and operation order ONCE per clip (not once per frame) so every frame in the
    clip receives an identical appearance shift. Independent per-frame color jitter would
    introduce frame-to-frame flicker unrelated to the sign itself, the same alignment
    problem apply_spatial_clip's shared crop avoids for geometry.
    """
    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
        transforms.ColorJitter.get_params(
            color_jitter.brightness,
            color_jitter.contrast,
            color_jitter.saturation,
            color_jitter.hue,
        )
    )
    out: list[Image.Image] = []
    for image in images:
        img = image
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = TF.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = TF.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = TF.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = TF.adjust_hue(img, hue_factor)
        out.append(img)
    return out


def apply_spatial_clip(
    images: list[Image.Image],
    *,
    training: bool,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    color_jitter: "transforms.ColorJitter | None" = None,
) -> torch.Tensor:
    """Apply the SAME geometric crop to every frame in a clip.

    Independent per-frame RandomResizedCrop destroys temporal alignment and was a
    leading cause of 100-class collapse despite tiny-overfit success on CenterCrop.

    color_jitter is optional and defaults to None, which preserves the exact prior
    behavior of this function for every caller that does not pass it. When provided, it
    is applied ONLY when training=True (never to eval/val/test clips), after the shared
    crop and before normalization, via apply_clip_color_jitter (same factors for every
    frame in the clip).
    """
    if not images:
        raise ValueError("apply_spatial_clip requires at least one image")
    mean_t = tuple(mean)
    std_t = tuple(std)
    if training:
        resized = [TF.resize(image, [image_size + 32, image_size + 32]) for image in images]
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            resized[0],
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
        )
        cropped = [
            TF.resized_crop(image, top, left, height, width, [image_size, image_size])
            for image in resized
        ]
        if color_jitter is not None:
            cropped = apply_clip_color_jitter(cropped, color_jitter)
    else:
        resized = [TF.resize(image, [image_size + 16, image_size + 16]) for image in images]
        cropped = [TF.center_crop(image, [image_size, image_size]) for image in resized]
    tensors = [TF.normalize(TF.to_tensor(image), mean_t, std_t) for image in cropped]
    return torch.stack(tensors, dim=0)


def uniform_temporal_indices(frame_count: int, num_frames: int) -> list[int]:
    """Deterministic evenly spaced 0-based frame indices across the full clip.

    For frame_count < num_frames, indices repeat via linspace rounding so the
    model always receives exactly num_frames indices without crashing.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if frame_count == 1:
        return [0] * num_frames
    positions = np.linspace(0, frame_count - 1, num=num_frames)
    indices = [int(round(float(p))) for p in positions]
    indices = [min(max(i, 0), frame_count - 1) for i in indices]
    return indices


def jittered_temporal_indices(
    frame_count: int,
    num_frames: int,
    max_jitter: int,
    rng: random.Random,
) -> list[int]:
    """Evenly spaced indices with mild per-sample jitter (training only)."""
    base = uniform_temporal_indices(frame_count, num_frames)
    if max_jitter <= 0 or frame_count <= 1:
        return base
    jittered: list[int] = []
    for index in base:
        offset = rng.randint(-max_jitter, max_jitter)
        jittered.append(min(max(index + offset, 0), frame_count - 1))
    return jittered
