"""On-the-fly ASL Citizen video dataset with temporal frame sampling."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.asl_citizen.config import ASLCitizenConfig, default_config
from src.asl_citizen.preprocessing import (
    apply_spatial_clip,
    eval_spatial_transforms,
    jittered_temporal_indices,
    train_spatial_transforms,
    uniform_temporal_indices,
)
from src.asl_citizen.utils import load_class_mappings, load_manifest_rows, resolve_video_path


def _read_frame_bgr(capture: cv2.VideoCapture, index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
    ok, frame = capture.read()
    if ok and frame is not None:
        return frame
    return None


def get_frame_count(capture: cv2.VideoCapture) -> int:
    reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if reported > 0:
        return reported
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    return count


def decode_sampled_frames(
    video_path: Path,
    indices: list[int],
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        frame_count = get_frame_count(capture)
        if frame_count <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

        needed = sorted(set(indices))
        decoded: dict[int, np.ndarray] = {}

        for index in needed:
            frame = _read_frame_bgr(capture, index)
            if frame is not None:
                decoded[index] = frame

        if len(decoded) < len(needed):
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current = 0
            needed_set = set(needed) - set(decoded.keys())
            ok, frame = capture.read()
            while ok and frame is not None and needed_set:
                if current in needed_set:
                    decoded[current] = frame
                    needed_set.remove(current)
                current += 1
                ok, frame = capture.read()

        frames: list[np.ndarray] = []
        last: np.ndarray | None = None
        for index in indices:
            frame = decoded.get(index, last)
            if frame is None:
                height = next(iter(decoded.values())).shape[0] if decoded else 224
                width = next(iter(decoded.values())).shape[1] if decoded else 224
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            frames.append(frame)
            last = frame
        return frames
    finally:
        capture.release()


def sample_temporal_indices(
    frame_count: int,
    num_frames: int,
    training: bool,
    jitter_frames: int,
    rng: random.Random | None,
) -> list[int]:
    if training and jitter_frames > 0:
        return jittered_temporal_indices(frame_count, num_frames, jitter_frames, rng or random.Random(0))
    return uniform_temporal_indices(frame_count, num_frames)


class ASLCitizenDataset(Dataset):
    def __init__(
        self,
        manifest_rows: list[dict[str, Any]],
        *,
        video_root: Path,
        class_to_idx: dict[str, int],
        num_frames: int = 16,
        transform: Callable | None = None,
        training: bool = False,
        jitter_frames: int = 2,
        seed: int = 42,
        consistent_clip_transforms: bool = False,
        image_size: int = 224,
        imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        color_jitter: transforms.ColorJitter | None = None,
    ) -> None:
        self.rows = list(manifest_rows)
        self.video_root = video_root
        self.class_to_idx = dict(class_to_idx)
        self.num_frames = num_frames
        self.transform = transform
        self.training = training
        self.jitter_frames = jitter_frames
        self.consistent_clip_transforms = consistent_clip_transforms
        self.image_size = image_size
        self.imagenet_mean = imagenet_mean
        self.imagenet_std = imagenet_std
        # Optional, defaults to None (identical behavior to before this field existed).
        # Only ever applied when self.training is True; apply_spatial_clip additionally
        # gates on training=True itself, so eval/val/test clips are never jittered even
        # if a caller mistakenly passes color_jitter with training=False.
        self.color_jitter = color_jitter if training else None
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        video_path = resolve_video_path(self.video_root, row["video_file"])
        gloss = row["gloss"]
        if gloss not in self.class_to_idx:
            raise KeyError(f"Unknown gloss in manifest: {gloss}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        try:
            frame_count = get_frame_count(capture)
        finally:
            capture.release()
        if frame_count <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

        indices = sample_temporal_indices(
            frame_count,
            self.num_frames,
            training=self.training,
            jitter_frames=self.jitter_frames,
            rng=self._rng,
        )
        frames_bgr = decode_sampled_frames(video_path, indices)
        pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames_bgr]
        if self.consistent_clip_transforms:
            clip = apply_spatial_clip(
                pil_frames,
                training=self.training,
                image_size=self.image_size,
                mean=self.imagenet_mean,
                std=self.imagenet_std,
                color_jitter=self.color_jitter,
            )
        else:
            tensors: list[torch.Tensor] = []
            for image in pil_frames:
                if self.transform is not None:
                    tensors.append(self.transform(image))
                else:
                    tensors.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0)
            clip = torch.stack(tensors, dim=0)
        label = int(row.get("class_idx", self.class_to_idx[gloss]))
        return {
            "video": clip,
            "label": label,
            "gloss": gloss,
            "video_file": row["video_file"],
            "participant_id": row["participant_id"],
            "split": row.get("split", ""),
        }


def build_dataloader(
    split: str,
    cfg: ASLCitizenConfig | None = None,
    *,
    shuffle: bool | None = None,
    batch_size: int | None = None,
    color_jitter: transforms.ColorJitter | None = None,
) -> tuple[DataLoader, ASLCitizenDataset]:
    cfg = cfg or default_config()
    class_to_idx, _ = load_class_mappings(cfg.class_to_idx_path, cfg.idx_to_class_path)

    manifest_map = {
        "train": cfg.train_manifest,
        "val": cfg.val_manifest,
        "test": cfg.test_manifest,
    }
    if split not in manifest_map:
        raise ValueError(f"Unknown split: {split}")

    rows = load_manifest_rows(manifest_map[split])
    training = split == "train"
    consistent = bool(getattr(cfg, "consistent_clip_transforms", True))
    transform = None
    if not consistent:
        transform = (
            train_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
            if training
            else eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
        )
    dataset = ASLCitizenDataset(
        rows,
        video_root=cfg.video_root,
        class_to_idx=class_to_idx,
        num_frames=cfg.num_frames,
        transform=transform,
        training=training,
        jitter_frames=cfg.train_jitter_frames,
        seed=cfg.seed,
        consistent_clip_transforms=consistent,
        image_size=cfg.image_size,
        imagenet_mean=cfg.imagenet_mean,
        imagenet_std=cfg.imagenet_std,
        color_jitter=color_jitter if training else None,
    )
    if shuffle is None:
        shuffle = training

    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader, dataset
