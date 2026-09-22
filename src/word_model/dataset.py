"""Video dataset with uniform temporal sampling for WLASL100."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.word_model.config import ALLOWED_FRAME_COUNTS, WordModelConfig, default_config
from src.word_model.utils import load_manifest_rows, resolve_project_path, rows_for_split


def word_train_transforms(image_size: int, mean: Sequence[float], std: Sequence[float]) -> transforms.Compose:
    """Baseline v2 training transforms: center crop only. No horizontal flip (ASL handedness)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


def word_eval_transforms(image_size: int, mean: Sequence[float], std: Sequence[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


def to_manifest_frame_index(internal_index: int) -> int:
    """Convert an internal 0-based frame index to manifest 1-based frame numbering."""
    return int(internal_index) + 1


def clip_frame_span(frame_count: int, frame_start: int, frame_end: int) -> tuple[int, int]:
    """Convert manifest 1-based inclusive frame_start/frame_end to inclusive 0-based indices.

    NSLT/WLASL manifests use 1-based frame numbering where frame 1 is the first decoded frame.
    """
    if frame_count <= 0:
        return 0, 0
    start = 1 if frame_start is None else int(frame_start)
    if start < 1:
        start = 1
    start_idx = min(max(start - 1, 0), frame_count - 1)
    if frame_end is None or int(frame_end) < 0:
        end_idx = frame_count - 1
    else:
        end_idx = int(frame_end) - 1
        end_idx = min(max(end_idx, start_idx), frame_count - 1)
    return start_idx, end_idx


def sample_indices_for_span(
    frame_count: int,
    frame_start: int,
    frame_end: int,
    num_frames: int,
) -> list[int]:
    """Return internal 0-based sampled indices constrained to the signing span."""
    start_idx, end_idx = clip_frame_span(frame_count, frame_start, frame_end)
    return uniform_sample_indices(start_idx, end_idx, num_frames)


def uniform_sample_indices(start_idx: int, end_idx: int, num_frames: int) -> list[int]:
    if num_frames not in ALLOWED_FRAME_COUNTS:
        raise ValueError(f"num_frames must be one of {ALLOWED_FRAME_COUNTS}")
    if end_idx < start_idx:
        end_idx = start_idx
    span = np.linspace(start_idx, end_idx, num=num_frames)
    return [int(round(value)) for value in span]


def _read_frame_bgr(capture: cv2.VideoCapture, index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
    ok, frame = capture.read()
    if ok and frame is not None:
        return frame
    return None


def sample_video_frames(
    video_path: Path,
    num_frames: int = 16,
    frame_start: int = 1,
    frame_end: int = -1,
) -> list[np.ndarray]:
    """Decode only the uniformly sampled frames (BGR uint8)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = reported if reported > 0 else 0
        if frame_count <= 0:
            # Sequential count fallback for containers that omit frame_count.
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                frame_count += 1
            capture.release()
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Unable to reopen video: {video_path}")
        if frame_count <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

        start_idx, end_idx = clip_frame_span(frame_count, frame_start, frame_end)
        indices = uniform_sample_indices(start_idx, end_idx, num_frames)

        needed = sorted(set(indices))
        decoded: dict[int, np.ndarray] = {}
        sequential_needed = False
        for index in needed:
            frame = _read_frame_bgr(capture, index)
            if frame is None:
                sequential_needed = True
                break
            decoded[index] = frame

        if sequential_needed:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current = 0
            needed_set = set(needed)
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
                # Pad with black if even the first frame failed.
                height = next(iter(decoded.values())).shape[0] if decoded else 224
                width = next(iter(decoded.values())).shape[1] if decoded else 224
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            frames.append(frame)
            last = frame
        if len(frames) < num_frames and frames:
            while len(frames) < num_frames:
                frames.append(frames[-1])
        return frames
    finally:
        capture.release()


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


class WLASL100VideoDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        num_frames: int = 16,
        transform: Callable | None = None,
        training: bool = False,
    ) -> None:
        if num_frames not in ALLOWED_FRAME_COUNTS:
            raise ValueError(f"num_frames must be one of {ALLOWED_FRAME_COUNTS}")
        self.rows = list(rows)
        self.num_frames = num_frames
        self.transform = transform
        self.training = training

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        row = self.rows[index]
        path = resolve_project_path(row["video_path"])
        frames_bgr = sample_video_frames(
            path,
            num_frames=self.num_frames,
            frame_start=_as_int(row.get("frame_start"), 1),
            frame_end=_as_int(row.get("frame_end"), -1),
        )
        tensors: list[torch.Tensor] = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            if self.transform is not None:
                tensors.append(self.transform(image))
            else:
                tensors.append(transforms.functional.to_tensor(image))
        clip = torch.stack(tensors, dim=0)
        label = _as_int(row["class_id"])
        meta = {
            "video_id": row["video_id"],
            "gloss": row["gloss"],
            "split": row.get("split", ""),
            "path": str(path),
        }
        return clip, label, meta


def build_word_dataloader(
    split: str,
    cfg: WordModelConfig | None = None,
    *,
    shuffle: bool | None = None,
    batch_size: int | None = None,
) -> tuple[DataLoader, WLASL100VideoDataset]:
    cfg = cfg or default_config()
    rows = rows_for_split(load_manifest_rows(cfg.manifest_path), split)
    training = split == "train"
    transform = (
        word_train_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
        if training
        else word_eval_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    )
    dataset = WLASL100VideoDataset(
        rows,
        num_frames=cfg.frame_count,
        transform=transform,
        training=training,
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
