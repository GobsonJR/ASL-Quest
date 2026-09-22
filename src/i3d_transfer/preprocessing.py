"""I3D-specific preprocessing, matching the OFFICIAL microsoft/ASL-citizen-code
preprocessing exactly (verified by reading I3D/aslcitizen_dataset.py and
I3D/videotransforms.py from the source repository, not inferred).

Deliberately NOT reused from src/asl_citizen/preprocessing.py (the ResNet18+GRU
pipeline's transforms): the I3D checkpoint was trained with a different frame count,
different resize/crop policy, and critically a different color space and value range,
so blindly reusing the ResNet pipeline's transforms would silently feed the pretrained
I3D weights out-of-distribution input.

Confirmed from the official source (I3D/aslcitizen_dataset.py:
load_rgb_frames_from_video):
  - Frames are read with cv2 and NEVER converted from BGR to RGB. The I3D checkpoint
    was trained on BGR frames. This module preserves that (it does not call
    cv2.cvtColor), unlike src/asl_citizen/dataset.py which explicitly converts to RGB.
  - Up to 64 frames per clip, with an adaptive frameskip based on total frame count:
    total_frames >= 160 -> skip 3, >= 96 -> skip 2, else -> skip 1.
  - The sampling window is centered on the middle of the video (not the start).
  - Frames are resized so the short side is >= 226 and the long side is <= 256,
    preserving aspect ratio (NOT a square resize).
  - Pixel values are mapped from [0, 255] to [-1, 1] via (img / 255.) * 2 - 1
    (NOT ImageNet mean/std normalization).
  - Short clips are padded (by repeating the first or last frame) up to 64 frames.
  - Official spatial crop: RandomCrop(224) + RandomHorizontalFlip for train,
    CenterCrop(224) for eval (I3D/videotransforms.py).

Deliberate deviation from the official pipeline, documented here per Phase 7 Task 4:
this module NEVER applies RandomHorizontalFlip, even for the (unused, backbone-frozen)
train path, matching src/asl_citizen/preprocessing.py's existing project-wide
convention that ASL handedness makes horizontal flipping semantically wrong for a
sign-language clip. Center-cropping is additionally used for ALL splits (not just
eval) in the frozen-feature-extraction scripts that call this module, because a linear
probe trained on 149 examples benefits more from stable, reproducible input features
than from crop augmentation of a frozen backbone's *input* pixels; this is documented
again at the call site.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

I3D_MAX_FRAMES = 64
I3D_CROP_SIZE = 224
I3D_MIN_SHORT_SIDE = 226
I3D_MAX_LONG_SIDE = 256


def official_frameskip(total_frames: int) -> int:
    """Exact port of the frameskip logic in load_rgb_frames_from_video."""
    frameskip = 1
    if total_frames >= 96:
        frameskip = 2
    if total_frames >= 160:
        frameskip = 3
    return frameskip


def official_start_frame(total_frames: int, frameskip: int) -> int:
    """Exact port of the centered start-frame logic in load_rgb_frames_from_video."""
    if frameskip == 3:
        return int(np.clip(int((total_frames - 192) // 2), 0, 160))
    if frameskip == 2:
        return int(np.clip(int((total_frames - 128) // 2), 0, 96))
    return int(np.clip(int((total_frames - 64) // 2), 0, 64))


def load_rgb_frames_from_video(video_path: Path, max_frames: int = I3D_MAX_FRAMES) -> np.ndarray:
    """Port of I3D/aslcitizen_dataset.py's load_rgb_frames_from_video, unchanged
    except for accepting a Path and being isolated into this module. Despite the
    upstream function's name, frames stay in BGR (cv2's native order) — see module
    docstring. Output dtype float32, value range [-1, 1], shape (T, H, W, 3)."""
    vidcap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    frameskip = official_frameskip(total_frames)
    start = official_start_frame(total_frames, frameskip)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

    try:
        for offset in range(0, min(max_frames * frameskip, int(total_frames - start))):
            success, img = vidcap.read()
            if not success or img is None:
                break
            if offset % frameskip == 0:
                w, h, c = img.shape
                if w < I3D_MIN_SHORT_SIDE or h < I3D_MIN_SHORT_SIDE:
                    d = float(I3D_MIN_SHORT_SIDE) - min(w, h)
                    sc = 1 + d / min(w, h)
                    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
                if w > I3D_MAX_LONG_SIDE or h > I3D_MAX_LONG_SIDE:
                    img = cv2.resize(img, (math.ceil(w * (I3D_MAX_LONG_SIDE / w)), math.ceil(h * (I3D_MAX_LONG_SIDE / h))))
                img = (img / 255.0) * 2 - 1
                frames.append(img)
    finally:
        vidcap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return np.asarray(frames, dtype=np.float32)


def pad_to_length(imgs: np.ndarray, total_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Port of ASLCitizen.pad: pads a short clip up to total_frames by repeating
    either the first or last frame (chosen randomly, matching the official code)."""
    if imgs.shape[0] >= total_frames:
        return imgs
    num_padding = total_frames - imgs.shape[0]
    if rng.random() > 0.5:
        pad_img = imgs[0]
    else:
        pad_img = imgs[-1]
    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
    return np.concatenate([imgs, pad], axis=0)


def center_crop(imgs: np.ndarray, size: int = I3D_CROP_SIZE) -> np.ndarray:
    """Port of videotransforms.CenterCrop. imgs shape (T, H, W, C)."""
    t, h, w, c = imgs.shape
    th, tw = size, size
    i = int(np.round((h - th) / 2.0))
    j = int(np.round((w - tw) / 2.0))
    return imgs[:, i : i + th, j : j + tw, :]


def video_to_tensor_array(imgs: np.ndarray) -> np.ndarray:
    """(T, H, W, C) float32 in [-1, 1] -> (C, T, H, W) float32, matching the official
    video_to_tensor's axis order (kept as numpy here; converted to a torch tensor by
    the caller so this module has no torch import)."""
    return imgs.transpose(3, 0, 1, 2)


def load_i3d_clip(video_path: Path, rng: np.random.Generator) -> np.ndarray:
    """Full official-equivalent pipeline for ONE clip, center-cropped (used for every
    split in this experiment — see module docstring for why train also uses
    center-crop here, deviating from the official RandomCrop+flip). Returns
    (C=3, T=64, H=224, W=224) float32 in [-1, 1], BGR channel order.
    """
    frames = load_rgb_frames_from_video(video_path, I3D_MAX_FRAMES)
    frames = pad_to_length(frames, I3D_MAX_FRAMES, rng)
    frames = center_crop(frames, I3D_CROP_SIZE)
    return video_to_tensor_array(frames)
