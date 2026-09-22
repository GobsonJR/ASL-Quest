"""Inference for a single ASL Citizen video (not integrated with frontend)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.checkpoint import load_checkpoint
from src.asl_citizen.config import BEST_CHECKPOINT_PATH
from src.asl_citizen.dataset import decode_sampled_frames, get_frame_count
from src.asl_citizen.model import build_model
from src.asl_citizen.preprocessing import eval_spatial_transforms, uniform_temporal_indices
from src.utils.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict isolated ASL sign from one video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=BEST_CHECKPOINT_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def build_top_k(probs: torch.Tensor, idx_to_class: dict[str, str], k: int) -> list[dict]:
    k = min(k, probs.shape[0])
    values, indices = torch.topk(probs, k=k)
    return [
        {
            "gloss": idx_to_class[str(int(idx))],
            "confidence": float(conf),
        }
        for conf, idx in zip(values.tolist(), indices.tolist())
    ]


@torch.no_grad()
def predict_video(video: Path, checkpoint: Path, top_k: int = 5) -> dict:
    device = get_device()
    payload, cfg, _class_to_idx, idx_to_class = load_checkpoint(checkpoint, map_location=device)
    transform = eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video}")
    try:
        frame_count = get_frame_count(capture)
    finally:
        capture.release()
    indices = uniform_temporal_indices(frame_count, cfg.num_frames)
    frames_bgr = decode_sampled_frames(video, indices)

    tensors = []
    for frame in frames_bgr:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensors.append(transform(Image.fromarray(rgb)))
    clip = torch.stack(tensors, dim=0).unsqueeze(0).to(device)

    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    logits = model(clip)
    probs = torch.softmax(logits, dim=1)[0]
    top_predictions = build_top_k(probs, idx_to_class, top_k)
    best = top_predictions[0]
    return {
        "prediction": best["gloss"],
        "confidence": best["confidence"],
        "top_k": top_predictions,
        "video": str(video),
        "num_frames": cfg.num_frames,
        "image_size": cfg.image_size,
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint at {args.checkpoint}")
    print(json.dumps(predict_video(args.video, args.checkpoint, args.top_k), indent=2))


if __name__ == "__main__":
    main()
