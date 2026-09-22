"""Predict a single WLASL100 word from one video file."""

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

from src.word_model.checkpoint import load_word_checkpoint
from src.word_model.config import WORD_BEST_MODEL_PATH
from src.word_model.dataset import sample_video_frames, word_eval_transforms
from src.word_model.model import build_word_model
from src.word_model.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one isolated ASL word video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=WORD_BEST_MODEL_PATH)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def build_top_k(probs: torch.Tensor, class_names: list[str], k: int) -> list[dict]:
    k = min(k, len(class_names))
    values, indices = torch.topk(probs, k=k)
    return [
        {
            "word": class_names[int(idx)],
            "class_id": int(idx),
            "confidence": float(conf),
        }
        for conf, idx in zip(values.tolist(), indices.tolist())
    ]


@torch.no_grad()
def predict_video(
    video: Path,
    checkpoint: Path,
    frame_start: int = 1,
    frame_end: int = -1,
    top_k: int = 5,
) -> dict:
    device = get_device()
    payload, cfg, class_names = load_word_checkpoint(checkpoint, map_location=device)
    transform = word_eval_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    frames = sample_video_frames(video, cfg.frame_count, frame_start, frame_end)
    tensors = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensors.append(transform(Image.fromarray(rgb)))
    clip = torch.stack(tensors, dim=0).unsqueeze(0).to(device)

    model = build_word_model(cfg, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    logits = model(clip)
    probs = torch.softmax(logits, dim=1)[0]
    top_predictions = build_top_k(probs, class_names, top_k)
    best = top_predictions[0]
    return {
        "word": best["word"],
        "prediction": best["word"],
        "class_id": best["class_id"],
        "confidence": best["confidence"],
        "top_k": top_predictions,
        "video": str(video),
        "frame_count": cfg.frame_count,
        "image_size": cfg.image_size,
        "seed": cfg.seed,
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"No word-model checkpoint at {args.checkpoint}. Train before predicting.")
    print(json.dumps(predict_video(args.video, args.checkpoint, args.frame_start, args.frame_end, args.top_k), indent=2))


if __name__ == "__main__":
    main()
