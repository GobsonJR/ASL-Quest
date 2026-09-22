"""Phase 7 Task 4 preprocessing smoke test.

Verifies, on ONE real native-10 training clip, before any training or feature
extraction runs:
  - input tensor shape matches I3D's expected (C=3, T=64, H=224, W=224)
  - dtype is float32
  - all values are finite
  - temporal ordering is preserved (frames are not shuffled)
  - values are in the expected [-1, 1] range (I3D normalization, not ImageNet)
  - one real forward pass through the loaded checkpoint succeeds and produces
    finite logits and finite pooled features

Read-only: does not train anything, does not write any checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.config import ASL_CITIZEN_VIDEOS_DIR
from src.asl_citizen.utils import load_manifest_rows, resolve_video_path
from src.i3d_transfer.preprocessing import I3D_CROP_SIZE, I3D_MAX_FRAMES, load_i3d_clip
from src.i3d_transfer.pytorch_i3d import InceptionI3d

DATA_DIR = ROOT / "data" / "asl_citizen_native_10_i3d"
CHECKPOINT = ROOT / "models" / "pretrained" / "asl_citizen_i3d" / "ASL_citizen_I3D_weights.pt"


def main() -> None:
    print("=" * 70)
    print("PHASE 7 TASK 4: I3D PREPROCESSING SMOKE TEST")
    print("=" * 70)

    rows = load_manifest_rows(DATA_DIR / "train.csv")
    row = rows[0]
    video_path = resolve_video_path(ASL_CITIZEN_VIDEOS_DIR, row["video_file"])
    print(f"Sample clip: gloss={row['gloss']} participant={row['participant_id']} file={row['video_file']}")
    print(f"Resolved path: {video_path}")
    assert video_path.exists(), f"video not found: {video_path}"

    rng = np.random.default_rng(42)
    clip = load_i3d_clip(video_path, rng)

    print("\n--- Shape / dtype checks ---")
    print("shape:", clip.shape)
    assert clip.shape == (3, I3D_MAX_FRAMES, I3D_CROP_SIZE, I3D_CROP_SIZE), f"unexpected shape {clip.shape}"
    print("dtype:", clip.dtype)
    assert clip.dtype == np.float32

    print("\n--- Finite value check ---")
    finite = np.isfinite(clip).all()
    print("all finite:", bool(finite))
    assert finite

    print("\n--- Value range check (expected [-1, 1], I3D normalization) ---")
    print("min:", float(clip.min()), "max:", float(clip.max()))
    assert clip.min() >= -1.0 - 1e-5 and clip.max() <= 1.0 + 1e-5

    print("\n--- Temporal ordering check ---")
    # Frame-to-frame mean-abs-difference should be small and smoothly varying for a
    # real video (not the large, unstructured jumps that shuffled frames would show).
    frame_means = clip.mean(axis=(0, 2, 3))  # (T,)
    diffs = np.abs(np.diff(frame_means))
    print("per-frame channel-mean (first 5):", [round(float(v), 4) for v in frame_means[:5]])
    print("mean abs frame-to-frame delta:", round(float(diffs.mean()), 5))
    print("max abs frame-to-frame delta:", round(float(diffs.max()), 5))
    assert diffs.mean() < 0.5, "frame-to-frame deltas look unreasonably large for a real clip"

    print("\n--- BGR channel-order note ---")
    print(
        "Channel order is preserved as decoded by cv2 (BGR), matching the official "
        "training pipeline. No cv2.cvtColor call is present in "
        "src/i3d_transfer/preprocessing.py (verified by inspection, not re-asserted "
        "numerically here since BGR vs RGB is not distinguishable from pixel "
        "statistics alone)."
    )

    print("\n--- Real forward pass through the loaded checkpoint ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(2731)
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    load_result = i3d.load_state_dict(state_dict, strict=True)
    print("missing_keys:", load_result.missing_keys, "unexpected_keys:", load_result.unexpected_keys)
    i3d.to(device)
    i3d.eval()

    batch = torch.from_numpy(clip).unsqueeze(0).to(device)  # (1, 3, 64, 224, 224)
    print("batch tensor shape:", tuple(batch.shape), "dtype:", batch.dtype)

    with torch.no_grad():
        pooled = i3d.extract_features(batch)  # (1, 1024, 7, 1, 1)
        logits = i3d(batch, pretrained=False)  # (1, 2731, 7)

    print("pooled feature shape:", tuple(pooled.shape))
    print("logits shape:", tuple(logits.shape))
    assert torch.isfinite(pooled).all(), "pooled features contain non-finite values"
    assert torch.isfinite(logits).all(), "logits contain non-finite values"

    feature_vec = pooled.mean(dim=2).squeeze()  # temporal mean-pool -> (1024,)
    print("mean-pooled feature vector shape:", tuple(feature_vec.shape))
    print("feature vector stats: min=%.4f max=%.4f mean=%.4f" % (
        float(feature_vec.min()), float(feature_vec.max()), float(feature_vec.mean())
    ))

    top_gloss_logits = logits.mean(dim=2).squeeze()  # (2731,)
    top5 = torch.topk(top_gloss_logits, 5).indices.cpu().tolist()
    print("top-5 predicted class indices on the full 2731-way head (sanity only):", top5)

    print("\nSMOKE TEST PASSED — safe to proceed to full feature extraction.")


if __name__ == "__main__":
    main()
