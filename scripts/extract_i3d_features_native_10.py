"""Phase 7 Task 3/5/6: extract FROZEN I3D features for every native-10 clip.

The I3D backbone is never put in train() mode and every forward pass here runs under
torch.no_grad() — there is no gradient path into the backbone anywhere in this script,
by construction, not by convention. This is the "PRETRAINED I3D -> FROZEN -> feature
extraction" half of Task 3; scripts/train_native_10_i3d_frozen.py trains only the
lightweight classifier head on the cached output of this script.

Temporal pooling choice (documented per Task 4): the official checkpoint's own
inference procedure (I3D/aslcitizen_testing.py) takes a temporal MAX over per-frame
*logits* for its own 2,731-way classifier. That is a decision about how to use the
model's own trained classifier head, which we discard entirely (we only call
extract_features(), never the original logits layer). For a frozen *feature* embedding
feeding a brand-new small classifier on 149 training examples, temporal MEAN-pooling
of the 1024-dim pre-classifier features is the more standard and lower-variance choice
(closer to global average pooling than to a per-frame arg-max convention that assumes
a *trained-for-this-task* classifier is scoring each frame). Both this choice and the
alternative are reported honestly here rather than silently picked: mean pooling is
used for the primary experiment.

Isolation: reads from data/asl_citizen_native_10_i3d/ (a copy of the native-10
manifest, not the original data/asl_citizen_native_10/). Writes only under
outputs/asl_citizen_native_10/experiments/i3d_frozen/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.config import ASL_CITIZEN_VIDEOS_DIR
from src.asl_citizen.utils import load_class_mappings, load_manifest_rows, resolve_video_path
from src.i3d_transfer.preprocessing import load_i3d_clip
from src.i3d_transfer.pytorch_i3d import InceptionI3d

DATA_DIR = ROOT / "data" / "asl_citizen_native_10_i3d"
CHECKPOINT = ROOT / "models" / "pretrained" / "asl_citizen_i3d" / "ASL_citizen_I3D_weights.pt"
OUT_FEATURES = ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "i3d_frozen" / "features"


def build_model(device: torch.device) -> InceptionI3d:
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(2731)
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    load_result = i3d.load_state_dict(state_dict, strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys, (
        f"checkpoint/architecture mismatch: missing={load_result.missing_keys} "
        f"unexpected={load_result.unexpected_keys}"
    )
    i3d.to(device)
    i3d.eval()
    for p in i3d.parameters():
        p.requires_grad_(False)
    return i3d


@torch.no_grad()
def extract_split(split: str, i3d: InceptionI3d, class_to_idx: dict[str, int], device: torch.device) -> dict:
    rows = load_manifest_rows(DATA_DIR / f"{split}.csv")
    rng = np.random.default_rng(42)

    features = np.zeros((len(rows), 1024), dtype=np.float32)
    labels = np.zeros(len(rows), dtype=np.int64)
    participants: list[str] = []
    glosses: list[str] = []
    video_files: list[str] = []

    started = time.perf_counter()
    for i, row in enumerate(rows):
        video_path = resolve_video_path(ASL_CITIZEN_VIDEOS_DIR, row["video_file"])
        clip = load_i3d_clip(video_path, rng)
        batch = torch.from_numpy(clip).unsqueeze(0).to(device)
        pooled = i3d.extract_features(batch)  # (1, 1024, T', 1, 1)
        feat = pooled.mean(dim=2).squeeze().cpu().numpy()  # temporal mean-pool -> (1024,)
        assert np.isfinite(feat).all(), f"non-finite feature for {row['video_file']}"

        features[i] = feat
        labels[i] = class_to_idx[row["gloss"]]
        participants.append(row["participant_id"])
        glosses.append(row["gloss"])
        video_files.append(row["video_file"])

        if (i + 1) % 25 == 0 or (i + 1) == len(rows):
            print(f"  [{split}] {i + 1}/{len(rows)} clips extracted")

    duration = time.perf_counter() - started
    OUT_FEATURES.mkdir(parents=True, exist_ok=True)
    out_path = OUT_FEATURES / f"{split}_features.npz"
    np.savez(
        out_path,
        features=features,
        labels=labels,
        participants=np.array(participants),
        glosses=np.array(glosses),
        video_files=np.array(video_files),
    )
    print(f"  wrote {out_path.relative_to(ROOT)} ({len(rows)} clips, {duration:.1f}s)")
    return {
        "split": split,
        "num_clips": len(rows),
        "feature_dim": 1024,
        "duration_sec": round(duration, 2),
        "output_path": str(out_path),
    }


def main() -> None:
    print("=" * 70)
    print("PHASE 7 TASK 3/6: FROZEN I3D FEATURE EXTRACTION")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class_to_idx, idx_to_class = load_class_mappings(
        DATA_DIR / "class_to_idx.json", DATA_DIR / "idx_to_class.json"
    )
    assert len(class_to_idx) == 10
    print("class mapping:", class_to_idx)

    i3d = build_model(device)
    trainable = sum(p.numel() for p in i3d.parameters() if p.requires_grad)
    total = sum(p.numel() for p in i3d.parameters())
    print(f"I3D backbone loaded: total_params={total} trainable_params={trainable} (must be 0 — frozen)")
    assert trainable == 0, "I3D backbone must be fully frozen for this experiment"

    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None

    summary = {"device": str(device), "temporal_pooling": "mean", "splits": {}}
    for split in ("train", "val", "test"):
        print(f"\nExtracting split: {split}")
        summary["splits"][split] = extract_split(split, i3d, class_to_idx, device)

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        summary["peak_gpu_memory_mb"] = round(peak_mb, 1)
        print(f"\nPeak GPU memory during extraction: {peak_mb:.1f} MB")

    summary_path = OUT_FEATURES / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary -> {summary_path}")
    print("\nFEATURE EXTRACTION COMPLETE.")


if __name__ == "__main__":
    main()
