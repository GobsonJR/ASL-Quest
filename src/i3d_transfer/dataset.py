"""Raw-clip dataset for I3D partial fine-tuning (Phase 8).

Phase 7's feature-extraction script cached POST-Mixed_5c pooled features, which is
correct for a frozen-backbone linear probe but useless for fine-tuning Mixed_5c
itself (fine-tuning needs the PRE-Mixed_5c activations, recomputed every forward pass
since Mixed_5c's weights change during training). This dataset therefore decodes raw
video clips through the identical, unmodified official-preprocessing pipeline from
src/i3d_transfer/preprocessing.py (same 64-frame/BGR/[-1,1]/center-crop-224 pipeline
used for feature extraction — nothing about preprocessing changes between Phase 7 and
Phase 8, per Phase 8's explicit instruction not to change it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.asl_citizen.utils import load_manifest_rows, resolve_video_path
from src.i3d_transfer.preprocessing import load_i3d_clip


class I3DClipDataset(Dataset):
    def __init__(self, manifest_path: Path, video_root: Path, class_to_idx: dict[str, int], seed: int = 42) -> None:
        self.rows = load_manifest_rows(manifest_path)
        self.video_root = video_root
        self.class_to_idx = dict(class_to_idx)
        self._seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        video_path = resolve_video_path(self.video_root, row["video_file"])
        # Deterministic per-sample RNG (index-derived) so the pad-with-first-or-last-
        # frame coin flip in load_i3d_clip is reproducible across epochs and matches
        # the deterministic, non-augmented spirit already documented for this
        # experiment's preprocessing (see preprocessing.py's module docstring).
        rng = np.random.default_rng(self._seed + index)
        clip = load_i3d_clip(video_path, rng)
        label = int(row.get("class_idx", self.class_to_idx[row["gloss"]]))
        return {
            "video": torch.from_numpy(clip),
            "label": label,
            "gloss": row["gloss"],
            "participant_id": row["participant_id"],
            "video_file": row["video_file"],
        }
