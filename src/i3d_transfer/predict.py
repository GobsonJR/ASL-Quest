"""Single-clip inference for the native-10 frozen-I3D + linear-head pipeline.

This chains two EXISTING, already-validated components exactly as they are
used in the Phase 7 research scripts — no new preprocessing, architecture, or
training logic is introduced here:

  - Backbone loading: matches scripts/extract_i3d_features_native_10.py's
    build_model() (InceptionI3d(400) -> replace_logits(2731) -> load the
    official ASL Citizen checkpoint -> eval() -> frozen).
  - Preprocessing: reuses src/i3d_transfer/preprocessing.py::load_i3d_clip
    unchanged (the official-equivalent BGR, [-1, 1], center-crop pipeline).
  - Feature pooling: temporal mean-pool of extract_features(), matching the
    extraction script's documented choice.
  - Head architecture/checkpoint format: matches
    scripts/train_native_10_i3d_frozen.py's LinearHead and its
    torch.save({"model_state_dict", "class_to_idx", "idx_to_class", "config", ...})
    payload exactly.

Isolated from src/inference/engine.py (the A-Z ResNet18 path) by construction:
different module, different model classes, different checkpoint files. Nothing
here is imported by, or imports from, the A-Z inference path.

Not wired into the FastAPI backend yet — see the validation script instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.i3d_transfer.preprocessing import load_i3d_clip
from src.i3d_transfer.pytorch_i3d import InceptionI3d

_ROOT = Path(__file__).resolve().parents[2]
I3D_BACKBONE_CHECKPOINT = _ROOT / "models" / "pretrained" / "asl_citizen_i3d" / "ASL_citizen_I3D_weights.pt"
DEFAULT_HEAD_CHECKPOINT = (
    _ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "i3d_frozen" / "checkpoints" / "head_best.pt"
)


class LinearHead(nn.Module):
    """Exact architecture match for scripts/train_native_10_i3d_frozen.py::LinearHead."""

    def __init__(self, feature_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x))


def _build_backbone(device: torch.device, checkpoint: Path) -> InceptionI3d:
    """Load the frozen ASL Citizen I3D backbone, matching
    scripts/extract_i3d_features_native_10.py::build_model exactly."""
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(2731)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_result = i3d.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "I3D backbone checkpoint/architecture mismatch: "
            f"missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}"
        )
    i3d.to(device)
    i3d.eval()
    for p in i3d.parameters():
        p.requires_grad_(False)
    return i3d


def _build_head(device: torch.device, checkpoint: Path) -> tuple[LinearHead, dict[str, str]]:
    """Load the trained native-10 linear head from a checkpoint written by
    scripts/train_native_10_i3d_frozen.py (head_best.pt / head_last.pt)."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"Native-10 head checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    idx_to_class = {str(k): str(v) for k, v in payload["idx_to_class"].items()}
    head = LinearHead(cfg["feature_dim"], cfg["num_classes"], cfg["feature_dropout"])
    head.load_state_dict(payload["model_state_dict"])
    head.to(device)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head, idx_to_class


class NativeSignPredictor:
    """Loads the frozen I3D backbone and the trained linear head ONCE, then
    serves repeated single-clip predictions without reloading either model."""

    def __init__(
        self,
        head_checkpoint: Path = DEFAULT_HEAD_CHECKPOINT,
        backbone_checkpoint: Path = I3D_BACKBONE_CHECKPOINT,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_checkpoint = backbone_checkpoint
        self.head_checkpoint = head_checkpoint
        self.backbone = _build_backbone(self.device, backbone_checkpoint)
        self.head, self.idx_to_class = _build_head(self.device, head_checkpoint)
        self.num_classes = len(self.idx_to_class)

    @torch.no_grad()
    def predict_video(self, video_path: Path, top_k: int = 5) -> dict[str, Any]:
        rng = np.random.default_rng(42)
        clip = load_i3d_clip(Path(video_path), rng)  # existing preprocessing, unmodified: (C, T, H, W)
        batch = torch.from_numpy(clip).unsqueeze(0).to(self.device)  # (1, C, T, H, W)

        pooled = self.backbone.extract_features(batch)  # (1, 1024, T', 1, 1)
        feature = pooled.mean(dim=2).squeeze(-1).squeeze(-1)  # temporal mean-pool -> (1, 1024)

        logits = self.head(feature)
        probs = torch.softmax(logits, dim=1)[0]

        k = min(top_k, probs.shape[0])
        values, indices = torch.topk(probs, k=k)
        top_predictions = [
            {"gloss": self.idx_to_class[str(int(idx))], "confidence": float(conf)}
            for conf, idx in zip(values.tolist(), indices.tolist())
        ]
        best = top_predictions[0]
        return {
            "prediction": best["gloss"],
            "confidence": best["confidence"],
            "top_k": top_predictions,
            "video": str(video_path),
            "num_classes": self.num_classes,
        }


_PREDICTORS: dict[tuple[str, str], NativeSignPredictor] = {}


def get_predictor(
    checkpoint: Path = DEFAULT_HEAD_CHECKPOINT,
    backbone_checkpoint: Path = I3D_BACKBONE_CHECKPOINT,
) -> NativeSignPredictor:
    """Process-wide cache so the backbone/head are loaded once per checkpoint
    pair and reused across predictions, instead of reloading on every call."""
    key = (str(checkpoint), str(backbone_checkpoint))
    if key not in _PREDICTORS:
        _PREDICTORS[key] = NativeSignPredictor(head_checkpoint=checkpoint, backbone_checkpoint=backbone_checkpoint)
    return _PREDICTORS[key]


def predict_video(video_path: Path, checkpoint: Path = DEFAULT_HEAD_CHECKPOINT, top_k: int = 5) -> dict[str, Any]:
    """Public single-clip inference entry point, mirroring
    src/asl_citizen/predict.py::predict_video's return shape.

    Reuses a cached NativeSignPredictor (backbone + head loaded once) rather
    than reloading either model on every call.
    """
    predictor = get_predictor(checkpoint=checkpoint)
    return predictor.predict_video(video_path, top_k=top_k)
