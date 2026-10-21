"""Development-only capture of what the classifier REALLY received for a live prediction.

When ``ASL_QUEST_DEV_DIAGNOSTICS=1`` and a request asks for it (``/predict?debug=true&save_diagnostics=true``),
one folder is written per prediction, all produced inside that same request:

    1_original_frame.jpg|png   the exact bytes the server received (before any processing)
    2_hand_box_overlay.png     the frame with the MediaPipe landmark box (red) and the square crop window (green)
    3_classifier_crop.png      the exact image handed to the model (lossless)
    4_model_input_224.png      the tensor the network actually sees (resized + normalised), converted back
                               to a viewable image
    result.json                top-5 scores, box coordinates, sizes, client mirror flag, checkpoint path + hash

It is OFF by default and never exposes images through an endpoint: files go to a local folder
(``outputs/diagnostics/live_requests``, git-ignored) that keeps only the newest few requests. The images
can contain your hand and surroundings, so do not share the folder.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.config import IMAGENET_MEAN, IMAGENET_STD

_SHA_CACHE: dict[str, str] = {}


def file_sha256(path: Path) -> str:
    """SHA-256 of a checkpoint (cached; the file is read once)."""
    key = f"{path}:{path.stat().st_mtime_ns}"
    if key not in _SHA_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        _SHA_CACHE[key] = digest.hexdigest()
    return _SHA_CACHE[key]


def tensor_to_image(tensor, mean, std) -> Image.Image:
    """Undo Normalize(mean, std) on a (3, H, W) tensor so the model input can be looked at."""
    import torch

    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    arr = (tensor.cpu() * std_t + mean_t).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


class LiveDiagnostics:
    def __init__(self, root: Path, enabled: bool = False, keep: int = 60, min_interval_s: float = 0.5) -> None:
        self.root = root
        self.enabled = enabled
        self.keep = keep
        self.min_interval_s = min_interval_s
        self._last_saved = 0.0

    def save(
        self,
        *,
        payload: bytes,
        frame: Image.Image,
        source_image: Image.Image,
        hand_box: tuple[float, float, float, float] | None,
        crop_window: tuple[int, int, int, int] | None,
        predictor: Any,
        result: dict[str, Any],
        meta: dict[str, Any],
        force: bool = False,
    ) -> dict[str, str] | None:
        """Write one bundle. Returns {request_id, saved_to}, or None when disabled / rate-limited."""
        if not self.enabled:
            return None
        now = time.monotonic()
        if not force and now - self._last_saved < self.min_interval_s:
            return None
        self._last_saved = now

        request_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        folder = self.root / request_id
        folder.mkdir(parents=True, exist_ok=True)

        ext = "jpg" if (frame.format or "").upper() in ("JPEG", "MPO") else "png"
        (folder / f"1_original_frame.{ext}").write_bytes(payload)  # the exact received bytes

        overlay = frame.convert("RGB").copy()
        draw = ImageDraw.Draw(overlay)
        if hand_box:
            draw.rectangle(hand_box, outline=(255, 40, 40), width=3)
        if crop_window:
            draw.rectangle(crop_window, outline=(40, 220, 40), width=3)
        overlay.save(folder / "2_hand_box_overlay.png")

        source_image.convert("RGB").save(folder / "3_classifier_crop.png")  # exactly what predict_pil() received

        tensor = predictor.transform(source_image.convert("RGB"))
        tensor_to_image(tensor, IMAGENET_MEAN, IMAGENET_STD).save(folder / "4_model_input_224.png")

        checkpoint = Path(predictor.model_path)
        record = {
            "request_id": request_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "class_order_is_a_to_z": list(predictor.class_names) == [chr(65 + i) for i in range(26)],
            "device": str(predictor.device),
            "received_bytes": len(payload),
            "received_format": frame.format,
            "frame_size": [frame.width, frame.height],
            "landmark_box": [round(v, 1) for v in hand_box] if hand_box else None,
            "crop_window_clipped": list(crop_window) if crop_window else None,
            "classifier_crop_size": [source_image.width, source_image.height],
            "model_input_shape": list(tensor.shape),
            "model_input_channel_mean": [round(float(v), 3) for v in tensor.mean(dim=(1, 2))],
            "model_input_channel_std": [round(float(v), 3) for v in tensor.std(dim=(1, 2))],
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "low_confidence": result.get("low_confidence"),
            "top_predictions": result.get("top_predictions"),
            "client": meta,
        }
        (folder / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        self._prune()
        return {"request_id": request_id, "saved_to": str(folder)}

    def _prune(self) -> None:
        folders = sorted(p for p in self.root.iterdir() if p.is_dir())
        for old in folders[: max(0, len(folders) - self.keep)]:
            shutil.rmtree(old, ignore_errors=True)
