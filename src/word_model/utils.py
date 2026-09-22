"""Helpers for the WLASL100 word pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from src.config import PROJECT_ROOT
from src.word_model.config import CLASSES_PATH, MANIFEST_PATH

MANIFEST_COLUMNS = [
    "video_id",
    "video_path",
    "gloss",
    "class_id",
    "split",
    "signer_id",
    "source",
    "fps",
    "frame_start",
    "frame_end",
    "variation_id",
]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_class_names(path: Path | None = None) -> list[str]:
    target = path or CLASSES_PATH
    if not target.exists():
        raise FileNotFoundError(f"WLASL100 class list not found: {target}")
    names: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            _, name = line.split("\t", 1)
            names.append(name.strip())
        else:
            names.append(line)
    if len(names) != 100:
        raise ValueError(f"Expected 100 WLASL100 classes, found {len(names)} in {target}")
    return names


def load_manifest_rows(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or MANIFEST_PATH
    if not target.exists():
        raise FileNotFoundError(f"WLASL100 manifest not found: {target}. Run scripts/prepare_wlasl100.py first.")
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Empty manifest: {target}")
    missing = [col for col in MANIFEST_COLUMNS if col not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"Manifest missing columns {missing}")
    return rows


def rows_for_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    wanted = split.strip().lower()
    return [row for row in rows if str(row.get("split", "")).strip().lower() == wanted]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_cuda() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": None,
        "gpu_memory_gb": None,
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
    return info


def choose_batch_size(requested: int, device: torch.device) -> int:
    """Pick a conservative batch size for a 6 GB laptop GPU."""
    size = max(int(requested), 1)
    if device.type != "cuda":
        return min(size, 2)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb <= 6.5:
        return min(size, 4)
    return min(size, 8)


def estimate_vram_mb(batch_size: int, frame_count: int, image_size: int, feature_dim: int = 512, hidden_size: int = 256) -> float:
    """Rough activation + param estimate in MiB (not a profiler)."""
    input_elems = batch_size * frame_count * 3 * image_size * image_size
    feat_elems = batch_size * frame_count * feature_dim
    hidden_elems = batch_size * hidden_size
    # fp16 activations + fp32 params (~12M for ResNet18 + small GRU)
    activation_bytes = (input_elems + feat_elems + hidden_elems) * 2
    param_bytes = 12_000_000 * 4
    return round((activation_bytes + param_bytes) / (1024**2), 1)
