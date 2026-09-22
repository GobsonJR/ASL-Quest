"""Shared helpers for the ASL Citizen pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.asl_citizen.config import (
    ASL_CITIZEN_SPLITS_DIR,
    CLASS_TO_IDX_PATH,
    IDX_TO_CLASS_PATH,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    default_config,
)
from src.utils.device import describe_device, get_device, seed_everything

MANIFEST_COLUMNS = [
    "participant_id",
    "video_file",
    "gloss",
    "asl_lex_code",
    "class_idx",
    "split",
]

SOURCE_CSV_COLUMNS = ["Participant ID", "Video file", "Gloss", "ASL-LEX Code"]


def load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_mapping(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_class_mappings(
    class_to_idx_path: Path | None = None,
    idx_to_class_path: Path | None = None,
) -> tuple[dict[str, int], dict[str, str]]:
    class_to_idx_raw = load_json_mapping(class_to_idx_path or CLASS_TO_IDX_PATH)
    idx_to_class_raw = load_json_mapping(idx_to_class_path or IDX_TO_CLASS_PATH)
    class_to_idx = {str(k): int(v) for k, v in class_to_idx_raw.items()}
    idx_to_class = {str(k): str(v) for k, v in idx_to_class_raw.items()}
    if len(class_to_idx) != len(idx_to_class):
        raise ValueError("class_to_idx and idx_to_class lengths differ")
    for gloss, idx in class_to_idx.items():
        if idx_to_class.get(str(idx)) != gloss:
            raise ValueError(f"Inconsistent mapping for gloss={gloss} idx={idx}")
    return class_to_idx, idx_to_class


def resolve_video_path(video_root: Path, video_file: str) -> Path:
    path = video_root / video_file
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    missing = [col for col in MANIFEST_COLUMNS if col not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"Manifest {path} missing columns: {missing}")
    return rows


def load_source_split_rows(split_name: str) -> list[dict[str, str]]:
    path = ASL_CITIZEN_SPLITS_DIR / f"{split_name}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [col for col in SOURCE_CSV_COLUMNS if col not in fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    "participant_id": row["Participant ID"].strip(),
                    "video_file": row["Video file"].strip(),
                    "gloss": row["Gloss"].strip(),
                    "asl_lex_code": row["ASL-LEX Code"].strip(),
                    "split": split_name,
                }
            )
        return rows


def load_all_source_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        rows.extend(load_source_split_rows(split))
    return rows


def manifest_paths(cfg=None) -> dict[str, Path]:
    cfg = cfg or default_config()
    return {
        "train": cfg.train_manifest,
        "val": cfg.val_manifest,
        "test": cfg.test_manifest,
    }


def choose_batch_size(requested: int, device=None) -> int:
    import torch

    device = device or get_device()
    size = max(int(requested), 1)
    if device.type != "cuda":
        return min(size, 2)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb <= 6.5:
        return min(size, 4)
    return min(size, 8)


def describe_runtime_device() -> dict[str, Any]:
    return describe_device(get_device())


def seed_pipeline(seed: int) -> None:
    seed_everything(seed)
