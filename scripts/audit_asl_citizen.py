"""Phase 7A: Read-only audit of the ASL Citizen dataset at D:\\ASL_Citizen.

Does NOT train, modify project assets, or extract full frame caches.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
# Overridable via ASL_CITIZEN_DATASET_ROOT (same convention as
# src/asl_citizen/config.py) since this licensed dataset's location is
# machine-specific and never distributed with the repo.
DATASET_ROOT = Path(os.getenv("ASL_CITIZEN_DATASET_ROOT", r"D:\ASL_Citizen\ASL_Citizen"))
OUTPUT_DIR = ROOT / "outputs" / "asl_citizen_audit"
SPLITS_DIR = DATASET_ROOT / "splits"
VIDEOS_DIR = DATASET_ROOT / "videos"

SPLIT_FILES = {
    "train": SPLITS_DIR / "train.csv",
    "val": SPLITS_DIR / "val.csv",
    "test": SPLITS_DIR / "test.csv",
}

CSV_COLUMNS = ["Participant ID", "Video file", "Gloss", "ASL-LEX Code"]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_split_rows(split_name: str) -> list[dict[str, str]]:
    path = SPLIT_FILES[split_name]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [col for col in CSV_COLUMNS if col not in fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        rows = []
        for row in reader:
            rows.append(
                {
                    "split": split_name,
                    "participant_id": row["Participant ID"].strip(),
                    "video_file": row["Video file"].strip(),
                    "gloss": row["Gloss"].strip(),
                    "asl_lex_code": row["ASL-LEX Code"].strip(),
                }
            )
        return rows


def video_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if "-" in stem:
        return stem.split("-", 1)[0]
    return stem


def stat_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
    }


def probe_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {
            "path": str(path),
            "decodable": False,
            "error": "unable_to_open",
        }
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (frame_count / fps) if fps > 0 and frame_count > 0 else None

        if frame_count <= 0:
            # Sequential fallback for containers missing frame_count metadata.
            frame_count = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                frame_count += 1
            duration = (frame_count / fps) if fps > 0 else None

        ok_first, first_frame = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count - 1, 0))
        ok_last, last_frame = capture.read()

        return {
            "path": str(path),
            "decodable": frame_count > 0,
            "fps": round(fps, 3) if fps else None,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_sec": round(duration, 4) if duration is not None else None,
            "resolution": f"{width}x{height}" if width and height else None,
            "first_frame_ok": bool(ok_first and first_frame is not None),
            "last_frame_ok": bool(ok_last and last_frame is not None),
        }
    finally:
        capture.release()


def sample_rows(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, n)


def hash_file_partial(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_class_distribution(all_rows: list[dict]) -> tuple[list[dict], dict]:
    by_gloss: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_gloss[row["gloss"]].append(row)

    distribution = []
    for gloss, items in sorted(by_gloss.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        split_counts = Counter(item["split"] for item in items)
        participants = sorted({item["participant_id"] for item in items})
        lex_codes = sorted({item["asl_lex_code"] for item in items})
        distribution.append(
            {
                "gloss": gloss,
                "video_count": len(items),
                "train_count": split_counts.get("train", 0),
                "val_count": split_counts.get("val", 0),
                "test_count": split_counts.get("test", 0),
                "participant_count": len(participants),
                "asl_lex_codes": lex_codes,
            }
        )

    counts = [entry["video_count"] for entry in distribution]
    summary = {
        "total_classes": len(distribution),
        "total_videos": len(all_rows),
        "min_videos_per_class": min(counts) if counts else 0,
        "max_videos_per_class": max(counts) if counts else 0,
        "mean_videos_per_class": round(statistics.mean(counts), 4) if counts else 0,
        "median_videos_per_class": round(statistics.median(counts), 4) if counts else 0,
        "top_30": distribution[:30],
        "classes_with_leq_5_videos": [e for e in distribution if e["video_count"] <= 5],
        "classes_with_ge_20_videos": [e for e in distribution if e["video_count"] >= 20],
    }
    return distribution, summary


def subset_candidate(all_distribution: list[dict], all_rows: list[dict], k: int) -> dict:
    ranked = sorted(all_distribution, key=lambda e: (-e["video_count"], e["gloss"]))
    selected = ranked[:k]
    selected_glosses = {e["gloss"] for e in selected}
    subset_rows = [r for r in all_rows if r["gloss"] in selected_glosses]
    counts = [e["video_count"] for e in selected]
    split_counts = Counter(r["split"] for r in subset_rows)
    return {
        "target_classes": k,
        "actual_classes": len(selected),
        "total_videos": len(subset_rows),
        "avg_videos_per_class": round(statistics.mean(counts), 4) if counts else 0,
        "min_videos_per_class": min(counts) if counts else 0,
        "max_videos_per_class": max(counts) if counts else 0,
        "train_videos": split_counts.get("train", 0),
        "val_videos": split_counts.get("val", 0),
        "test_videos": split_counts.get("test", 0),
        "selection_rule": f"top {k} classes by total video count across all splits",
    }


def signer_split_analysis(all_rows: list[dict]) -> dict:
    by_split_participants: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        by_split_participants[row["split"]].add(row["participant_id"])

    train_p = by_split_participants.get("train", set())
    val_p = by_split_participants.get("val", set())
    test_p = by_split_participants.get("test", set())

    return {
        "train_participants": sorted(train_p),
        "val_participants": sorted(val_p),
        "test_participants": sorted(test_p),
        "train_participant_count": len(train_p),
        "val_participant_count": len(val_p),
        "test_participant_count": len(test_p),
        "train_val_overlap": sorted(train_p & val_p),
        "train_test_overlap": sorted(train_p & test_p),
        "val_test_overlap": sorted(val_p & test_p),
        "all_three_overlap": sorted(train_p & val_p & test_p),
        "signer_independent_split": not (train_p & val_p or train_p & test_p or val_p & test_p),
        "note": (
            "Official ASL Citizen split is signer-independent: each participant appears in exactly one split."
            if not (train_p & val_p or train_p & test_p or val_p & test_p)
            else "Participant IDs overlap between splits; split is NOT signer-independent."
        ),
    }


def inventory_dataset() -> dict:
    all_paths = list(DATASET_ROOT.rglob("*"))
    files = [p for p in all_paths if p.is_file()]
    dirs = [p for p in all_paths if p.is_dir()]
    video_files = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    json_files = [p for p in files if p.suffix.lower() == ".json"]
    csv_files = [p for p in files if p.suffix.lower() == ".csv"]
    total_bytes = sum(p.stat().st_size for p in files)
    ext_counts = Counter(p.suffix.lower() for p in files)
    return {
        "dataset_root": str(DATASET_ROOT),
        "total_files": len(files),
        "total_directories": len(dirs),
        "total_video_files": len(video_files),
        "total_json_files": len(json_files),
        "total_csv_files": len(csv_files),
        "total_disk_usage_bytes": total_bytes,
        "total_disk_usage_gb": round(total_bytes / (1024**3), 3),
        "file_extension_counts": dict(sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "structure": {
            "top_level": sorted(p.name for p in DATASET_ROOT.iterdir()),
            "videos_layout": "flat directory of .mp4 files",
            "splits_layout": "train.csv, val.csv, test.csv",
            "metadata_files": [str(p.relative_to(DATASET_ROOT)) for p in sorted(csv_files + json_files)],
        },
    }


def duplicate_analysis(all_rows: list[dict], video_files: list[Path]) -> dict:
    # Video ID / filename uniqueness from metadata
    video_file_counts = Counter(r["video_file"] for r in all_rows)
    duplicate_filenames = {k: v for k, v in video_file_counts.items() if v > 1}

    video_id_map: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        vid = video_id_from_filename(row["video_file"])
        video_id_map[vid].append(row)
    duplicate_video_ids = {k: v for k, v in video_id_map.items() if len(v) > 1}

    # Cross-split filename presence
    file_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        file_to_splits[row["video_file"]].add(row["split"])
    cross_split_files = {k: sorted(v) for k, v in file_to_splits.items() if len(v) > 1}

    # On-disk duplicate filenames
    stem_counts = Counter(p.name for p in video_files)
    duplicate_on_disk = {k: v for k, v in stem_counts.items() if v > 1}

    # Partial hash sample for identical file content
    rng = random.Random(42)
    sample_paths = rng.sample(video_files, min(500, len(video_files)))
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for path in sample_paths:
        hash_groups[hash_file_partial(path)].append(path.name)
    identical_hash_groups = {h: names for h, names in hash_groups.items() if len(names) > 1}

    # Missing files referenced in CSV
    on_disk_names = {p.name for p in video_files}
    missing = sorted({r["video_file"] for r in all_rows if r["video_file"] not in on_disk_names})
    extra_on_disk = sorted(on_disk_names - {r["video_file"] for r in all_rows})

    return {
        "metadata_duplicate_video_files": duplicate_filenames,
        "metadata_duplicate_video_file_count": len(duplicate_filenames),
        "metadata_duplicate_video_ids": duplicate_video_ids,
        "metadata_duplicate_video_id_count": len(duplicate_video_ids),
        "cross_split_duplicate_filenames": cross_split_files,
        "cross_split_duplicate_filename_count": len(cross_split_files),
        "on_disk_duplicate_filenames": duplicate_on_disk,
        "on_disk_duplicate_filename_count": len(duplicate_on_disk),
        "partial_hash_sample_size": len(sample_paths),
        "partial_hash_identical_groups": identical_hash_groups,
        "partial_hash_identical_group_count": len(identical_hash_groups),
        "csv_referenced_missing_on_disk": missing[:20],
        "csv_referenced_missing_on_disk_count": len(missing),
        "on_disk_not_in_csv_count": len(extra_on_disk),
        "on_disk_not_in_csv_examples": extra_on_disk[:20],
    }


def video_quality_audit(all_rows: list[dict], video_files: list[Path]) -> dict:
    by_gloss: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_gloss[row["gloss"]].append(row)

    # Stratified sample: up to 3 videos per split from diverse glosses
    rng = random.Random(7)
    glosses = sorted(by_gloss.keys())
    rng.shuffle(glosses)
    chosen_rows: list[dict] = []
    for gloss in glosses[:120]:
        split_groups = defaultdict(list)
        for row in by_gloss[gloss]:
            split_groups[row["split"]].append(row)
        for split_name in ("train", "val", "test"):
            if split_groups[split_name]:
                chosen_rows.append(rng.choice(split_groups[split_name]))
        if len(chosen_rows) >= 300:
            break

    probes = []
    for row in chosen_rows:
        path = VIDEOS_DIR / row["video_file"]
        probe = probe_video(path)
        probe.update(
            {
                "gloss": row["gloss"],
                "participant_id": row["participant_id"],
                "split": row["split"],
                "asl_lex_code": row["asl_lex_code"],
            }
        )
        probes.append(probe)

    decodable = [p for p in probes if p.get("decodable")]
    durations = [p["duration_sec"] for p in decodable if p.get("duration_sec") is not None]
    frame_counts = [p["frame_count"] for p in decodable if p.get("frame_count")]
    fps_values = [p["fps"] for p in decodable if p.get("fps")]
    resolutions = Counter(p.get("resolution") for p in decodable if p.get("resolution"))
    fps_dist = Counter(round(f, 1) for f in fps_values)

    short_clips = [p for p in decodable if (p.get("duration_sec") or 0) < 0.5]
    corrupt = [p for p in probes if not p.get("decodable")]

    # Heuristic quality notes from sampled metadata (not full vision model)
    quality_notes = {
        "sample_size": len(probes),
        "decodable_count": len(decodable),
        "undecodable_count": len(corrupt),
        "short_clips_under_0_5s": len(short_clips),
        "duration_stats_sec": stat_summary(durations),
        "frame_count_stats": stat_summary([float(x) for x in frame_counts]),
        "fps_distribution_sample": dict(sorted(fps_dist.items(), key=lambda kv: (-kv[1], kv[0]))),
        "resolution_distribution_sample": dict(sorted(resolutions.items(), key=lambda kv: (-kv[1], kv[0]))),
        "observations": [
            "Videos are self-recorded webcam/phone clips with varied backgrounds and lighting (crowdsourced).",
            "Most clips are short isolated-sign recordings; temporal modeling is required for movement-heavy signs.",
            "Some filenames include variant suffixes (e.g., 'SOCCER 2', 'TALL 2') mapped to distinct glosses.",
            "Idle frames before/after signing are likely present based on dataset paper preprocessing notes.",
        ],
        "example_probes": probes[:12],
        "short_clip_examples": short_clips[:8],
        "corrupt_examples": corrupt[:8],
    }
    return quality_notes


def full_video_stats_sample(video_files: list[Path], n: int = 1200) -> dict:
    rng = random.Random(99)
    sample = rng.sample(video_files, min(n, len(video_files)))
    durations = []
    frame_counts = []
    fps_values = []
    resolutions = Counter()
    failed = 0
    for path in sample:
        probe = probe_video(path)
        if not probe.get("decodable"):
            failed += 1
            continue
        if probe.get("duration_sec") is not None:
            durations.append(probe["duration_sec"])
        if probe.get("frame_count"):
            frame_counts.append(float(probe["frame_count"]))
        if probe.get("fps"):
            fps_values.append(probe["fps"])
        if probe.get("resolution"):
            resolutions[probe["resolution"]] += 1
    return {
        "sample_size": len(sample),
        "failed_decode": failed,
        "duration_stats_sec": stat_summary(durations),
        "frame_count_stats": stat_summary(frame_counts),
        "fps_stats": stat_summary(fps_values),
        "fps_distribution": dict(Counter(round(f, 1) for f in fps_values)),
        "resolution_distribution": dict(sorted(resolutions.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def split_summary(all_rows: list[dict], distribution: list[dict]) -> dict:
    split_counts = Counter(r["split"] for r in all_rows)
    classes_by_split = {
        split: len({r["gloss"] for r in all_rows if r["split"] == split})
        for split in ("train", "val", "test")
    }
    participants_by_split = {
        split: len({r["participant_id"] for r in all_rows if r["split"] == split})
        for split in ("train", "val", "test")
    }
    return {
        "official_split_files": {k: str(v) for k, v in SPLIT_FILES.items()},
        "train_videos": split_counts.get("train", 0),
        "val_videos": split_counts.get("val", 0),
        "test_videos": split_counts.get("test", 0),
        "total_videos": len(all_rows),
        "train_classes": classes_by_split["train"],
        "val_classes": classes_by_split["val"],
        "test_classes": classes_by_split["test"],
        "total_distinct_classes": len(distribution),
        "train_participants": participants_by_split["train"],
        "val_participants": participants_by_split["val"],
        "test_participants": participants_by_split["test"],
        "label_mapping": {
            "sign_label_field": "Gloss",
            "signer_id_field": "Participant ID",
            "video_id_source": "numeric prefix before first '-' in Video file",
            "asl_lex_field": "ASL-LEX Code",
            "example_rows": all_rows[:5],
        },
    }


def storage_analysis(inventory: dict) -> dict:
    import shutil

    total, used, free = shutil.disk_usage("D:\\")
    dataset_gb = inventory["total_disk_usage_gb"]
    return {
        "extracted_dataset_gb": dataset_gb,
        "d_drive_free_gb": round(free / (1024**3), 2),
        "d_drive_used_gb": round(used / (1024**3), 2),
        "d_drive_total_gb": round(total / (1024**3), 2),
        "estimated_additional_storage_gb": {
            "manifests_and_metadata": 0.05,
            "optional_cached_frames_16f_224_subset100": 8,
            "optional_cached_frames_16f_224_full2731": 120,
            "model_checkpoints_resnet18_gru": 0.15,
            "training_outputs_logs": 0.5,
        },
    }


def write_class_distribution_csv(distribution: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gloss",
                "video_count",
                "train_count",
                "val_count",
                "test_count",
                "participant_count",
                "asl_lex_codes",
            ],
        )
        writer.writeheader()
        for row in distribution:
            out = dict(row)
            out["asl_lex_codes"] = "|".join(row["asl_lex_codes"])
            writer.writerow(out)


def build_audit_report(payload: dict) -> str:
    inv = payload["inventory"]
    split = payload["split_summary"]
    cls = payload["class_summary"]
    signer = payload["signer_analysis"]
    quality = payload["video_quality"]
    dup = payload["duplicates"]
    subsets = payload["subset_candidates"]
    storage = payload["storage"]

    lines = [
        "# ASL Citizen Dataset Audit Report (Phase 7A)",
        "",
        f"Generated: {payload['generated_at']}",
        f"Dataset root: `{inv['dataset_root']}`",
        "",
        "## 1. Dataset Structure",
        "",
        "```",
        "D:\\ASL_Citizen\\",
        "└── ASL_Citizen\\",
        "    ├── splits\\",
        "    │   ├── train.csv",
        "    │   ├── val.csv",
        "    │   └── test.csv",
        "    ├── videos\\          (flat .mp4 files)",
        "    └── use.txt            (Microsoft Research license)",
        "```",
        "",
        f"- Total files: **{inv['total_files']}**",
        f"- Total directories: **{inv['total_directories']}**",
        f"- Video files: **{inv['total_video_files']}**",
        f"- CSV/metadata files: **{inv['total_csv_files']}** CSV, **{inv['total_json_files']}** JSON",
        f"- Disk usage: **{inv['total_disk_usage_gb']} GB**",
        f"- Video formats: **{inv['file_extension_counts']}**",
        "",
        "## 2. Label / Metadata",
        "",
        "CSV columns: `Participant ID`, `Video file`, `Gloss`, `ASL-LEX Code`.",
        "",
        "A video maps to a sign via the **Gloss** column. Signer is **Participant ID** (P1..P52).",
        "Video ID is the numeric prefix before the first `-` in the filename.",
        "",
        "### Example rows",
        "",
    ]
    for row in split["label_mapping"]["example_rows"]:
        lines.append(
            f"- `{row['video_file']}` → gloss **{row['gloss']}**, signer **{row['participant_id']}**, "
            f"ASL-LEX **{row['asl_lex_code']}**, split **{row['split']}**"
        )

    lines.extend(
        [
            "",
            "## 3. Class Distribution",
            "",
            f"- Distinct signs/classes: **{cls['total_classes']}**",
            f"- Total videos (metadata): **{cls['total_videos']}**",
            f"- Videos/class — min: **{cls['min_videos_per_class']}**, max: **{cls['max_videos_per_class']}**, "
            f"mean: **{cls['mean_videos_per_class']}**, median: **{cls['median_videos_per_class']}**",
            "",
            "### Top 30 classes by video count",
            "",
        ]
    )
    for i, entry in enumerate(cls["top_30"], 1):
        lines.append(f"{i}. **{entry['gloss']}** — {entry['video_count']} videos")

    lines.extend(
        [
            "",
            f"- Classes with ≤5 videos: **{len(cls['classes_with_leq_5_videos'])}**",
            f"- Classes with ≥20 videos: **{len(cls['classes_with_ge_20_videos'])}**",
            "",
            "## 4. Official Split",
            "",
            f"| Split | Videos | Classes | Participants |",
            f"|-------|--------|---------|--------------|",
            f"| train | {split['train_videos']} | {split['train_classes']} | {split['train_participants']} |",
            f"| val   | {split['val_videos']} | {split['val_classes']} | {split['val_participants']} |",
            f"| test  | {split['test_videos']} | {split['test_classes']} | {split['test_participants']} |",
            "",
            "## 5. Signer Split Analysis",
            "",
            f"- Signer-independent: **{signer['signer_independent_split']}**",
            f"- Train∩Val overlap: **{len(signer['train_val_overlap'])}** participants",
            f"- Train∩Test overlap: **{len(signer['train_test_overlap'])}** participants",
            f"- Val∩Test overlap: **{len(signer['val_test_overlap'])}** participants",
            f"- Note: {signer['note']}",
            "",
            "## 6. Video Quality (sampled)",
            "",
            f"- Sample size: {quality['sample_size']}",
            f"- Decodable: {quality['decodable_count']} / Undecodable: {quality['undecodable_count']}",
            f"- Duration stats (sec): {quality['duration_stats_sec']}",
            f"- Frame count stats: {quality['frame_count_stats']}",
            f"- FPS distribution (sample): {quality['fps_distribution_sample']}",
            f"- Resolution distribution (sample): {quality['resolution_distribution_sample']}",
            "",
            "## 7. Duplicates / Leakage",
            "",
            f"- Duplicate filenames in metadata: {dup['metadata_duplicate_video_file_count']}",
            f"- Duplicate video IDs in metadata: {dup['metadata_duplicate_video_id_count']}",
            f"- Same filename across splits: {dup['cross_split_duplicate_filename_count']}",
            f"- Missing CSV-referenced files on disk: {dup['csv_referenced_missing_on_disk_count']}",
            f"- Partial-hash identical groups (500-file sample): {dup['partial_hash_identical_group_count']}",
            "",
            "## 8. Static vs Dynamic Signs",
            "",
            "ASL Citizen is an isolated-sign video dataset. Many signs require movement/path "
            "(e.g., signs with motion paths, two-part compounds). Single-frame classification is insufficient "
            "for a large portion of the vocabulary; temporal models (sampled frames + RNN/Transformer) are appropriate.",
            "",
            "## 9. Image vs Video Representation",
            "",
            "Recommended: **video → uniformly sampled temporal frames → spatial encoder → temporal model → class**.",
            "Matches existing ResNet18+GRU pipeline and dataset characteristics (short clips, movement, idle padding).",
            "",
            "## 10. Candidate Subsets",
            "",
        ]
    )
    for key in ("50", "100", "200", "300"):
        s = subsets[key]
        lines.append(
            f"- **{key} classes**: {s['total_videos']} videos, avg {s['avg_videos_per_class']}/class, "
            f"min {s['min_videos_per_class']}, train/val/test {s['train_videos']}/{s['val_videos']}/{s['test_videos']}"
        )

    lines.extend(
        [
            "",
            "**Recommendation:** Start with **100 classes** for Phase 7B — strong per-class support, "
            "manageable training time on RTX 4050 6GB, and useful vocabulary size for a teaching platform.",
            "",
            "## 11. Storage",
            "",
            f"- Dataset on disk: {storage['extracted_dataset_gb']} GB",
            f"- D: free space: {storage['d_drive_free_gb']} GB",
            f"- Estimated extras: {storage['estimated_additional_storage_gb']}",
            "",
            "## 12. Safety",
            "",
            "- No training performed.",
            "- No protected project assets modified.",
            "- No dataset files moved/deleted.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    if not DATASET_ROOT.exists():
        print(f"Dataset not found: {DATASET_ROOT}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for split_name in ("train", "val", "test"):
        all_rows.extend(load_split_rows(split_name))

    inventory = inventory_dataset()
    video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
    distribution, class_summary = build_class_distribution(all_rows)
    split = split_summary(all_rows, distribution)
    signer = signer_split_analysis(all_rows)
    duplicates = duplicate_analysis(all_rows, video_files)
    video_stats = full_video_stats_sample(video_files, n=1200)
    quality = video_quality_audit(all_rows, video_files)

    subset_candidates = {
        str(k): subset_candidate(distribution, all_rows, k) for k in (50, 100, 200, 300)
    }
    storage = storage_analysis(inventory)

    dataset_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "7A_audit_only",
        "inventory": inventory,
        "split_summary": split,
        "class_summary": class_summary,
        "video_probe_sample": video_stats,
        "subset_candidates": subset_candidates,
        "storage": storage,
        "published_reference": {
            "paper": "ASL Citizen (NeurIPS 2023)",
            "reported_videos": 83399,
            "reported_classes": 2731,
            "reported_signers": 52,
            "reported_mean_videos_per_class": 30.5,
            "reported_top1_accuracy_i3d": 0.631,
            "reported_recall_at_10": 0.9086,
        },
    }

    write_class_distribution_csv(distribution, OUTPUT_DIR / "class_distribution.csv")
    (OUTPUT_DIR / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "split_summary.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "signer_split_analysis.json").write_text(json.dumps(signer, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "video_quality_summary.json").write_text(
        json.dumps({"video_stats_sample": video_stats, "quality_audit": quality, "duplicates": duplicates}, indent=2),
        encoding="utf-8",
    )

    report_payload = {
        "generated_at": dataset_summary["generated_at"],
        "inventory": inventory,
        "split_summary": split,
        "class_summary": class_summary,
        "signer_analysis": signer,
        "video_quality": quality,
        "duplicates": duplicates,
        "subset_candidates": subset_candidates,
        "storage": storage,
    }
    (OUTPUT_DIR / "audit_report.md").write_text(build_audit_report(report_payload), encoding="utf-8")

    print(json.dumps(
        {
            "output_dir": str(OUTPUT_DIR),
            "total_videos_metadata": len(all_rows),
            "total_classes": class_summary["total_classes"],
            "signer_independent": signer["signer_independent_split"],
            "video_files_on_disk": len(video_files),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
