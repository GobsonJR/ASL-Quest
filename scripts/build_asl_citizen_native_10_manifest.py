"""Build the ASL Citizen "native_10" manifest: a small, curated, everyday-vocabulary
isolated-sign subset used to sanity-check the native-sign pipeline before any larger
training run.

This is a NEW, isolated experiment. It never touches data/asl_citizen_100 or the
100-class outputs/config: it builds its own ASLCitizenConfig pointed at
data/asl_citizen_native_10 and outputs/asl_citizen_native_10, and reuses the existing
generic helpers in src/asl_citizen/manifest.py and src/asl_citizen/utils.py (which all
take explicit paths/rows and are experiment-agnostic) without modifying those files.

Selection: 10 glosses drawn from the curated, human-reviewed
ASL_CITIZEN_20_GLOSSES list in src/asl_citizen/vocab.py (Stage 1 "everyday isolated
signs for a student demonstration"). That list already enforces: exact labels from the
ASL Citizen source CSVs (never invented), presence in all three official splits, and
one lexical variant per concept (e.g. EAT1 not EAT2). We pick 10 of those 20 covering
greetings, courtesy, basic responses, one person-noun, everyday nouns, and actions.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.config import ASLCitizenConfig, default_config
from src.asl_citizen.manifest import (
    ClassSelection,
    class_balance_report,
    compute_gloss_stats,
    verify_signer_leakage,
    write_classes_csv,
    write_manifest_csv,
)
from src.asl_citizen.utils import load_all_source_rows, save_json_mapping

EXPERIMENT_NAME = "asl_citizen_native_10"
DATA_DIR = ROOT / "data" / EXPERIMENT_NAME
OUTPUTS_DIR = ROOT / "outputs" / EXPERIMENT_NAME
REPORTS_DIR = OUTPUTS_DIR / "reports"

# 10 of the 20 curated glosses from src/asl_citizen/vocab.py (ASL_CITIZEN_20_GLOSSES).
# Every gloss below is an exact, unmodified label from the ASL Citizen source CSVs.
NATIVE_10_GLOSSES: tuple[str, ...] = (
    "HELLO",
    "THANKYOU",
    "PLEASE",
    "YES",
    "NO",
    "MOTHER",
    "WATER",
    "EAT1",
    "HELP",
    "BOOK",
)

SUITABILITY_REASONS: dict[str, str] = {
    "HELLO": "Core greeting; natural first sign for a student demo.",
    "THANKYOU": "High-frequency courtesy sign, visually distinct from HELLO.",
    "PLEASE": "High-frequency courtesy sign; pairs with THANKYOU.",
    "YES": "Basic affirmative response; short, high-motion sign.",
    "NO": "Basic negative response; pairs with YES for contrast.",
    "MOTHER": "Common people/family noun; single lexical variant in the dataset.",
    "WATER": "Everyday concrete noun with a distinctive handshape.",
    "EAT1": "Everyday action verb; highest video count of the 10 (39), aids memorization; "
    "EAT1 kept, EAT2 excluded as a duplicate lexical variant.",
    "HELP": "Common actionable request verb, distinct two-handed motion.",
    "BOOK": "Everyday object noun with a distinctive static handshape (contrast to the "
    "motion-heavy signs above).",
}


def native_10_config() -> ASLCitizenConfig:
    """Isolated config for this experiment. Never overlaps asl_citizen_100 paths."""
    base = default_config()
    checkpoints = OUTPUTS_DIR / "checkpoints"
    metrics = OUTPUTS_DIR / "metrics"
    return replace(
        base,
        experiment_name=EXPERIMENT_NAME,
        data_dir=DATA_DIR,
        train_manifest=DATA_DIR / "train.csv",
        val_manifest=DATA_DIR / "val.csv",
        test_manifest=DATA_DIR / "test.csv",
        class_to_idx_path=DATA_DIR / "class_to_idx.json",
        idx_to_class_path=DATA_DIR / "idx_to_class.json",
        classes_csv_path=DATA_DIR / "classes.csv",
        outputs_dir=OUTPUTS_DIR,
        checkpoints_dir=OUTPUTS_DIR / "checkpoints",
        reports_dir=REPORTS_DIR,
        metrics_dir=metrics,
        logs_dir=OUTPUTS_DIR / "logs",
        training_history_path=metrics / "training_history.json",
        best_checkpoint_path=OUTPUTS_DIR / "checkpoints" / "asl_citizen_native_10_resnet18_gru_best.pth",
        last_checkpoint_path=OUTPUTS_DIR / "checkpoints" / "asl_citizen_native_10_resnet18_gru_last.pth",
        num_classes=len(NATIVE_10_GLOSSES),
    )


def build_selections(rows: list[dict[str, str]]) -> list[ClassSelection]:
    stats, splits_present = compute_gloss_stats(rows)
    required = {"train", "val", "test"}
    selections: list[ClassSelection] = []
    missing: list[str] = []
    for rank, gloss in enumerate(NATIVE_10_GLOSSES, start=1):
        if gloss not in stats or not required.issubset(splits_present.get(gloss, set())):
            missing.append(gloss)
            continue
        counts = stats[gloss]
        selections.append(
            ClassSelection(
                rank=rank,
                gloss=gloss,
                total=counts["total"],
                train=counts.get("train", 0),
                val=counts.get("val", 0),
                test=counts.get("test", 0),
            )
        )
    if missing:
        raise RuntimeError(f"Glosses not present in all splits (dataset may have changed): {missing}")
    return selections


def build_class_mappings(selections: list[ClassSelection]) -> tuple[dict[str, int], dict[str, str]]:
    glosses = sorted(selection.gloss for selection in selections)
    class_to_idx = {gloss: index for index, gloss in enumerate(glosses)}
    idx_to_class = {str(index): gloss for gloss, index in class_to_idx.items()}
    return class_to_idx, idx_to_class


def filter_rows(rows: list[dict[str, str]], class_to_idx: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        gloss = row["gloss"]
        if gloss not in class_to_idx:
            continue
        by_split[row["split"]].append(
            {
                "participant_id": row["participant_id"],
                "video_file": row["video_file"],
                "gloss": gloss,
                "asl_lex_code": row["asl_lex_code"],
                "class_idx": class_to_idx[gloss],
                "split": row["split"],
            }
        )
    return by_split


def verify_video_paths(rows: list[dict[str, Any]], video_root: Path) -> dict[str, Any]:
    missing: list[str] = []
    for row in rows:
        if not (video_root / row["video_file"]).is_file():
            missing.append(row["video_file"])
    return {"checked": len(rows), "missing_count": len(missing), "missing": missing[:20]}


def verify_no_duplicate_video_ids(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def video_id(filename: str) -> str:
        return filename.split("-", 1)[0]

    ids = [video_id(row["video_file"]) for row in rows]
    filenames = [row["video_file"] for row in rows]
    id_dupes = [vid for vid, count in Counter(ids).items() if count > 1]
    filename_dupes = [f for f, count in Counter(filenames).items() if count > 1]
    return {
        "total_videos": len(rows),
        "unique_video_ids": len(set(ids)),
        "duplicate_video_ids": id_dupes,
        "duplicate_filenames": filename_dupes,
    }


def video_duration_stats(rows: list[dict[str, Any]], video_root: Path) -> dict[str, Any]:
    durations: list[float] = []
    frame_counts: list[float] = []
    unreadable: list[str] = []
    for row in rows:
        path = video_root / row["video_file"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            unreadable.append(row["video_file"])
            capture.release()
            continue
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if frames > 0 and fps > 0:
            durations.append(frames / fps)
            frame_counts.append(frames)
        else:
            unreadable.append(row["video_file"])
    if not durations:
        return {"sample_size": 0, "unreadable": unreadable}
    return {
        "sample_size": len(durations),
        "unreadable_count": len(unreadable),
        "duration_seconds": {
            "min": round(min(durations), 3),
            "max": round(max(durations), 3),
            "mean": round(statistics.mean(durations), 3),
            "median": round(statistics.median(durations), 3),
        },
        "frame_count": {
            "min": min(frame_counts),
            "max": max(frame_counts),
            "mean": round(statistics.mean(frame_counts), 2),
            "median": round(statistics.median(frame_counts), 2),
        },
    }


def main() -> dict[str, Any]:
    cfg = native_10_config()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = load_all_source_rows()
    selections = build_selections(all_rows)
    class_to_idx, idx_to_class = build_class_mappings(selections)
    by_split = filter_rows(all_rows, class_to_idx)

    write_classes_csv(cfg.classes_csv_path, selections)
    write_manifest_csv(cfg.train_manifest, by_split["train"])
    write_manifest_csv(cfg.val_manifest, by_split["val"])
    write_manifest_csv(cfg.test_manifest, by_split["test"])
    save_json_mapping(cfg.class_to_idx_path, class_to_idx)
    save_json_mapping(cfg.idx_to_class_path, idx_to_class)

    subset_rows = by_split["train"] + by_split["val"] + by_split["test"]

    leakage = verify_signer_leakage(subset_rows)
    balance = class_balance_report(subset_rows)
    path_check = verify_video_paths(subset_rows, cfg.video_root)
    dup_check = verify_no_duplicate_video_ids(subset_rows)
    duration_stats = video_duration_stats(subset_rows, cfg.video_root)

    if path_check["missing_count"] > 0:
        raise RuntimeError(f"Missing video files on disk: {path_check['missing']}")
    if dup_check["duplicate_video_ids"] or dup_check["duplicate_filenames"]:
        raise RuntimeError("Duplicate video IDs/filenames detected in native_10 subset")

    report: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "num_classes": len(NATIVE_10_GLOSSES),
        "selection_rule": (
            "10 glosses hand-picked from the curated 20-word ASL_CITIZEN_20_GLOSSES list "
            "(src/asl_citizen/vocab.py), which itself requires exact source-CSV labels, "
            "presence in all three official splits, and one lexical variant per concept."
        ),
        "mapping_rule": "class indices assigned by alphabetical gloss order among selected classes",
        "selections": [
            {**selection.__dict__, "suitability": SUITABILITY_REASONS[selection.gloss]}
            for selection in selections
        ],
        "split_counts": {
            "train": len(by_split["train"]),
            "val": len(by_split["val"]),
            "test": len(by_split["test"]),
            "total": len(subset_rows),
        },
        "leakage_check": leakage,
        "class_balance": balance,
        "video_path_check": path_check,
        "duplicate_check": dup_check,
        "video_duration_stats": duration_stats,
        "data_dir": str(DATA_DIR),
        "outputs_dir": str(OUTPUTS_DIR),
    }

    report_path = REPORTS_DIR / "manifest_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
