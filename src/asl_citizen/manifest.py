"""Deterministic 100-class manifest generation for ASL Citizen."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.asl_citizen.config import (
    CLASS_TO_IDX_PATH,
    CLASSES_CSV_PATH,
    DATA_DIR,
    IDX_TO_CLASS_PATH,
    NUM_CLASSES,
    REPORTS_DIR,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    default_config,
)
from src.asl_citizen.utils import MANIFEST_COLUMNS, load_all_source_rows, save_json_mapping


@dataclass(frozen=True)
class ClassSelection:
    rank: int
    gloss: str
    total: int
    train: int
    val: int
    test: int


def compute_gloss_stats(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, int]], dict[str, set[str]]]:
    stats: dict[str, Counter] = defaultdict(Counter)
    splits_present: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        gloss = row["gloss"]
        split = row["split"]
        stats[gloss][split] += 1
        stats[gloss]["total"] += 1
        splits_present[gloss].add(split)
    return {gloss: dict(counter) for gloss, counter in stats.items()}, splits_present


def select_top_classes(
    rows: list[dict[str, str]],
    num_classes: int = NUM_CLASSES,
) -> list[ClassSelection]:
    stats, splits_present = compute_gloss_stats(rows)
    required_splits = {"train", "val", "test"}
    eligible = [
        gloss
        for gloss, present in splits_present.items()
        if required_splits.issubset(present)
    ]
    if len(eligible) < num_classes:
        raise ValueError(
            f"Only {len(eligible)} glosses present in all splits; need {num_classes}"
        )

    ranked = sorted(
        eligible,
        key=lambda gloss: (-stats[gloss]["total"], gloss),
    )[:num_classes]

    selections: list[ClassSelection] = []
    for rank, gloss in enumerate(ranked, start=1):
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
    return selections


def build_class_mappings(selections: list[ClassSelection]) -> tuple[dict[str, int], dict[str, str]]:
    """Deterministic gloss→idx mapping: alphabetical order among selected glosses."""
    glosses = sorted(selection.gloss for selection in selections)
    class_to_idx = {gloss: index for index, gloss in enumerate(glosses)}
    idx_to_class = {str(index): gloss for gloss, index in class_to_idx.items()}
    return class_to_idx, idx_to_class


def filter_rows_for_selection(
    rows: list[dict[str, str]],
    class_to_idx: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
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


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_classes_csv(path: Path, selections: list[ClassSelection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "gloss", "total", "train", "val", "test"],
        )
        writer.writeheader()
        for selection in selections:
            writer.writerow(
                {
                    "rank": selection.rank,
                    "gloss": selection.gloss,
                    "total": selection.total,
                    "train": selection.train,
                    "val": selection.val,
                    "test": selection.test,
                }
            )


def verify_signer_leakage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, set[str]] = defaultdict(set)
    filenames_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_split[row["split"]].add(row["participant_id"])
        filenames_by_split[row["split"]].add(row["video_file"])

    train_p = by_split["train"]
    val_p = by_split["val"]
    test_p = by_split["test"]
    train_val_overlap = sorted(train_p & val_p)
    train_test_overlap = sorted(train_p & test_p)
    val_test_overlap = sorted(val_p & test_p)

    cross_split_files = sorted(
        (filenames_by_split["train"] & filenames_by_split["val"])
        | (filenames_by_split["train"] & filenames_by_split["test"])
        | (filenames_by_split["val"] & filenames_by_split["test"])
    )

    leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap or cross_split_files)
    if leakage:
        raise RuntimeError(
            "Signer or filename leakage detected in 100-class manifest: "
            f"train∩val={train_val_overlap}, train∩test={train_test_overlap}, "
            f"val∩test={val_test_overlap}, cross_split_files={cross_split_files[:5]}"
        )

    return {
        "signer_independent": True,
        "train_participants": sorted(train_p),
        "val_participants": sorted(val_p),
        "test_participants": sorted(test_p),
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "cross_split_duplicate_filenames": cross_split_files,
    }


def class_balance_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_gloss: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_gloss[row["gloss"]][row["split"]] += 1
        by_gloss[row["gloss"]]["total"] += 1

    table = []
    for gloss in sorted(by_gloss.keys()):
        counts = by_gloss[gloss]
        table.append(
            {
                "gloss": gloss,
                "train": counts.get("train", 0),
                "val": counts.get("val", 0),
                "test": counts.get("test", 0),
                "total": counts.get("total", 0),
            }
        )

    totals = [entry["total"] for entry in table]
    import statistics

    missing_split = [
        entry["gloss"]
        for entry in table
        if entry["train"] == 0 or entry["val"] == 0 or entry["test"] == 0
    ]
    if missing_split:
        raise RuntimeError(f"Selected classes missing from a split: {missing_split[:10]}")

    return {
        "num_classes": len(table),
        "per_class": table,
        "min_total": min(totals),
        "max_total": max(totals),
        "mean_total": round(statistics.mean(totals), 4),
        "median_total": round(statistics.median(totals), 4),
        "classes_missing_any_split": missing_split,
    }


def build_manifests(num_classes: int = NUM_CLASSES, cfg=None) -> dict[str, Any]:
    cfg = cfg or default_config()
    all_rows = load_all_source_rows()
    selections = select_top_classes(all_rows, num_classes=num_classes)
    class_to_idx, idx_to_class = build_class_mappings(selections)
    by_split = filter_rows_for_selection(all_rows, class_to_idx)

    write_classes_csv(CLASSES_CSV_PATH, selections)
    write_manifest_csv(TRAIN_MANIFEST_PATH, by_split["train"])
    write_manifest_csv(VAL_MANIFEST_PATH, by_split["val"])
    write_manifest_csv(TEST_MANIFEST_PATH, by_split["test"])
    save_json_mapping(CLASS_TO_IDX_PATH, class_to_idx)
    save_json_mapping(IDX_TO_CLASS_PATH, idx_to_class)

    subset_rows = by_split["train"] + by_split["val"] + by_split["test"]
    leakage = verify_signer_leakage(subset_rows)
    balance = class_balance_report(subset_rows)

    report = {
        "num_classes": num_classes,
        "selection_rule": "top N glosses by total video count; ties broken alphabetically",
        "mapping_rule": "class indices assigned by alphabetical gloss order among selected classes",
        "selections": [selection.__dict__ for selection in selections],
        "split_counts": {
            "train": len(by_split["train"]),
            "val": len(by_split["val"]),
            "test": len(by_split["test"]),
            "total": len(subset_rows),
        },
        "leakage_check": leakage,
        "class_balance": balance,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "manifest_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
