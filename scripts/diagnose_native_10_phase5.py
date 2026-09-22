"""Phase 5 native-10 signer-independent generalization diagnostic.

Read-only on checkpoints/manifests/dataset videos. Writes only under
outputs/asl_citizen_native_10/diagnostics/phase5_generalization/.

Does NOT train any model. The only forward/backward passes here are single-batch,
no-optimizer-step diagnostic probes (diagnostic 9), mirroring the existing
scripts/diagnose_layer4_100class_failure.py convention in this repo.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_asl_citizen_native_10_manifest import native_10_config
from src.asl_citizen.checkpoint import load_checkpoint
from src.asl_citizen.dataset import build_dataloader, decode_sampled_frames, get_frame_count
from src.asl_citizen.model import build_model, gradient_flow_report
from src.asl_citizen.preprocessing import jittered_temporal_indices, uniform_temporal_indices
from src.asl_citizen.utils import load_class_mappings, load_manifest_rows, resolve_video_path, seed_pipeline

OUT = ROOT / "outputs" / "asl_citizen_native_10" / "diagnostics" / "phase5_generalization"
OUT_VISUAL = OUT / "visual_audit"
OUT_TEMPORAL = OUT / "temporal_sampling"
OUT_SPATIAL = OUT / "spatial_preprocessing"
OUT_FEATURES = OUT / "feature_analysis"

FROZEN_BEST = ROOT / "outputs" / "asl_citizen_native_10" / "checkpoints" / "asl_citizen_native_10_resnet18_gru_best.pth"
LAYER4_BEST = (
    ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "layer4" / "checkpoints"
    / "asl_citizen_native_10_layer4_resnet18_gru_best.pth"
)
FROZEN_EVAL_REPORT = ROOT / "outputs" / "asl_citizen_native_10" / "reports" / "test_evaluation_report.json"
LAYER4_EVAL_REPORT = (
    ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "layer4" / "reports" / "test_evaluation_report.json"
)
LAYER4_HISTORY = ROOT / "outputs" / "asl_citizen_native_10" / "experiments" / "layer4" / "metrics" / "training_history.json"
FROZEN_HISTORY = ROOT / "outputs" / "asl_citizen_native_10" / "reports" / "train_run_summary.json"
TINY_FROZEN = ROOT / "outputs" / "asl_citizen_native_10" / "diagnostics" / "tiny_overfit_frozen.json"
TINY_LAYER4 = ROOT / "outputs" / "asl_citizen_native_10" / "diagnostics" / "tiny_overfit_layer4.json"
NATIVE10_CLASS_TO_IDX = ROOT / "data" / "asl_citizen_native_10" / "class_to_idx.json"
NATIVE10_IDX_TO_CLASS = ROOT / "data" / "asl_citizen_native_10" / "idx_to_class.json"


def dump(name: str, payload: Any, subdir: Path | None = None) -> Path:
    target_dir = subdir or OUT
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def stat_block(values: list[float]) -> dict[str, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }


def video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    frame_count = get_frame_count(cap)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps > 0 else None
    return {
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "duration_sec": round(duration, 4) if duration else None,
        "aspect_ratio": round(width / height, 4) if height else None,
    }


def load_all_rows(cfg) -> dict[str, list[dict[str, str]]]:
    return {
        "train": load_manifest_rows(cfg.train_manifest),
        "val": load_manifest_rows(cfg.val_manifest),
        "test": load_manifest_rows(cfg.test_manifest),
    }


# ---------------------------------------------------------------------------
# Diagnostic 1: data distribution
# ---------------------------------------------------------------------------


def diagnostic_1(cfg, rows_by_split: dict[str, list[dict]]) -> dict:
    print("Diagnostic 1: data distribution")
    per_video: dict[str, dict] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            path = resolve_video_path(cfg.video_root, row["video_file"])
            meta = video_metadata(path)
            meta.update({"split": split, "gloss": row["gloss"], "participant_id": row["participant_id"]})
            per_video[f"{split}/{row['video_file']}"] = meta

    glosses = sorted({row["gloss"] for rows in rows_by_split.values() for row in rows})
    per_class_split: dict[str, dict[str, Any]] = {}
    for gloss in glosses:
        per_class_split[gloss] = {}
        for split, rows in rows_by_split.items():
            class_rows = [r for r in rows if r["gloss"] == gloss]
            metas = [per_video[f"{split}/{r['video_file']}"] for r in class_rows]
            signer_counts = Counter(r["participant_id"] for r in class_rows)
            per_class_split[gloss][split] = {
                "count": len(class_rows),
                "duration_sec": stat_block([m["duration_sec"] for m in metas]),
                "frame_count": stat_block([m["frame_count"] for m in metas]),
                "fps": stat_block([m["fps"] for m in metas]),
                "aspect_ratio": stat_block([m["aspect_ratio"] for m in metas]),
                "unique_signers": len(signer_counts),
                "clips_per_signer": dict(sorted(signer_counts.items())),
            }

    resolution_counts = Counter((m["width"], m["height"]) for m in per_video.values())
    fps_counts = Counter(m["fps"] for m in per_video.values())

    report = {
        "num_classes": len(glosses),
        "glosses": glosses,
        "split_totals": {split: len(rows) for split, rows in rows_by_split.items()},
        "per_class_split": per_class_split,
        "resolution_distribution": {f"{w}x{h}": c for (w, h), c in resolution_counts.items()},
        "fps_distribution": {str(k): v for k, v in fps_counts.items()},
        "overall_duration_sec": stat_block([m["duration_sec"] for m in per_video.values()]),
        "overall_frame_count": stat_block([m["frame_count"] for m in per_video.values()]),
    }
    dump("data_distribution_report.json", report)
    return report


# ---------------------------------------------------------------------------
# Diagnostic 2: visual frame audit
# ---------------------------------------------------------------------------


def _thumbnail(frame_bgr: np.ndarray, size: int = 200) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (32, 32, 32))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def diagnostic_2(cfg, rows_by_split: dict[str, list[dict]], glosses: list[str]) -> dict:
    print("Diagnostic 2: visual frame audit")
    rng = random.Random(7)
    manifest: dict[str, Any] = {}
    tile = 200
    label_h = 18
    for gloss in glosses:
        chosen: dict[str, dict] = {}
        for split, rows in rows_by_split.items():
            candidates = [r for r in rows if r["gloss"] == gloss]
            if candidates:
                chosen[split] = rng.choice(candidates)
        splits_present = [s for s in ("train", "val", "test") if s in chosen]
        grid = Image.new(
            "RGB",
            (tile * 3, (tile + label_h) * len(splits_present)),
            (16, 16, 16),
        )
        draw = ImageDraw.Draw(grid)
        video_records = []
        for row_i, split in enumerate(splits_present):
            row = chosen[split]
            path = resolve_video_path(cfg.video_root, row["video_file"])
            cap = cv2.VideoCapture(str(path))
            frame_count = get_frame_count(cap)
            cap.release()
            indices = [0, frame_count // 2, max(frame_count - 1, 0)]
            frames = decode_sampled_frames(path, indices)
            for col_i, (idx, frame) in enumerate(zip(indices, frames)):
                thumb = _thumbnail(frame, tile)
                y = row_i * (tile + label_h)
                grid.paste(thumb, (col_i * tile, y + label_h))
                label = f"{split}:{['begin','mid','end'][col_i]} f{idx}/{frame_count}"
                draw.text((col_i * tile + 4, y + 2), label, fill=(255, 255, 0))
            video_records.append(
                {
                    "split": split,
                    "video_file": row["video_file"],
                    "participant_id": row["participant_id"],
                    "frame_count": frame_count,
                    "sampled_indices": indices,
                }
            )
        sheet_path = OUT_VISUAL / f"{gloss}_contact_sheet.png"
        OUT_VISUAL.mkdir(parents=True, exist_ok=True)
        grid.save(sheet_path)
        manifest[gloss] = {"contact_sheet": str(sheet_path.relative_to(ROOT)), "videos": video_records}
        print(f"  wrote {sheet_path.relative_to(ROOT)}")
    dump("visual_audit_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Diagnostic 3: temporal sampling strategies (statistics only, no training)
# ---------------------------------------------------------------------------


def center_focused_indices(frame_count: int, num_frames: int) -> list[int]:
    if frame_count <= num_frames:
        return uniform_temporal_indices(frame_count, num_frames)
    lo = int(frame_count * 0.25)
    hi = int(frame_count * 0.75)
    if hi <= lo:
        lo, hi = 0, frame_count - 1
    span = hi - lo
    positions = np.linspace(lo, hi, num=num_frames) if span > 0 else [lo] * num_frames
    return [min(max(int(round(p)), 0), frame_count - 1) for p in positions]


def start_mid_end_indices(frame_count: int, num_frames: int) -> list[int]:
    segments = 3
    per_segment = [num_frames // segments] * segments
    for i in range(num_frames - sum(per_segment)):
        per_segment[i] += 1
    bounds = np.linspace(0, frame_count, num=segments + 1)
    indices: list[int] = []
    for seg_i, n in enumerate(per_segment):
        lo, hi = bounds[seg_i], max(bounds[seg_i + 1] - 1, bounds[seg_i])
        if n <= 0:
            continue
        positions = np.linspace(lo, hi, num=n)
        indices.extend(min(max(int(round(p)), 0), frame_count - 1) for p in positions)
    return sorted(indices)[:num_frames]


def dense_indices(frame_count: int, num_frames: int) -> list[int]:
    dense_count = min(frame_count, num_frames * 2)
    return uniform_temporal_indices(frame_count, dense_count)


def _motion_energy(frames: list[np.ndarray]) -> dict:
    if len(frames) < 2:
        return {"sum": 0.0, "mean_per_pair": 0.0, "n_pairs": 0}
    diffs = []
    for a, b in zip(frames, frames[1:]):
        diffs.append(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()))
    return {"sum": round(sum(diffs), 4), "mean_per_pair": round(sum(diffs) / len(diffs), 4), "n_pairs": len(diffs)}


def diagnostic_3(cfg, rows_by_split: dict[str, list[dict]], glosses: list[str]) -> dict:
    print("Diagnostic 3: temporal sampling")
    rng = random.Random(11)
    per_video: list[dict] = []
    motion_subset: list[dict] = []
    motion_budget = 5
    for gloss in glosses:
        candidates = rows_by_split["train"]
        rows = [r for r in candidates if r["gloss"] == gloss]
        if not rows:
            continue
        sample_rows = rng.sample(rows, min(2, len(rows)))
        for row in sample_rows:
            path = resolve_video_path(cfg.video_root, row["video_file"])
            cap = cv2.VideoCapture(str(path))
            frame_count = get_frame_count(cap)
            cap.release()
            strategies = {
                "A_uniform_current": uniform_temporal_indices(frame_count, 16),
                "B_dense_2x": dense_indices(frame_count, 16),
                "C_center_focused": center_focused_indices(frame_count, 16),
                "D_start_mid_end": start_mid_end_indices(frame_count, 16),
            }
            entry = {"gloss": gloss, "video_file": row["video_file"], "frame_count": frame_count, "strategies": {}}
            for name, indices in strategies.items():
                span = (max(indices) - min(indices)) if len(indices) > 1 else 0
                coverage = round(span / max(frame_count - 1, 1), 4)
                spacings = np.diff(sorted(indices)) if len(indices) > 1 else np.array([0])
                entry["strategies"][name] = {
                    "num_sampled": len(indices),
                    "indices": indices,
                    "coverage_fraction_of_video": coverage,
                    "mean_spacing": round(float(np.mean(spacings)), 3),
                    "max_spacing": int(np.max(spacings)) if len(spacings) else 0,
                }
            per_video.append(entry)

            if len(motion_subset) < motion_budget:
                energies = {}
                for name, indices in strategies.items():
                    frames = decode_sampled_frames(path, indices)
                    energies[name] = _motion_energy(frames)
                motion_subset.append(
                    {"gloss": gloss, "video_file": row["video_file"], "frame_count": frame_count, "motion_energy": energies}
                )

    jitter_note = (
        "Production training uses jittered_temporal_indices with train_jitter_frames=2 layered on top of strategy "
        "A_uniform_current; jitter magnitude is compared against real per-class spacing in diagnostic 8."
    )
    report = {
        "strategies_defined": ["A_uniform_current", "B_dense_2x", "C_center_focused", "D_start_mid_end"],
        "per_video": per_video,
        "motion_energy_subset": motion_subset,
        "note": jitter_note,
    }
    dump("temporal_sampling_report.json", report)

    # Visualization: timeline plot for up to 4 example videos
    OUT_TEMPORAL.mkdir(parents=True, exist_ok=True)
    examples = per_video[:4]
    fig, axes = plt.subplots(len(examples), 1, figsize=(9, 2.0 * len(examples)), squeeze=False)
    colors = {"A_uniform_current": "tab:blue", "B_dense_2x": "tab:orange", "C_center_focused": "tab:green", "D_start_mid_end": "tab:red"}
    for ax_i, entry in enumerate(examples):
        ax = axes[ax_i][0]
        for row_i, (name, data) in enumerate(entry["strategies"].items()):
            ax.scatter(data["indices"], [row_i] * len(data["indices"]), color=colors[name], s=14, label=name)
        ax.set_yticks(range(len(entry["strategies"])))
        ax.set_yticklabels(list(entry["strategies"].keys()), fontsize=7)
        ax.set_xlim(0, entry["frame_count"])
        ax.set_title(f"{entry['gloss']} — {entry['video_file']} ({entry['frame_count']} frames)", fontsize=8)
    fig.tight_layout()
    plot_path = OUT_TEMPORAL / "sampling_strategy_timelines.png"
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {plot_path.relative_to(ROOT)}")
    return report


# ---------------------------------------------------------------------------
# Diagnostic 4: spatial preprocessing visualization
# ---------------------------------------------------------------------------


def diagnostic_4(cfg, rows_by_split: dict[str, list[dict]], glosses: list[str]) -> dict:
    print("Diagnostic 4: spatial preprocessing")
    rng = random.Random(21)
    OUT_SPATIAL.mkdir(parents=True, exist_ok=True)
    samples = []
    splits = ["train", "val", "test"]
    for i, gloss in enumerate(glosses[:6]):
        split = splits[i % len(splits)]
        rows = [r for r in rows_by_split[split] if r["gloss"] == gloss]
        if not rows:
            continue
        samples.append((split, rng.choice(rows)))

    records = []
    for split, row in samples:
        path = resolve_video_path(cfg.video_root, row["video_file"])
        cap = cv2.VideoCapture(str(path))
        frame_count = get_frame_count(cap)
        cap.release()
        mid = frame_count // 2
        frame_bgr = decode_sampled_frames(path, [mid])[0]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        original = Image.fromarray(rgb)

        # Train path: resize to 256, RandomResizedCrop(scale=0.85-1.0, ratio=0.9-1.1) -> 224
        resized_train = TF.resize(original, [256, 256])
        top, left, h, w = RandomResizedCrop.get_params(resized_train, scale=(0.85, 1.0), ratio=(0.9, 1.1))
        cropped_train = TF.resized_crop(resized_train, top, left, h, w, [224, 224])
        train_area_fraction = round((h * w) / (256 * 256), 4)

        # Eval path: resize to 240, center crop 224
        resized_eval = TF.resize(original, [240, 240])
        eval_crop_side = 224
        eval_top = (240 - eval_crop_side) // 2
        eval_left = (240 - eval_crop_side) // 2
        cropped_eval = TF.center_crop(resized_eval, [224, 224])
        eval_area_fraction = round((eval_crop_side * eval_crop_side) / (240 * 240), 4)

        fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
        axes[0].imshow(original)
        axes[0].set_title(f"original\n{original.size}", fontsize=8)

        axes[1].imshow(resized_train)
        rect = plt.Rectangle((left, top), w, h, edgecolor="lime", facecolor="none", linewidth=2)
        axes[1].add_patch(rect)
        axes[1].set_title(f"resized 256 + train crop box\narea={train_area_fraction}", fontsize=8)

        axes[2].imshow(cropped_train)
        axes[2].set_title("train crop -> 224", fontsize=8)

        axes[3].imshow(resized_eval)
        rect_e = plt.Rectangle((eval_left, eval_top), eval_crop_side, eval_crop_side, edgecolor="cyan", facecolor="none", linewidth=2)
        axes[3].add_patch(rect_e)
        axes[3].set_title(f"resized 240 + eval crop box\narea={eval_area_fraction}", fontsize=8)

        axes[4].imshow(cropped_eval)
        axes[4].set_title("eval center-crop -> 224", fontsize=8)

        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{row['gloss']} ({split}) — {row['video_file']}", fontsize=10)
        fig.tight_layout()
        out_path = OUT_SPATIAL / f"{row['gloss']}_{split}_preprocessing.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"  wrote {out_path.relative_to(ROOT)}")

        records.append(
            {
                "gloss": row["gloss"],
                "split": split,
                "video_file": row["video_file"],
                "original_size": list(original.size),
                "train_crop_box_ltwh": [left, top, w, h],
                "train_crop_area_fraction_of_resized": train_area_fraction,
                "eval_crop_area_fraction_of_resized": eval_area_fraction,
                "image_path": str(out_path.relative_to(ROOT)),
            }
        )

    report = {
        "train_resize_target": [256, 256],
        "train_random_resized_crop": {"output_size": 224, "scale": [0.85, 1.0], "ratio": [0.9, 1.1]},
        "eval_resize_target": [240, 240],
        "eval_center_crop_output_size": 224,
        "samples": records,
        "note": (
            "No hand/pose landmarks are available in this pipeline (unlike the production A-Z MediaPipe path), "
            "so hand visibility after cropping can only be assessed visually from the saved images, not measured "
            "automatically."
        ),
    }
    dump("spatial_preprocessing_report.json", report)
    return report


# ---------------------------------------------------------------------------
# Diagnostics 5 & 6: feature representation + signer analysis (shared inference)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference_with_features(checkpoint_path: Path, split: str, device: torch.device) -> dict:
    payload, cfg, class_to_idx, idx_to_class = load_checkpoint(checkpoint_path, map_location=device)
    cfg = replace(cfg, class_to_idx_path=NATIVE10_CLASS_TO_IDX, idx_to_class_path=NATIVE10_IDX_TO_CLASS)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    loader, dataset = build_dataloader(split, cfg, batch_size=8, shuffle=False)
    feats, labels, preds, participants, glosses, files = [], [], [], [], [], []
    for batch in loader:
        clips = batch["video"].to(device)
        encoded = model.encode_frames(clips)
        gru_out, _ = model.gru(encoded)
        pooled = gru_out[:, -1, :]
        logits = model.classifier(pooled)
        feats.append(pooled.cpu())
        labels.append(batch["label"])
        preds.extend(logits.argmax(1).cpu().tolist())
        participants.extend(batch["participant_id"])
        glosses.extend(batch["gloss"])
        files.extend(batch["video_file"])
    feats_np = torch.cat(feats).numpy()
    labels_np = torch.cat(labels).numpy()
    return {
        "features": feats_np,
        "labels": labels_np,
        "preds": np.array(preds),
        "participants": participants,
        "glosses": glosses,
        "files": files,
        "idx_to_class": idx_to_class,
        "class_to_idx": class_to_idx,
        "num_samples": len(dataset),
    }


def diagnostic_5_and_6(device: torch.device) -> tuple[dict, dict]:
    print("Diagnostic 5+6: feature representation & signer analysis")
    OUT_FEATURES.mkdir(parents=True, exist_ok=True)
    checkpoints = {"frozen": FROZEN_BEST, "layer4": LAYER4_BEST}
    feature_report: dict[str, Any] = {}
    signer_report: dict[str, Any] = {}

    for name, ckpt in checkpoints.items():
        result = run_inference_with_features(ckpt, "test", device)
        feats = result["features"]
        labels = result["labels"]
        preds = result["preds"]
        participants = result["participants"]
        idx_to_class = result["idx_to_class"]
        num_classes = len(idx_to_class)

        # centroids and intra/inter distances
        centroids = np.stack([feats[labels == c].mean(axis=0) if np.any(labels == c) else np.zeros(feats.shape[1]) for c in range(num_classes)])
        intra = {}
        for c in range(num_classes):
            members = feats[labels == c]
            if len(members) > 1:
                d = np.linalg.norm(members[:, None, :] - members[None, :, :], axis=-1)
                intra[idx_to_class[str(c)]] = round(float(d[np.triu_indices(len(members), k=1)].mean()), 4)
            else:
                intra[idx_to_class[str(c)]] = None
        inter_matrix = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
        inter_mean = float(inter_matrix[np.triu_indices(num_classes, k=1)].mean())

        # nearest-centroid classification (independent probe of feature separability)
        dist_to_centroid = np.linalg.norm(feats[:, None, :] - centroids[None, :, :], axis=-1)
        nearest_centroid_pred = dist_to_centroid.argmin(axis=1)
        nearest_centroid_acc = float((nearest_centroid_pred == labels).mean())
        classifier_acc = float((preds == labels).mean())

        # silhouette scores: does the representation separate better by class or by signer?
        signer_ids = sorted(set(participants))
        signer_to_idx = {s: i for i, s in enumerate(signer_ids)}
        signer_labels = np.array([signer_to_idx[p] for p in participants])
        sil_class = float(silhouette_score(feats, labels)) if num_classes > 1 else None
        sil_signer = float(silhouette_score(feats, signer_labels)) if len(signer_ids) > 1 else None

        # PCA projection for visualization only
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(feats)
        explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]

        for color_by, color_labels, cmap_name in (
            ("class", labels, "tab10"),
            ("signer", signer_labels, "tab20"),
        ):
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(proj[:, 0], proj[:, 1], c=color_labels, cmap=cmap_name, s=24)
            ax.set_title(f"{name} checkpoint — test features colored by {color_by}\nPCA explained var={explained}", fontsize=9)
            legend = ax.legend(*scatter.legend_elements(num=min(10, len(set(color_labels)))), fontsize=6, loc="best", ncol=2)
            ax.add_artist(legend)
            fig.tight_layout()
            plot_path = OUT_FEATURES / f"{name}_pca_by_{color_by}.png"
            fig.savefig(plot_path, dpi=110)
            plt.close(fig)
            print(f"  wrote {plot_path.relative_to(ROOT)}")

        feature_report[name] = {
            "num_test_samples": int(len(feats)),
            "feature_dim": int(feats.shape[1]),
            "intra_class_mean_distance": intra,
            "inter_class_centroid_mean_distance": round(inter_mean, 4),
            "nearest_centroid_accuracy": round(nearest_centroid_acc, 4),
            "classifier_top1_accuracy_cross_check": round(classifier_acc, 4),
            "silhouette_score_by_true_class": round(sil_class, 4) if sil_class is not None else None,
            "silhouette_score_by_signer": round(sil_signer, 4) if sil_signer is not None else None,
            "pca_explained_variance_ratio": explained,
            "num_unique_test_signers": len(signer_ids),
        }

        # signer-level accuracy + predicted class distribution
        per_signer: dict[str, Any] = {}
        for signer in signer_ids:
            mask = [p == signer for p in participants]
            idxs = [i for i, m in enumerate(mask) if m]
            s_labels = labels[idxs]
            s_preds = preds[idxs]
            pred_dist = Counter(idx_to_class[str(p)] for p in s_preds)
            per_signer[signer] = {
                "total": len(idxs),
                "correct": int((s_labels == s_preds).sum()),
                "accuracy": round(float((s_labels == s_preds).mean()), 4) if idxs else None,
                "predicted_class_distribution": dict(pred_dist),
            }
        signer_report[name] = per_signer

    dump("feature_analysis.json", feature_report)
    dump("signer_analysis_predictions.json", signer_report)
    return feature_report, signer_report


def diagnostic_6_train_val_signers(rows_by_split: dict[str, list[dict]]) -> dict:
    result = {}
    for split in ("train", "val", "test"):
        by_class_signers: dict[str, dict[str, int]] = defaultdict(dict)
        for row in rows_by_split[split]:
            by_class_signers[row["gloss"]][row["participant_id"]] = (
                by_class_signers[row["gloss"]].get(row["participant_id"], 0) + 1
            )
        result[split] = {gloss: signers for gloss, signers in sorted(by_class_signers.items())}
    return result


# ---------------------------------------------------------------------------
# Diagnostic 7: class confusion (from already-saved confusion matrices)
# ---------------------------------------------------------------------------


def diagnostic_7() -> dict:
    print("Diagnostic 7: class confusion")
    report = {}
    for name, path in (("frozen", FROZEN_EVAL_REPORT), ("layer4", LAYER4_EVAL_REPORT)):
        data = json.loads(path.read_text(encoding="utf-8"))
        labels = data["confusion_matrix"]["labels"]
        matrix = np.array(data["confusion_matrix"]["matrix"])
        n = len(labels)

        never_correct = [labels[i] for i in range(n) if matrix[i, i] == 0]
        never_predicted = [labels[i] for i in range(n) if matrix[:, i].sum() == 0]
        support = matrix.sum(axis=1)
        predicted_count = matrix.sum(axis=0)
        over_predicted = sorted(
            (
                {"gloss": labels[i], "support": int(support[i]), "predicted_count": int(predicted_count[i]), "excess": int(predicted_count[i] - support[i])}
                for i in range(n)
            ),
            key=lambda r: r["excess"],
            reverse=True,
        )

        off_diag = []
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i, j] > 0:
                    off_diag.append({"true": labels[i], "predicted": labels[j], "count": int(matrix[i, j])})
        off_diag.sort(key=lambda r: r["count"], reverse=True)

        asymmetries = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = int(matrix[i, j]), int(matrix[j, i])
                if a + b > 0:
                    asymmetries.append({"pair": [labels[i], labels[j]], f"{labels[i]}_predicted_as_{labels[j]}": a, f"{labels[j]}_predicted_as_{labels[i]}": b, "abs_diff": abs(a - b)})
        asymmetries.sort(key=lambda r: r["abs_diff"], reverse=True)

        report[name] = {
            "never_correctly_predicted_classes": never_correct,
            "never_predicted_at_all_classes": never_predicted,
            "most_over_predicted_classes": over_predicted[:5],
            "top_confused_pairs": off_diag[:10],
            "most_asymmetric_pairs": asymmetries[:5],
        }
    dump("confusion_analysis.json", report)
    return report


# ---------------------------------------------------------------------------
# Diagnostic 8: augmentation effect (config-based, evidence from real stats)
# ---------------------------------------------------------------------------


def diagnostic_8(data_dist: dict) -> dict:
    print("Diagnostic 8: augmentation effect")
    tiny_frozen = json.loads(TINY_FROZEN.read_text(encoding="utf-8")) if TINY_FROZEN.exists() else None
    tiny_layer4 = json.loads(TINY_LAYER4.read_text(encoding="utf-8")) if TINY_LAYER4.exists() else None

    train_crop_area_min = round((0.85), 4)  # scale lower bound = min area fraction of the 256x256 resize retained
    train_crop_area_max = 1.0

    # jitter magnitude vs typical native_10 sample spacing (frame_count / 16), from diagnostic 1's per-class stats
    spacings = []
    for gloss, splits in data_dist["per_class_split"].items():
        train_stats = splits.get("train", {})
        mean_frames = train_stats.get("frame_count", {}).get("mean")
        if mean_frames:
            spacings.append(mean_frames / 16)
    mean_spacing = round(statistics.mean(spacings), 3) if spacings else None
    jitter_frames = 2
    jitter_fraction_of_spacing = round(jitter_frames / mean_spacing, 3) if mean_spacing else None

    report = {
        "train_transform": {
            "resize": [256, 256],
            "random_resized_crop_scale": [0.85, 1.0],
            "random_resized_crop_ratio": [0.9, 1.1],
            "crop_area_retained_fraction_range": [train_crop_area_min, train_crop_area_max],
            "temporal_jitter_frames": jitter_frames,
            "consistent_clip_transform": True,
        },
        "eval_transform": {
            "resize": [240, 240],
            "center_crop_output": 224,
            "temporal_jitter_frames": 0,
        },
        "tiny_overfit_transform_config": {
            "frozen": {
                "deterministic_eval_transforms": tiny_frozen.get("deterministic_eval_transforms") if tiny_frozen else None,
                "temporal_jitter": tiny_frozen.get("temporal_jitter") if tiny_frozen else None,
                "best_train_accuracy": tiny_frozen.get("best_train_accuracy") if tiny_frozen else None,
            },
            "layer4": {
                "deterministic_eval_transforms": tiny_layer4.get("deterministic_eval_transforms") if tiny_layer4 else None,
                "temporal_jitter": tiny_layer4.get("temporal_jitter") if tiny_layer4 else None,
                "best_train_accuracy": tiny_layer4.get("best_train_accuracy") if tiny_layer4 else None,
            },
        },
        "mean_native10_train_frame_spacing_at_16_samples": mean_spacing,
        "jitter_fraction_of_mean_spacing": jitter_fraction_of_spacing,
        "evidence_summary": (
            "Tiny-overfit used eval_spatial_transforms (deterministic center-crop, no RandomResizedCrop) and "
            "jitter_frames=0 for BOTH backbone modes, and reached 96-98% train accuracy on a 50-clip subset. "
            "Real training uses RandomResizedCrop (crop area 85-100% of a 256x256 resize, so at most ~15% area "
            "loss and mild 0.9-1.1 aspect distortion) plus +/-2 frame temporal jitter. "
            f"The +/-2 frame jitter is {jitter_fraction_of_spacing if jitter_fraction_of_spacing else 'N/A'} "
            "of the mean spacing between the 16 uniformly sampled frames in native_10 train clips, which is "
            "non-trivial (jitter is a meaningful fraction of the natural spacing) but the spatial crop is mild. "
            "This is evidence that augmentation differs meaningfully between tiny-overfit and real training, but "
            "the magnitude of spatial cropping alone is unlikely to fully explain a drop from ~96% to ~13% "
            "generalization; it is a contributing-but-unproven factor, not an established sole cause."
        ),
    }
    dump("augmentation_analysis.json", report)
    return report


# ---------------------------------------------------------------------------
# Diagnostic 9: layer4 epoch-1 NaN investigation (single-batch probes only)
# ---------------------------------------------------------------------------


def _tensor_finite_stats(t: torch.Tensor) -> dict:
    t = t.detach().float()
    return {
        "finite": bool(torch.isfinite(t).all()),
        "min": float(t.min()) if t.numel() else None,
        "max": float(t.max()) if t.numel() else None,
        "abs_max": float(t.abs().max()) if t.numel() else None,
        "mean": float(t.mean()) if t.numel() else None,
    }


def _single_batch_probe(cfg, device: torch.device, batch_size: int, use_amp: bool, seed: int) -> dict:
    seed_pipeline(seed)
    model = build_model(cfg).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    loader, _ = build_dataloader("train", cfg, batch_size=batch_size, shuffle=True)
    batch = next(iter(loader))
    clips = batch["video"].to(device)
    labels = batch["label"].to(device)

    stages: dict[str, Any] = {"input": _tensor_finite_stats(clips), "batch_size": int(labels.size(0)), "use_amp": use_amp}
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    model.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        encoded = model.encode_frames(clips)
        stages["encoder_output"] = _tensor_finite_stats(encoded)
        gru_out, _ = model.gru(encoded)
        pooled = gru_out[:, -1, :]
        stages["gru_pooled_output"] = _tensor_finite_stats(pooled)
        logits = model.classifier(pooled)
        stages["logits"] = _tensor_finite_stats(logits)
        loss = criterion(logits, labels)
        stages["loss"] = {"value": float(loss.detach().float()), "finite": bool(torch.isfinite(loss))}

    if use_amp:
        scaler.scale(loss).backward()
        scale_value = float(scaler.get_scale())
        scaled_grad = gradient_flow_report(model)["layer4"]
        grads = [p.grad for p in model.encoder.layer4.parameters() if p.grad is not None]
        unscaled_stats = {"finite": None, "mean_abs": None, "max_abs": None}
        if grads:
            stacked = torch.cat([g.detach().flatten().float() for g in grads]) / scale_value
            unscaled_stats = {
                "finite": bool(torch.isfinite(stacked).all()),
                "mean_abs": float(stacked.abs().mean()),
                "max_abs": float(stacked.abs().max()),
            }
        stages["layer4_grad_as_stored_scaled"] = scaled_grad
        stages["layer4_grad_manually_unscaled"] = unscaled_stats
        stages["scaler_scale_value"] = scale_value
    else:
        loss.backward()
        stages["layer4_grad_fp32"] = gradient_flow_report(model)["layer4"]

    return stages


def diagnostic_9(device: torch.device) -> dict:
    print("Diagnostic 9: layer4 NaN investigation")
    cfg = replace(native_10_config(), backbone_train_mode="layer4", freeze_backbone=False, batch_size=4, num_workers=0)

    history = json.loads(LAYER4_HISTORY.read_text(encoding="utf-8"))["epochs"]
    epoch1 = next((e for e in history if e["epoch"] == 1), None)
    epoch2 = next((e for e in history if e["epoch"] == 2), None)

    n_train = 149
    batch_size = 4
    num_batches = math.ceil(n_train / batch_size)
    last_batch_size = n_train - (num_batches - 1) * batch_size

    trials = []
    for trial in range(3):
        try:
            probe_amp_b4 = _single_batch_probe(cfg, device, batch_size=4, use_amp=True, seed=100 + trial)
        except RuntimeError as exc:
            probe_amp_b4 = {"error": str(exc)}
        try:
            probe_amp_b1 = _single_batch_probe(cfg, device, batch_size=1, use_amp=True, seed=100 + trial)
        except RuntimeError as exc:
            probe_amp_b1 = {"error": str(exc)}
        try:
            probe_fp32_b1 = _single_batch_probe(cfg, device, batch_size=1, use_amp=False, seed=100 + trial)
        except RuntimeError as exc:
            probe_fp32_b1 = {"error": str(exc)}
        trials.append({"seed": 100 + trial, "amp_batch4": probe_amp_b4, "amp_batch1": probe_amp_b1, "fp32_batch1": probe_fp32_b1})

    report = {
        "from_training_log": {
            "epoch_1_layer4_gradient": epoch1.get("layer4_gradient") if epoch1 else None,
            "epoch_2_layer4_gradient": epoch2.get("layer4_gradient") if epoch2 else None,
            "note": (
                "training_history.json records ONE gradient snapshot per epoch, taken from whatever batch was "
                "last processed in that epoch (gradient_flow_report is called once after the full epoch loop, "
                "so it reflects the LAST train batch of that epoch, not necessarily where within the epoch the "
                "instability first appeared)."
            ),
        },
        "dataset_epoch_structure": {
            "train_samples": n_train,
            "batch_size": batch_size,
            "num_batches_per_epoch": num_batches,
            "last_batch_size": last_batch_size,
            "note": "drop_last=False, so the final batch of every epoch has only 1 sample (149 = 37*4 + 1).",
        },
        "reproducibility_caveat": (
            "This probe uses a freshly re-initialized layer4 model with the same seeding calls as "
            "train_native_10_layer4.py, but DataLoader uses num_workers=2 in the real training run; on this "
            "Windows (spawn) multiprocessing setup, worker-process scheduling order is not guaranteed to be "
            "bit-identical across runs even with the same seed. This probe therefore characterizes the MECHANISM "
            "(is small-batch AMP training on a freshly-initialized layer4 model prone to non-finite gradients) "
            "rather than certifying a bit-exact reproduction of the original epoch-1 batch."
        ),
        "trials": trials,
    }
    dump("layer4_nan_analysis.json", report)
    return report


# ---------------------------------------------------------------------------
# Diagnostic 10: dataset size / statistical limitation
# ---------------------------------------------------------------------------


def diagnostic_10(data_dist: dict, signer_by_split: dict) -> dict:
    print("Diagnostic 10: dataset size limitation")
    train_counts = {g: s["train"]["count"] for g, s in data_dist["per_class_split"].items()}
    train_signers = {g: s["train"]["unique_signers"] for g, s in data_dist["per_class_split"].items()}
    clips_per_signer_per_class = {}
    for gloss, signers in signer_by_split["train"].items():
        clips_per_signer_per_class[gloss] = round(sum(signers.values()) / len(signers), 3) if signers else 0

    report = {
        "total_train_clips": sum(train_counts.values()),
        "clips_per_class": train_counts,
        "clips_per_class_stats": stat_block(list(train_counts.values())),
        "unique_signers_per_class_train": train_signers,
        "unique_signers_per_class_stats": stat_block(list(train_signers.values())),
        "mean_clips_per_signer_per_class_train": clips_per_signer_per_class,
        "implication": (
            "With a mean of roughly 1-2 clips per signer per class in training, most classes are represented by "
            "only a handful of distinct people. Any visual attribute correlated with a particular signer "
            "(clothing, skin tone, camera framing, background) is nearly as predictive of the training label as "
            "the sign motion itself, because the model rarely sees the SAME class performed by many DIFFERENT "
            "signers to average out signer-specific appearance. This does not by itself prove the model learned "
            "signer identity (see diagnostic 5's silhouette comparison for direct evidence), but it describes a "
            "concrete statistical mechanism by which a class-vs-signer shortcut is available in this dataset."
        ),
        "what_would_help": (
            "The limiting factor by these counts is signers-per-class, not raw clip count: adding more clips "
            "from the SAME already-seen training signers would not by itself teach signer-invariant features, "
            "since the model can still associate class with the small existing signer set. Additional clips of "
            "each of the 10 glosses from NEW, previously-unseen training signers would directly increase the "
            "signer-diversity-per-class the model is exposed to during training, which is what should force it "
            "away from signer-specific shortcuts if a shortcut is in fact being learned."
        ),
    }
    dump("dataset_size_report.json", report)
    return report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = native_10_config()
    rows_by_split = load_all_rows(cfg)
    glosses = sorted({row["gloss"] for rows in rows_by_split.values() for row in rows})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dist = diagnostic_1(cfg, rows_by_split)
    diagnostic_2(cfg, rows_by_split, glosses)
    diagnostic_3(cfg, rows_by_split, glosses)
    diagnostic_4(cfg, rows_by_split, glosses)
    diagnostic_5_and_6(device)
    signer_by_split = diagnostic_6_train_val_signers(rows_by_split)
    dump("signer_distribution_by_split.json", signer_by_split)
    diagnostic_7()
    diagnostic_8(data_dist)
    diagnostic_9(device)
    diagnostic_10(data_dist, signer_by_split)
    print("\nAll Phase 5 diagnostics complete.")
    print(f"Output directory: {OUT}")


if __name__ == "__main__":
    main()
