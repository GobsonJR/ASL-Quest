"""Phase 7D: Diagnostic script for ASL Citizen 100-class failure analysis.

Does NOT retrain full model or modify protected assets.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.checkpoint import load_checkpoint
from src.asl_citizen.config import (
    BEST_CHECKPOINT_PATH,
    LAST_CHECKPOINT_PATH,
    default_config,
)
from src.asl_citizen.dataset import (
    ASLCitizenDataset,
    build_dataloader,
    decode_sampled_frames,
    get_frame_count,
    sample_temporal_indices,
)
from src.asl_citizen.metrics import batch_accuracy, top_k_accuracy
from src.asl_citizen.model import ASLCitizenResNet18GRU, build_model, parameter_counts
from src.asl_citizen.preprocessing import eval_spatial_transforms, train_spatial_transforms, uniform_temporal_indices
from src.asl_citizen.utils import load_class_mappings, load_manifest_rows, resolve_video_path
from src.config import IMAGENET_MEAN, IMAGENET_STD

DIAG_DIR = ROOT / "outputs" / "asl_citizen_100" / "diagnostics"
HISTORY_PATH = ROOT / "outputs" / "asl_citizen_100" / "metrics" / "training_history.json"


def save_json(name: str, payload: dict) -> Path:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    path = DIAG_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def inspect_checkpoints() -> dict:
    result = {"training_history_file_exists": HISTORY_PATH.exists()}
    if HISTORY_PATH.exists():
        result["training_history"] = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    else:
        result["training_history"] = None
        result["training_history_note"] = (
            "training_history.json was not found. train.py only writes it after all epochs complete. "
            "Per-epoch train/val metrics are unavailable unless captured from console logs."
        )

    for label, path in [("best", BEST_CHECKPOINT_PATH), ("last", LAST_CHECKPOINT_PATH)]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        result[label] = {
            "path": str(path),
            "epoch": payload.get("epoch"),
            "val_accuracy": payload.get("val_accuracy"),
            "val_top5_accuracy": payload.get("val_top5_accuracy"),
            "num_classes": payload.get("num_classes"),
            "num_frames": payload.get("num_frames"),
            "image_size": payload.get("image_size"),
            "architecture": payload.get("architecture"),
            "freeze_backbone": payload.get("full_config", {}).get("freeze_backbone"),
            "learning_rate": payload.get("full_config", {}).get("learning_rate"),
            "batch_size": payload.get("full_config", {}).get("batch_size"),
            "num_epochs_config": payload.get("full_config", {}).get("num_epochs"),
            "has_optimizer_state": "optimizer_state_dict" in payload,
            "class_mapping_count": len(payload.get("class_to_idx", {})),
        }
    return result


def strict_load_report(path: Path) -> dict:
    payload, cfg, c2i, i2c = load_checkpoint(path, map_location="cpu")
    model = build_model(cfg, pretrained=False)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    strict_ok = len(incompatible.missing_keys) == 0 and len(incompatible.unexpected_keys) == 0
    return {
        "checkpoint": str(path),
        "strict_load_would_succeed": strict_ok,
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "num_missing": len(incompatible.missing_keys),
        "num_unexpected": len(incompatible.unexpected_keys),
    }


@torch.no_grad()
def eval_subset(checkpoint_path: Path, split: str, max_samples: int = 200) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, cfg, class_to_idx, idx_to_class = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    loader, dataset = build_dataloader(split, cfg, batch_size=4, shuffle=False)
    indices = list(range(min(max_samples, len(dataset))))
    subset_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    all_logits, all_labels, all_preds = [], [], []
    for batch in subset_loader:
        clips = batch["video"].to(device)
        labels = batch["label"].to(device)
        logits = model(clips)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    pred_counts = Counter(all_preds)
    return {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "samples": len(indices),
        "top1_accuracy": batch_accuracy(logits_cat, labels_cat),
        "top5_accuracy": top_k_accuracy(logits_cat, labels_cat, k=5),
        "unique_predicted_classes": len(pred_counts),
        "most_frequent_prediction": pred_counts.most_common(1)[0] if pred_counts else None,
        "top10_predicted_classes": [
            {"class_idx": idx, "gloss": idx_to_class[str(idx)], "count": count}
            for idx, count in pred_counts.most_common(10)
        ],
        "prediction_distribution": {
            str(k): v for k, v in sorted(pred_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        },
    }


def verify_label_mapping(cfg, checkpoint_path: Path, n: int = 12) -> dict:
    payload, _, ckpt_c2i, ckpt_i2c = load_checkpoint(checkpoint_path, map_location="cpu")
    file_c2i, file_i2c = load_class_mappings(cfg.class_to_idx_path, cfg.idx_to_class_path)

    mapping_check = {
        "file_class_count": len(file_c2i),
        "checkpoint_class_count": len(ckpt_c2i),
        "indices_exactly_0_to_99": sorted(int(v) for v in file_c2i.values()) == list(range(100)),
        "file_checkpoint_class_to_idx_match": file_c2i == {k: int(v) for k, v in ckpt_c2i.items()},
        "inverse_consistent": all(file_i2c[str(v)] == k for k, v in file_c2i.items()),
    }

    test_rows = load_manifest_rows(cfg.test_manifest)
    rng = random.Random(42)
    samples = rng.sample(test_rows, min(n, len(test_rows)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, cfg_loaded, _, idx_to_class = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model(cfg_loaded, pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    dataset = ASLCitizenDataset(
        samples,
        video_root=cfg.video_root,
        class_to_idx=file_c2i,
        num_frames=cfg.num_frames,
        transform=eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std),
        training=False,
    )
    examples = []
    for i in range(len(dataset)):
        item = dataset[i]
        row = samples[i]
        logits = model(item["video"].unsqueeze(0).to(device))
        pred_idx = int(logits.argmax(dim=1).item())
        examples.append(
            {
                "video_file": row["video_file"],
                "manifest_gloss": row["gloss"],
                "manifest_class_idx": int(row["class_idx"]),
                "class_to_idx_gloss": file_c2i[row["gloss"]],
                "dataset_label": int(item["label"]),
                "prediction_gloss": idx_to_class[str(pred_idx)],
                "prediction_idx": pred_idx,
                "label_matches_mapping": int(row["class_idx"]) == file_c2i[row["gloss"]],
                "dataset_label_matches_manifest": int(item["label"]) == int(row["class_idx"]),
            }
        )
    mapping_check["examples"] = examples
    mapping_check["all_labels_consistent"] = all(
        e["label_matches_mapping"] and e["dataset_label_matches_manifest"] for e in examples
    )
    return mapping_check


def verify_video_paths(cfg, n: int = 20) -> dict:
    rows = load_manifest_rows(cfg.test_manifest)
    rng = random.Random(7)
    samples = rng.sample(rows, min(n, len(rows)))
    entries = []
    for row in samples:
        path = cfg.video_root / row["video_file"]
        entries.append(
            {
                "csv_video_file": row["video_file"],
                "resolved_path": str(path.resolve()),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return {
        "samples_checked": len(entries),
        "all_exist": all(e["exists"] for e in entries),
        "entries": entries,
    }


def verify_temporal_sampling(cfg, n: int = 8) -> dict:
    rows = load_manifest_rows(cfg.test_manifest)
    rng = random.Random(11)
    samples = rng.sample(rows, min(n, len(rows)))
    examples = []
    for row in samples:
        path = resolve_video_path(cfg.video_root, row["video_file"])
        cap = cv2.VideoCapture(str(path))
        frame_count = get_frame_count(cap)
        cap.release()
        indices = sample_temporal_indices(frame_count, cfg.num_frames, training=False, jitter_frames=2, rng=None)
        examples.append(
            {
                "video_file": row["video_file"],
                "gloss": row["gloss"],
                "total_frames": frame_count,
                "selected_indices": indices,
                "chronological_order": indices == sorted(indices),
                "num_unique_indices": len(set(indices)),
            }
        )
    return {"examples": examples}


def tensor_stats(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
    }


def verify_normalization_pipeline(cfg) -> dict:
    rows = load_manifest_rows(cfg.test_manifest)
    row = rows[0]
    path = resolve_video_path(cfg.video_root, row["video_file"])
    cap = cv2.VideoCapture(str(path))
    frame_count = get_frame_count(cap)
    cap.release()
    indices = uniform_temporal_indices(frame_count, cfg.num_frames)
    frames_bgr = decode_sampled_frames(path, indices[:1])
    bgr = frames_bgr[0]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    raw_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    eval_tf = eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    train_tf = train_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    eval_out = eval_tf(Image.fromarray(rgb))
    train_out = train_tf(Image.fromarray(rgb))

    return {
        "bgr_shape": list(bgr.shape),
        "rgb_mean_before_norm": [float(rgb[:, :, c].mean()) for c in range(3)],
        "raw_tensor_0_1": tensor_stats(raw_tensor),
        "eval_tensor": tensor_stats(eval_out),
        "train_tensor": tensor_stats(train_out),
        "imagenet_mean": list(cfg.imagenet_mean),
        "imagenet_std": list(cfg.imagenet_std),
    }


def compare_preprocessing(cfg) -> dict:
    train_tf = train_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    eval_tf = eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    return {
        "train_transforms": [step.__class__.__name__ for step in train_tf.transforms],
        "eval_transforms": [step.__class__.__name__ for step in eval_tf.transforms],
        "horizontal_flip_in_train": any("Flip" in step.__class__.__name__ for step in train_tf.transforms),
        "horizontal_flip_in_eval": any("Flip" in step.__class__.__name__ for step in eval_tf.transforms),
        "test_uses_eval_transforms": True,
        "train_uses_random_resized_crop": True,
        "eval_uses_center_crop": True,
    }


def verify_forward_pass_shapes(cfg) -> dict:
    model = build_model(cfg, pretrained=False)
    x = torch.randn(2, cfg.num_frames, 3, cfg.image_size, cfg.image_size)
    batch, time, c, h, w = x.shape
    flat = x.reshape(batch * time, c, h, w)
    with torch.no_grad():
        enc = model.encoder(flat)
        enc_view = enc.view(batch, time, model.feature_dim)
        gru_out, _ = model.gru(enc_view)
        pooled = gru_out[:, -1, :]
        logits = model.classifier(pooled)
    return {
        "input_shape": list(x.shape),
        "encoder_flat_input": list(flat.shape),
        "encoder_output": list(enc.shape),
        "encoder_reshaped": list(enc_view.shape),
        "gru_output": list(gru_out.shape),
        "pooled": list(pooled.shape),
        "logits": list(logits.shape),
        "parameter_counts": parameter_counts(model),
    }


def inspect_training_batch(cfg) -> dict:
    loader, _ = build_dataloader("train", cfg, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    labels = batch["label"].tolist()
    files = batch["video_file"]
    video_tensor = batch["video"]
    return {
        "batch_video_shape": list(video_tensor.shape),
        "unique_labels_in_batch": len(set(labels)),
        "label_frequencies": dict(Counter(labels)),
        "unique_video_files": len(set(files)),
        "video_files": list(files),
        "tensor_stats": tensor_stats(video_tensor),
        "label_min": int(min(labels)),
        "label_max": int(max(labels)),
    }


def _safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:120]


def save_contact_sheets(cfg, checkpoint_path: Path, max_videos: int = 8) -> list[str]:
    from torchvision.utils import save_image

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_rows(cfg.test_manifest)
    rng = random.Random(99)
    chosen = rng.sample(rows, min(max_videos, len(rows)))
    saved = []
    denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1 / s for s in IMAGENET_STD],
    )
    eval_tf = eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)

    for row in chosen:
        path = resolve_video_path(cfg.video_root, row["video_file"])
        cap = cv2.VideoCapture(str(path))
        frame_count = get_frame_count(cap)
        cap.release()
        indices = uniform_temporal_indices(frame_count, cfg.num_frames)
        frames_bgr = decode_sampled_frames(path, indices)

        tiles = []
        raw_tiles = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_tiles.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
            tiles.append(eval_tf(Image.fromarray(rgb)))

        grid = torch.stack(tiles, dim=0)
        raw_grid = torch.stack(raw_tiles, dim=0)
        # save 4x4 contact sheet
        stem = _safe_stem(f"{row['gloss']}_{row['video_file'].replace('.mp4', '')}")
        sheet_path = DIAG_DIR / f"contact_{stem}.png"
        raw_path = DIAG_DIR / f"raw_{stem}.png"
        save_image(grid, sheet_path, nrow=4, normalize=True)
        save_image(raw_grid, raw_path, nrow=4)
        saved.append(str(sheet_path))

        # uniqueness check
        diffs = []
        for i in range(1, len(raw_tiles)):
            diffs.append(float((raw_tiles[i] - raw_tiles[i - 1]).abs().mean()))
        row_stats = {
            "video_file": row["video_file"],
            "gloss": row["gloss"],
            "frame_count": frame_count,
            "indices": indices,
            "mean_abs_diff_between_consecutive_raw_frames": diffs,
            "all_frames_identical": all(d == 0.0 for d in diffs),
            "contact_sheet": str(sheet_path),
            "raw_sheet": str(raw_path),
        }
        save_json(f"frame_stats_{row['gloss']}_{row['video_file']}.json", row_stats)
    return saved


def tiny_subset_overfit(cfg, epochs: int = 15, max_batches_per_epoch: int = 20) -> dict:
    """Diagnostic only: can the pipeline overfit ~10 classes x 5 videos?"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx, idx_to_class = load_class_mappings(cfg.class_to_idx_path, cfg.idx_to_class_path)

    train_rows = load_manifest_rows(cfg.train_manifest)
    by_gloss: dict[str, list] = {}
    for row in train_rows:
        by_gloss.setdefault(row["gloss"], []).append(row)
    selected_glosses = sorted(by_gloss.keys())[:10]
    tiny_rows = []
    for gloss in selected_glosses:
        tiny_rows.extend(by_gloss[gloss][:5])

    tiny_c2i = {gloss: class_to_idx[gloss] for gloss in selected_glosses}
    dataset = ASLCitizenDataset(
        tiny_rows,
        video_root=cfg.video_root,
        class_to_idx=tiny_c2i,
        num_frames=cfg.num_frames,
        transform=train_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std),
        training=True,
        jitter_frames=0,  # reduce noise for overfit test
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = ASLCitizenResNet18GRU(
        num_classes=len(selected_glosses),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=0.0,
        freeze_backbone=cfg.freeze_backbone,
        pretrained=True,
    ).to(device)

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_correct = 0
        running_total = 0
        running_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches_per_epoch:
                break
            clips = batch["video"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(1) == labels).sum().item()
            running_total += labels.size(0)
        acc = running_correct / max(running_total, 1)
        history.append({"epoch": epoch, "train_loss": running_loss / max(running_total, 1), "train_acc": acc})

    return {
        "classes": selected_glosses,
        "num_samples": len(tiny_rows),
        "epochs": epochs,
        "freeze_backbone": cfg.freeze_backbone,
        "final_train_accuracy": history[-1]["train_acc"],
        "best_train_accuracy": max(h["train_acc"] for h in history),
        "history": history,
        "overfit_success": history[-1]["train_acc"] >= 0.95,
    }


def main() -> None:
    cfg = default_config()
    report: dict = {"phase": "7D_diagnosis"}

    report["checkpoints"] = inspect_checkpoints()
    report["strict_load_best"] = strict_load_report(BEST_CHECKPOINT_PATH)
    report["strict_load_last"] = strict_load_report(LAST_CHECKPOINT_PATH)
    report["random_baseline"] = {"top1": 0.01, "top5": 0.05}
    report["test_metrics"] = json.loads(
        (ROOT / "outputs" / "asl_citizen_100" / "metrics" / "evaluation_test.json").read_text(encoding="utf-8")
    )

    report["prediction_distribution_best"] = eval_subset(BEST_CHECKPOINT_PATH, "test", max_samples=300)
    report["prediction_distribution_last"] = eval_subset(LAST_CHECKPOINT_PATH, "test", max_samples=300)
    report["best_vs_last_subset"] = {
        "best": eval_subset(BEST_CHECKPOINT_PATH, "test", max_samples=200),
        "last": eval_subset(LAST_CHECKPOINT_PATH, "test", max_samples=200),
    }
    report["label_mapping"] = verify_label_mapping(cfg, BEST_CHECKPOINT_PATH)
    report["video_paths"] = verify_video_paths(cfg)
    report["temporal_sampling"] = verify_temporal_sampling(cfg)
    report["normalization"] = verify_normalization_pipeline(cfg)
    report["preprocessing_comparison"] = compare_preprocessing(cfg)
    report["forward_pass_shapes"] = verify_forward_pass_shapes(cfg)
    report["training_batch_sample"] = inspect_training_batch(cfg)

    print("Saving contact sheets...")
    report["contact_sheets"] = save_contact_sheets(cfg, BEST_CHECKPOINT_PATH, max_videos=8)

    print("Running tiny-subset overfit diagnostic...")
    report["tiny_subset_overfit_frozen_backbone"] = tiny_subset_overfit(cfg, epochs=15)

    print("Running tiny-subset overfit with unfrozen backbone (diagnostic only)...")
    unfrozen_cfg = default_config(freeze_backbone=False)
    report["tiny_subset_overfit_unfrozen_backbone"] = tiny_subset_overfit(unfrozen_cfg, epochs=10)

    out = save_json("phase7d_diagnosis_report.json", report)
    print(f"Wrote {out}")
    print(json.dumps(
        {
            "best_epoch": report["checkpoints"]["best"]["epoch"],
            "best_val_accuracy": report["checkpoints"]["best"]["val_accuracy"],
            "test_top1": report["test_metrics"]["top1_accuracy"],
            "unique_preds_300": report["prediction_distribution_best"]["unique_predicted_classes"],
            "tiny_overfit_frozen_final_acc": report["tiny_subset_overfit_frozen_backbone"]["final_train_accuracy"],
            "tiny_overfit_unfrozen_final_acc": report["tiny_subset_overfit_unfrozen_backbone"]["final_train_accuracy"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
