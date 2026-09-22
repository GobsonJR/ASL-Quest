"""Layer-4 100-class failure diagnosis.

Read-only on checkpoints/manifests/datasets. Writes only under
outputs/asl_citizen_100/diagnostics/layer4_failure/.
Does not start training.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.checkpoint import load_checkpoint
from src.asl_citizen.config import BEST_CHECKPOINT_PATH, LAST_CHECKPOINT_PATH, default_config
from src.asl_citizen.dataset import (
    ASLCitizenDataset,
    decode_sampled_frames,
    get_frame_count,
    sample_temporal_indices,
)
from src.asl_citizen.metrics import batch_accuracy, top_k_accuracy
from src.asl_citizen.model import build_model, build_optimizer, encoder_trainability_report, gradient_flow_report
from src.asl_citizen.preprocessing import (
    eval_spatial_transforms,
    train_spatial_transforms,
    uniform_temporal_indices,
)
from src.asl_citizen.utils import load_class_mappings, load_manifest_rows, resolve_video_path

OUT = ROOT / "outputs" / "asl_citizen_100" / "diagnostics" / "layer4_failure"
HISTORY_PATH = ROOT / "outputs" / "asl_citizen_100" / "metrics" / "training_history.json"
FROZEN_PRED_BEST = ROOT / "outputs" / "asl_citizen_100" / "diagnostics" / "02_prediction_best.json"
FROZEN_PRED_LAST = ROOT / "outputs" / "asl_citizen_100" / "diagnostics" / "03_prediction_last.json"
FROZEN_EVAL = ROOT / "outputs" / "asl_citizen_100" / "metrics" / "evaluation_test.json"
FROZEN_META = ROOT / "outputs" / "asl_citizen_100" / "diagnostics" / "01_checkpoint_meta.json"


def dump(name: str, payload) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path.name}")
    return path


def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for n in counts.values():
        p = n / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return ent


def load_model(checkpoint_path: Path, device: torch.device):
    payload, cfg, class_to_idx, idx_to_class = load_checkpoint(checkpoint_path, map_location=device)
    model = build_model(cfg, pretrained=False)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return payload, cfg, class_to_idx, idx_to_class, model, incompatible


def make_eval_dataset(split: str, cfg, class_to_idx) -> ASLCitizenDataset:
    manifest = {"train": cfg.train_manifest, "val": cfg.val_manifest, "test": cfg.test_manifest}[split]
    rows = load_manifest_rows(manifest)
    return ASLCitizenDataset(
        rows,
        video_root=cfg.video_root,
        class_to_idx=class_to_idx,
        num_frames=cfg.num_frames,
        transform=eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std),
        training=False,
        jitter_frames=0,
        seed=cfg.seed,
    )


def make_train_aug_dataset(cfg, class_to_idx) -> ASLCitizenDataset:
    rows = load_manifest_rows(cfg.train_manifest)
    return ASLCitizenDataset(
        rows,
        video_root=cfg.video_root,
        class_to_idx=class_to_idx,
        num_frames=cfg.num_frames,
        transform=train_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std),
        training=True,
        jitter_frames=cfg.train_jitter_frames,
        seed=cfg.seed,
    )


@torch.no_grad()
def infer_full(model, dataset, device, batch_size: int = 4, max_samples: int | None = None):
    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(max_samples, len(dataset)))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_all, labels_all, preds, glosses, files = [], [], [], [], []
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    n = 0
    for batch in loader:
        clips = batch["video"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(clips)
        loss_sum += float(criterion(logits, labels).item())
        n += labels.size(0)
        logits_cpu = logits.cpu()
        labels_cpu = labels.cpu()
        logits_all.append(logits_cpu)
        labels_all.append(labels_cpu)
        preds.extend(logits_cpu.argmax(1).tolist())
        glosses.extend(list(batch["gloss"]))
        files.extend(list(batch["video_file"]))
    logits_cat = torch.cat(logits_all)
    labels_cat = torch.cat(labels_all)
    return {
        "logits": logits_cat,
        "labels": labels_cat,
        "preds": preds,
        "glosses": glosses,
        "files": files,
        "loss": loss_sum / max(n, 1),
        "top1": batch_accuracy(logits_cat, labels_cat),
        "top5": top_k_accuracy(logits_cat, labels_cat, 5),
        "n": n,
    }


def summarize_predictions(preds, labels, idx_to_class, num_classes: int = 100) -> dict:
    pred_counts = Counter(preds)
    true_counts = Counter(int(x) for x in labels)
    top20 = []
    for idx, count in pred_counts.most_common(20):
        top20.append(
            {
                "class_idx": int(idx),
                "gloss": idx_to_class.get(str(idx), "?"),
                "pred_count": int(count),
                "true_count": int(true_counts.get(int(idx), 0)),
            }
        )
    return {
        "unique_predicted_classes": len(pred_counts),
        "prediction_entropy_bits": entropy_from_counts(pred_counts),
        "max_entropy_bits_uniform_100": math.log(num_classes, 2),
        "collapse_ratio_top1": (pred_counts.most_common(1)[0][1] / len(preds)) if preds else 0.0,
        "collapse_ratio_top3": (sum(c for _, c in pred_counts.most_common(3)) / len(preds)) if preds else 0.0,
        "top20_predicted": top20,
        "true_class_count": len(true_counts),
        "true_entropy_bits": entropy_from_counts(true_counts),
        "never_predicted_count": num_classes - len(pred_counts),
    }


def per_class_report(preds, labels, idx_to_class, num_classes: int = 100) -> dict:
    y_true = [int(x) for x in labels]
    y_pred = [int(x) for x in preds]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    rows = []
    precisions, recalls, f1s = [], [], []
    for i in range(num_classes):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        pred_pos = int(cm[:, i].sum())
        rec = tp / support if support else 0.0
        pre = tp / pred_pos if pred_pos else 0.0
        f1 = (2 * pre * rec / (pre + rec)) if (pre + rec) else 0.0
        precisions.append(pre)
        recalls.append(rec)
        f1s.append(f1)
        rows.append(
            {
                "class_idx": i,
                "gloss": idx_to_class.get(str(i), "?"),
                "support": support,
                "correct": tp,
                "accuracy": rec,
                "precision": pre,
                "recall": rec,
                "f1": f1,
                "predicted_as_this": pred_pos,
            }
        )
    ranked = sorted(rows, key=lambda r: (r["f1"], r["accuracy"], r["support"]), reverse=True)
    return {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": rows,
        "strongest_20": ranked[:20],
        "weakest_20": list(reversed(ranked[-20:])),
        "confusion_matrix_shape": list(cm.shape),
        "confusion_matrix": cm,
    }


def checkpoint_meta(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    opt = payload.get("optimizer_state_dict")
    groups = []
    if opt and "param_groups" in opt:
        for i, g in enumerate(opt["param_groups"]):
            groups.append(
                {
                    "group": i,
                    "lr": g.get("lr"),
                    "weight_decay": g.get("weight_decay"),
                    "num_params": len(g.get("params", [])),
                }
            )
    cfg = payload.get("full_config", {})
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime,
        "epoch": payload.get("epoch"),
        "val_accuracy": payload.get("val_accuracy"),
        "val_top5_accuracy": payload.get("val_top5_accuracy"),
        "freeze_backbone": cfg.get("freeze_backbone"),
        "backbone_train_mode": cfg.get("backbone_train_mode"),
        "learning_rate": cfg.get("learning_rate"),
        "head_learning_rate": cfg.get("head_learning_rate"),
        "backbone_learning_rate": cfg.get("backbone_learning_rate"),
        "weight_decay": cfg.get("weight_decay"),
        "dropout": cfg.get("dropout"),
        "batch_size": cfg.get("batch_size"),
        "num_epochs": cfg.get("num_epochs"),
        "has_optimizer_state": opt is not None,
        "optimizer_param_groups": groups,
        "architecture": payload.get("architecture"),
        "num_classes": payload.get("num_classes"),
    }


def check_signers(cfg) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        rows = load_manifest_rows(getattr(cfg, f"{split}_manifest") if split != "test" else cfg.test_manifest)
        ids = sorted({r["participant_id"] for r in rows})
        out[split] = {"n_videos": len(rows), "n_signers": len(ids), "signer_ids": ids}
    train_ids = set(out["train"]["signer_ids"])
    val_ids = set(out["val"]["signer_ids"])
    test_ids = set(out["test"]["signer_ids"])
    out["intersections"] = {
        "train_val": sorted(train_ids & val_ids),
        "train_test": sorted(train_ids & test_ids),
        "val_test": sorted(val_ids & test_ids),
    }
    out["signer_independent"] = not any(out["intersections"].values())
    return out


def visual_sanity(cfg, class_to_idx, n_videos: int = 6) -> dict:
    vis_dir = OUT / "frames"
    vis_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_rows(cfg.val_manifest)
    by_gloss = defaultdict(list)
    for row in rows:
        by_gloss[row["gloss"]].append(row)
    rng = random.Random(42)
    glosses = rng.sample(sorted(by_gloss.keys()), min(n_videos, len(by_gloss)))
    eval_tf = eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    train_tf = train_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(cfg.imagenet_mean, cfg.imagenet_std)],
        std=[1 / s for s in cfg.imagenet_std],
    )
    reports = []
    for gloss in glosses:
        row = by_gloss[gloss][0]
        path = resolve_video_path(cfg.video_root, row["video_file"])
        cap = cv2.VideoCapture(str(path))
        frame_count = get_frame_count(cap)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        indices = uniform_temporal_indices(frame_count, cfg.num_frames)
        frames_bgr = decode_sampled_frames(path, indices)
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        eval_tiles = [eval_tf(Image.fromarray(rgb)) for rgb in rgb_frames]
        train_tiles = [train_tf(Image.fromarray(rgb)) for rgb in rgb_frames]
        raw = [torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0 for rgb in rgb_frames]
        diffs = [float((raw[i] - raw[i - 1]).abs().mean()) for i in range(1, len(raw))]
        eval_stack = torch.stack(eval_tiles)
        train_stack = torch.stack(train_tiles)
        # Independent train crops on identical first-frame copies: spatial inconsistency.
        copies = [train_tf(Image.fromarray(rgb_frames[0])) for _ in range(cfg.num_frames)]
        copy_stack = torch.stack(copies)
        copy_var = float((copy_stack - copy_stack.mean(0, keepdim=True)).abs().mean())
        frame_pair_diff = float((train_stack[1:] - train_stack[:-1]).abs().mean())
        eval_pair_diff = float((eval_stack[1:] - eval_stack[:-1]).abs().mean())

        stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in f"{gloss}_{row['video_file']}")[:80]
        save_image(torch.stack(raw), vis_dir / f"{stem}_01_raw_rgb.png", nrow=4)
        save_image(torch.stack([denorm(t) for t in eval_tiles]), vis_dir / f"{stem}_02_eval_224.png", nrow=4)
        save_image(torch.stack([denorm(t) for t in train_tiles]), vis_dir / f"{stem}_03_train_aug_224.png", nrow=4)
        # Channel order probe: save R and B as grayscale-ish 3ch
        r = raw[len(raw) // 2].clone()
        channel_probe = torch.stack([r * torch.tensor([1, 0, 0])[:, None, None], r * torch.tensor([0, 0, 1])[:, None, None]])
        save_image(channel_probe, vis_dir / f"{stem}_04_R_vs_B.png", nrow=2)

        reports.append(
            {
                "gloss": gloss,
                "video_file": row["video_file"],
                "manifest_class_idx": int(row["class_idx"]),
                "mapping_class_idx": int(class_to_idx[gloss]),
                "label_match": int(row["class_idx"]) == int(class_to_idx[gloss]),
                "frame_count": frame_count,
                "fps": fps,
                "sampled_indices": indices,
                "timestamps_sec": [ (i / fps) if fps else None for i in indices],
                "chronological": indices == sorted(indices),
                "mean_abs_diff_raw": diffs,
                "all_raw_frames_identical": all(d == 0.0 for d in diffs),
                "eval_consecutive_mean_abs_diff": eval_pair_diff,
                "train_aug_consecutive_mean_abs_diff": frame_pair_diff,
                "independent_crop_on_same_frame_mean_abs_dev": copy_var,
                "bgr_to_rgb_applied": True,
                "horizontal_flip_in_eval": False,
                "raw_hw": list(rgb_frames[0].shape[:2]),
                "eval_tensor_hw": list(eval_tiles[0].shape[-2:]),
            }
        )
    return {"videos": reports, "frames_dir": str(vis_dir)}


def inspect_transforms(cfg) -> dict:
    train_tf = train_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    eval_tf = eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)
    train_steps = []
    for step in train_tf.transforms:
        info = {"class": step.__class__.__name__}
        if hasattr(step, "size"):
            info["size"] = step.size
        if hasattr(step, "scale"):
            info["scale"] = list(step.scale)
        if hasattr(step, "ratio"):
            info["ratio"] = list(step.ratio)
        train_steps.append(info)
    eval_steps = [{"class": step.__class__.__name__, "size": getattr(step, "size", None)} for step in eval_tf.transforms]
    return {
        "train_steps": train_steps,
        "eval_steps": eval_steps,
        "horizontal_flip_train": any("Flip" in s["class"] for s in train_steps),
        "horizontal_flip_eval": any("Flip" in s["class"] for s in eval_steps),
        "note": (
            "Dataset applies the Compose independently to each of 16 frames. "
            "RandomResizedCrop therefore draws a NEW crop per frame, breaking temporal alignment."
        ),
    }


def activation_diagnostic(model, dataset, device, n: int = 8) -> dict:
    loader = DataLoader(
        torch.utils.data.Subset(dataset, list(range(min(n, len(dataset))))),
        batch_size=n,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    clips = batch["video"].to(device)
    labels = batch["label"].to(device)
    model.eval()
    with torch.no_grad():
        feats = model.encode_frames(clips)
        gru_out, _ = model.gru(feats)
        pooled = gru_out[:, -1, :]
        logits = model.classifier(pooled)
        probs = F.softmax(logits, dim=1)
    pred = logits.argmax(1)
    return {
        "batch": n,
        "labels": labels.cpu().tolist(),
        "preds": pred.cpu().tolist(),
        "feature_l2_mean": float(feats.norm(dim=-1).mean()),
        "feature_l2_std": float(feats.norm(dim=-1).std()),
        "feature_temporal_std_mean": float(feats.std(dim=1).mean()),
        "gru_out_l2_mean": float(gru_out.norm(dim=-1).mean()),
        "pooled_l2_mean": float(pooled.norm(dim=-1).mean()),
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std()),
        "logit_max_mean": float(logits.max(dim=1).values.mean()),
        "max_softmax_mean": float(probs.max(dim=1).values.mean()),
        "max_softmax_per_sample": probs.max(dim=1).values.cpu().tolist(),
        "logit_bias_top5": [
            {"class_idx": int(i), "mean_logit": float(v)}
            for v, i in zip(*torch.topk(logits.mean(0), k=5))
        ],
        "classifier_bias_top10": _classifier_bias(model),
    }


def _classifier_bias(model) -> list[dict]:
    linear = None
    for module in model.classifier.modules():
        if isinstance(module, torch.nn.Linear):
            linear = module
    if linear is None or linear.bias is None:
        return []
    bias = linear.bias.detach().cpu()
    vals, idxs = torch.topk(bias, k=min(10, bias.numel()))
    return [{"class_idx": int(i), "bias": float(v)} for v, i in zip(vals, idxs)]


def temporal_probe(model, dataset, device, n: int = 32) -> dict:
    subset = torch.utils.data.Subset(dataset, list(range(min(n, len(dataset)))))
    loader = DataLoader(subset, batch_size=4, shuffle=False, num_workers=0)
    model.eval()
    orig_correct = 0
    repeat_correct = 0
    shuffle_correct = 0
    orig_vs_repeat_agree = 0
    orig_vs_shuffle_agree = 0
    kl_repeat = []
    kl_shuffle = []
    total = 0
    with torch.no_grad():
        for batch in loader:
            clips = batch["video"].to(device)
            labels = batch["label"].to(device)
            B, T, C, H, W = clips.shape
            repeated = clips[:, 8:9].expand(-1, T, -1, -1, -1).contiguous()
            perm = torch.randperm(T, device=device)
            shuffled = clips[:, perm]
            o = model(clips)
            r = model(repeated)
            s = model(shuffled)
            orig_correct += int((o.argmax(1) == labels).sum())
            repeat_correct += int((r.argmax(1) == labels).sum())
            shuffle_correct += int((s.argmax(1) == labels).sum())
            orig_vs_repeat_agree += int((o.argmax(1) == r.argmax(1)).sum())
            orig_vs_shuffle_agree += int((o.argmax(1) == s.argmax(1)).sum())
            po, pr, ps = F.log_softmax(o, 1), F.softmax(r, 1), F.softmax(s, 1)
            kl_repeat.append(float(F.kl_div(po, pr, reduction="batchmean")))
            kl_shuffle.append(float(F.kl_div(po, ps, reduction="batchmean")))
            total += B
    return {
        "samples": total,
        "orig_top1": orig_correct / total,
        "repeated_frame_top1": repeat_correct / total,
        "shuffled_time_top1": shuffle_correct / total,
        "agree_orig_vs_repeat": orig_vs_repeat_agree / total,
        "agree_orig_vs_shuffle": orig_vs_shuffle_agree / total,
        "mean_kl_orig_vs_repeat": float(np.mean(kl_repeat)) if kl_repeat else 0.0,
        "mean_kl_orig_vs_shuffle": float(np.mean(kl_shuffle)) if kl_shuffle else 0.0,
        "interpretation_hint": (
            "If orig ≈ repeated ≈ shuffled, GRU is not using temporal order. "
            "If shuffle hurts more than repeat, order matters."
        ),
    }


def gradient_check(payload, cfg, dataset, device) -> dict:
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.train()
    report_before = encoder_trainability_report(model)
    optimizer = build_optimizer(
        model,
        head_lr=cfg.head_learning_rate,
        backbone_lr=cfg.backbone_learning_rate,
        weight_decay=cfg.weight_decay,
    )
    loader = DataLoader(
        torch.utils.data.Subset(dataset, list(range(min(4, len(dataset))))),
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    clips = batch["video"].to(device)
    labels = batch["label"].to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(clips)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    flow = gradient_flow_report(model)
    frozen_layer1_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.layer1.parameters())
    groups = [{"lr": g["lr"], "weight_decay": g["weight_decay"], "n_tensors": len(g["params"])} for g in optimizer.param_groups]
    return {
        "trainability": report_before,
        "optimizer_groups_reconstructed": groups,
        "scheduler_in_train_py": False,
        "grad_clip_in_train_py": False,
        "amp_used_in_train_py": True,
        "loss": float(loss.item()),
        "gradient_flow": flow,
        "layer1_unexpected_nonzero_grad": frozen_layer1_grad,
    }


def dataset_difficulty(cfg, class_to_idx, idx_to_class) -> dict:
    train_rows = load_manifest_rows(cfg.train_manifest)
    val_rows = load_manifest_rows(cfg.val_manifest)
    by_gloss = defaultdict(lambda: {"train": [], "val": []})
    for r in train_rows:
        by_gloss[r["gloss"]]["train"].append(r)
    for r in val_rows:
        by_gloss[r["gloss"]]["val"].append(r)
    rng = random.Random(0)
    sample_glosses = rng.sample(sorted(by_gloss.keys()), 12)
    details = []
    for gloss in sample_glosses:
        videos = (by_gloss[gloss]["train"] + by_gloss[gloss]["val"])[:8]
        durs = []
        signers_train = {r["participant_id"] for r in by_gloss[gloss]["train"]}
        signers_val = {r["participant_id"] for r in by_gloss[gloss]["val"]}
        for r in videos:
            path = resolve_video_path(cfg.video_root, r["video_file"])
            cap = cv2.VideoCapture(str(path))
            n = get_frame_count(cap)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            durs.append({"video_file": r["video_file"], "frames": n, "sec": n / fps if fps else None})
        details.append(
            {
                "gloss": gloss,
                "class_idx": class_to_idx[gloss],
                "n_train": len(by_gloss[gloss]["train"]),
                "n_val": len(by_gloss[gloss]["val"]),
                "train_signers": sorted(signers_train),
                "val_signers": sorted(signers_val),
                "signer_overlap": sorted(signers_train & signers_val),
                "duration_samples": durs,
            }
        )
    name_groups = {
        "food_related": [g for g in class_to_idx if any(x in g for x in ("EAT", "LUNCH", "BREAKFAST", "DINNER"))],
        "boy_girl_people": [g for g in class_to_idx if g in ("BOY",) or "GIRL" in g or "THEY" in g],
    }
    counts = {
        "train_videos_min_max_mean": _count_stats([len(by_gloss[g]["train"]) for g in class_to_idx]),
        "val_videos_min_max_mean": _count_stats([len(by_gloss[g]["val"]) for g in class_to_idx]),
    }
    return {"sampled_classes": details, "possible_similar_name_groups": name_groups, "split_count_stats": counts}


def _count_stats(xs):
    return {"min": min(xs), "max": max(xs), "mean": float(np.mean(xs))}


def frozen_comparison() -> dict:
    payload = {
        "frozen_best_checkpoint_still_on_disk": False,
        "reason": (
            "BEST/LAST paths were reused by the layer4 run (now ~119MB with optimizer). "
            "Previous frozen files were ~52MB. dry_run_test.pth remains but is a 2-batch dry-run, not the frozen 100-class model."
        ),
        "preserved_frozen_metrics_from_json": {},
    }
    if FROZEN_META.exists():
        payload["preserved_frozen_metrics_from_json"]["checkpoint_meta"] = json.loads(FROZEN_META.read_text(encoding="utf-8"))
    if FROZEN_EVAL.exists():
        payload["preserved_frozen_metrics_from_json"]["evaluation_test"] = json.loads(FROZEN_EVAL.read_text(encoding="utf-8"))
    if FROZEN_PRED_BEST.exists():
        payload["preserved_frozen_metrics_from_json"]["prediction_best_subset"] = json.loads(
            FROZEN_PRED_BEST.read_text(encoding="utf-8")
        )
    if FROZEN_PRED_LAST.exists():
        payload["preserved_frozen_metrics_from_json"]["prediction_last_subset"] = json.loads(
            FROZEN_PRED_LAST.read_text(encoding="utf-8")
        )
    dry = ROOT / "outputs" / "asl_citizen_100" / "checkpoints" / "dry_run_test.pth"
    if dry.exists():
        dry_payload = torch.load(dry, map_location="cpu", weights_only=False)
        payload["dry_run_test.pth"] = {
            "exists": True,
            "epoch": dry_payload.get("epoch"),
            "freeze_backbone": dry_payload.get("full_config", {}).get("freeze_backbone"),
            "backbone_train_mode": dry_payload.get("full_config", {}).get("backbone_train_mode"),
            "note": "Not a trained 100-class frozen run.",
        }
    return payload


def write_confusion_csv(per_class_rows, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_idx", "gloss", "support", "correct", "accuracy", "precision", "recall", "f1", "predicted_as_this"],
        )
        writer.writeheader()
        writer.writerows(per_class_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cfg_default = default_config()

    print("== checkpoint meta ==")
    meta = {"best": checkpoint_meta(BEST_CHECKPOINT_PATH), "last": checkpoint_meta(LAST_CHECKPOINT_PATH)}
    if HISTORY_PATH.exists():
        meta["training_history"] = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    dump("00_checkpoint_meta.json", meta)

    print("== check 4 signers ==")
    signers = check_signers(cfg_default)
    dump("04_signers.json", signers)

    print("== check 6 transforms ==")
    tinfo = inspect_transforms(cfg_default)
    dump("06_transforms.json", tinfo)

    print("== check 10 dataset difficulty ==")
    class_to_idx, idx_to_class = load_class_mappings(cfg_default.class_to_idx_path, cfg_default.idx_to_class_path)
    difficulty = dataset_difficulty(cfg_default, class_to_idx, idx_to_class)
    dump("10_dataset_difficulty.json", difficulty)

    print("== check 11 frozen comparison (json only; do not overwrite) ==")
    dump("11_frozen_comparison.json", frozen_comparison())

    print("== check 5 visual sanity ==")
    vis = visual_sanity(cfg_default, class_to_idx, n_videos=6)
    dump("05_visual_sanity.json", vis)

    print("== load best/last models ==")
    best_payload, best_cfg, c2i, i2c, best_model, best_inc = load_model(BEST_CHECKPOINT_PATH, device)
    last_payload, last_cfg, _, _, last_model, last_inc = load_model(LAST_CHECKPOINT_PATH, device)
    dump(
        "00b_load_compat.json",
        {
            "best_missing": best_inc.missing_keys,
            "best_unexpected": best_inc.unexpected_keys,
            "last_missing": last_inc.missing_keys,
            "last_unexpected": last_inc.unexpected_keys,
        },
    )

    val_ds = make_eval_dataset("val", best_cfg, c2i)
    train_eval_ds = make_eval_dataset("train", best_cfg, c2i)

    print("== check 7 gradients on one train-eval batch ==")
    grad = gradient_check(best_payload, best_cfg, train_eval_ds, device)
    dump("07_optimization_gradients.json", grad)

    print("== check 1 val inference BEST (full val) ==")
    best_val = infer_full(best_model, val_ds, device)
    best_val_sum = {
        "checkpoint": "best",
        "epoch": best_payload.get("epoch"),
        "split": "val",
        "n": best_val["n"],
        "loss": best_val["loss"],
        "top1": best_val["top1"],
        "top5": best_val["top5"],
        **summarize_predictions(best_val["preds"], best_val["labels"].tolist(), i2c),
    }
    dump("01_val_best.json", best_val_sum)

    print("== check 1 val inference LAST (full val) ==")
    last_val = infer_full(last_model, val_ds, device)
    last_val_sum = {
        "checkpoint": "last",
        "epoch": last_payload.get("epoch"),
        "split": "val",
        "n": last_val["n"],
        "loss": last_val["loss"],
        "top1": last_val["top1"],
        "top5": last_val["top5"],
        **summarize_predictions(last_val["preds"], last_val["labels"].tolist(), i2c),
    }
    dump("01_val_last.json", last_val_sum)

    print("== check 2 confusion BEST val ==")
    cm = per_class_report(best_val["preds"], best_val["labels"].tolist(), i2c)
    np.save(OUT / "02_confusion_matrix.npy", cm["confusion_matrix"])
    write_confusion_csv(cm["per_class"], OUT / "02_per_class_metrics.csv")
    cm_json = dict(cm)
    cm_json.pop("confusion_matrix")
    dump("02_per_class_summary.json", cm_json)

    print("== check 3 train vs val BEST (eval transforms, full train) ==")
    best_train = infer_full(best_model, train_eval_ds, device)
    train_sum = {
        "checkpoint": "best",
        "split": "train_eval_transforms",
        "n": best_train["n"],
        "loss": best_train["loss"],
        "top1": best_train["top1"],
        "top5": best_train["top5"],
        **summarize_predictions(best_train["preds"], best_train["labels"].tolist(), i2c),
    }
    dump("03_train_best_eval_tf.json", train_sum)
    dump(
        "03_train_vs_val.json",
        {
            "train": {"loss": best_train["loss"], "top1": best_train["top1"], "top5": best_train["top5"], "n": best_train["n"]},
            "val": {"loss": best_val["loss"], "top1": best_val["top1"], "top5": best_val["top5"], "n": best_val["n"]},
            "gap_top1": best_train["top1"] - best_val["top1"],
        },
    )

    print("== check 6 small matched-preprocess eval (train subset vs val subset) ==")
    train_subset = infer_full(best_model, train_eval_ds, device, max_samples=393)
    dump(
        "06_matched_preprocess_subset.json",
        {
            "train_first_393_eval_tf": {
                "n": train_subset["n"],
                "loss": train_subset["loss"],
                "top1": train_subset["top1"],
                "top5": train_subset["top5"],
                **summarize_predictions(train_subset["preds"], train_subset["labels"].tolist(), i2c),
            },
            "val_full_eval_tf": {"n": best_val["n"], "top1": best_val["top1"], "top5": best_val["top5"]},
            "plausible_that_crop_scale_alone_explains_failure": False,
            "reason": (
                "scale=(0.85,1.0) is mild. Independent per-frame RandomResizedCrop is a larger issue "
                "than CenterCrop vs RandomResizedCrop magnitude."
            ),
        },
    )

    print("== check 8 activations ==")
    dump("08_activations_best.json", activation_diagnostic(best_model, val_ds, device, n=8))
    dump("08_activations_last.json", activation_diagnostic(last_model, val_ds, device, n=8))

    print("== check 9 temporal probe ==")
    dump("09_temporal_best.json", temporal_probe(best_model, val_ds, device, n=48))
    dump("09_temporal_last.json", temporal_probe(last_model, val_ds, device, n=48))

    print("Done. All JSON under", OUT)


if __name__ == "__main__":
    main()
