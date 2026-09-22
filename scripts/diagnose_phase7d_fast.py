"""Fast Phase 7D diagnostics — saves incremental JSON, small subsets only."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asl_citizen.checkpoint import load_checkpoint
from src.asl_citizen.config import BEST_CHECKPOINT_PATH, LAST_CHECKPOINT_PATH, default_config
from src.asl_citizen.dataset import ASLCitizenDataset, build_dataloader, decode_sampled_frames, get_frame_count
from src.asl_citizen.metrics import batch_accuracy, top_k_accuracy
from src.asl_citizen.model import ASLCitizenResNet18GRU, build_model, parameter_counts
from src.asl_citizen.preprocessing import eval_spatial_transforms, train_spatial_transforms, uniform_temporal_indices
from src.asl_citizen.utils import load_class_mappings, load_manifest_rows, resolve_video_path

OUT = ROOT / "outputs" / "asl_citizen_100" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def dump(name: str, data: dict) -> None:
    (OUT / name).write_text(json.dumps(data, indent=2), encoding="utf-8")


@torch.no_grad()
def eval_n(path: Path, split: str, n: int = 200) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, cfg, _, i2c = load_checkpoint(path, map_location=device)
    model = build_model(cfg, pretrained=False)
    inc = model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    loader, ds = build_dataloader(split, cfg, batch_size=4, shuffle=False)
    idxs = list(range(min(n, len(ds))))
    sub = DataLoader(Subset(ds, idxs), batch_size=4, shuffle=False, num_workers=0)
    logits_all, labels_all, preds = [], [], []
    for batch in sub:
        logits = model(batch["video"].to(device))
        logits_all.append(logits.cpu())
        labels_all.append(batch["label"])
        preds.extend(logits.argmax(1).cpu().tolist())
    lc = torch.cat(logits_all)
    ll = torch.cat(labels_all)
    pc = Counter(preds)
    return {
        "checkpoint": path.name,
        "split": split,
        "samples": len(idxs),
        "strict_ok": not inc.missing_keys and not inc.unexpected_keys,
        "missing_keys": inc.missing_keys,
        "unexpected_keys": inc.unexpected_keys,
        "top1": batch_accuracy(lc, ll),
        "top5": top_k_accuracy(lc, ll, 5),
        "unique_predicted_classes": len(pc),
        "most_frequent": {"idx": pc.most_common(1)[0][0], "gloss": i2c[str(pc.most_common(1)[0][0])], "count": pc.most_common(1)[0][1]},
        "top10": [{"idx": i, "gloss": i2c[str(i)], "count": c} for i, c in pc.most_common(10)],
        "distribution": {str(k): v for k, v in sorted(pc.items(), key=lambda x: (-x[1], x[0]))},
    }


def tiny_overfit(freeze: bool, epochs: int = 12) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = default_config(freeze_backbone=freeze)
    c2i, _ = load_class_mappings()
    rows = load_manifest_rows(cfg.train_manifest)
    by_g = {}
    for r in rows:
        by_g.setdefault(r["gloss"], []).append(r)
    glosses = sorted(by_g)[:10]
    tiny = []
    for g in glosses:
        tiny.extend(by_g[g][:5])
    tiny_c2i = {g: c2i[g] for g in glosses}
    remap = {c2i[g]: i for i, g in enumerate(glosses)}
    remapped = []
    for r in tiny:
        rr = dict(r)
        rr["class_idx"] = remap[c2i[r["gloss"]]]
        remapped.append(rr)
    ds = ASLCitizenDataset(
        remapped,
        video_root=cfg.video_root,
        class_to_idx={g: remap[c2i[g]] for g in glosses},
        num_frames=cfg.num_frames,
        transform=train_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std),
        training=True,
        jitter_frames=0,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    model = ASLCitizenResNet18GRU(num_classes=10, hidden_size=256, dropout=0.0, freeze_backbone=freeze, pretrained=True).to(device)
    opt = AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        correct = total = 0
        loss_sum = 0.0
        for batch in loader:
            x, y = batch["video"].to(device), batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        hist.append({"epoch": ep, "loss": loss_sum / total, "acc": correct / total})
    return {"freeze_backbone": freeze, "epochs": epochs, "samples": len(remapped), "history": hist, "final_acc": hist[-1]["acc"]}


def main() -> None:
    cfg = default_config()
    hist_path = ROOT / "outputs" / "asl_citizen_100" / "metrics" / "training_history.json"

    ckpt_meta = {}
    for tag, p in [("best", BEST_CHECKPOINT_PATH), ("last", LAST_CHECKPOINT_PATH)]:
        pl = torch.load(p, map_location="cpu", weights_only=False)
        ckpt_meta[tag] = {
            "epoch": pl.get("epoch"),
            "val_accuracy": pl.get("val_accuracy"),
            "val_top5_accuracy": pl.get("val_top5_accuracy"),
            "full_config": pl.get("full_config"),
        }
    dump("01_checkpoint_meta.json", {
        "training_history_exists": hist_path.exists(),
        "missing_metrics_note": "train.py writes training_history.json only after ALL epochs finish. Per-epoch train_loss/train_acc are NOT stored in checkpoints.",
        "checkpoints": ckpt_meta,
    })

    dump("02_prediction_best.json", eval_n(BEST_CHECKPOINT_PATH, "test", 250))
    dump("03_prediction_last.json", eval_n(LAST_CHECKPOINT_PATH, "test", 250))

    c2i, i2c = load_class_mappings()
    dump("04_label_mapping.json", {
        "num_classes": len(c2i),
        "indices_0_99": sorted(int(v) for v in c2i.values()) == list(range(100)),
        "inverse_ok": all(i2c[str(v)] == k for k, v in c2i.items()),
        "ckpt_matches_file": load_checkpoint(BEST_CHECKPOINT_PATH)[2] == {k: int(v) for k, v in load_checkpoint(BEST_CHECKPOINT_PATH)[2].items()},
    })

    # label examples with model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload, cfg2, _, i2c = load_checkpoint(BEST_CHECKPOINT_PATH, map_location=device)
    model = build_model(cfg2, pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    test_rows = load_manifest_rows(cfg.test_manifest)
    examples = []
    for row in random.Random(42).sample(test_rows, 8):
        ds = ASLCitizenDataset([row], video_root=cfg.video_root, class_to_idx=c2i,
            num_frames=cfg.num_frames, transform=eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std), training=False)
        item = ds[0]
        pred = model(item["video"].unsqueeze(0).to(device)).argmax(1).item()
        examples.append({
            "video_file": row["video_file"], "manifest_gloss": row["gloss"], "manifest_class_idx": int(row["class_idx"]),
            "dataset_label": int(item["label"]), "pred_gloss": i2c[str(pred)], "pred_idx": int(pred),
            "mapping_ok": int(row["class_idx"]) == c2i[row["gloss"]] == int(item["label"]),
        })
    dump("05_label_examples.json", {"examples": examples, "all_mapping_ok": all(e["mapping_ok"] for e in examples)})

    paths = []
    for row in random.Random(7).sample(test_rows, 20):
        p = cfg.video_root / row["video_file"]
        paths.append({"csv": row["video_file"], "resolved": str(p.resolve()), "exists": p.is_file(), "size": p.stat().st_size if p.is_file() else None})
    dump("06_video_paths.json", {"all_exist": all(x["exists"] for x in paths), "entries": paths})

    temporal = []
    for row in random.Random(11).sample(test_rows, 6):
        p = resolve_video_path(cfg.video_root, row["video_file"])
        cap = cv2.VideoCapture(str(p)); fc = get_frame_count(cap); cap.release()
        idx = uniform_temporal_indices(fc, cfg.num_frames)
        temporal.append({"video": row["video_file"], "frames": fc, "indices": idx, "chronological": idx == sorted(idx)})
    dump("07_temporal_sampling.json", {"examples": temporal})

    row = test_rows[0]
    p = resolve_video_path(cfg.video_root, row["video_file"])
    cap = cv2.VideoCapture(str(p)); fc = get_frame_count(cap); cap.release()
    bgr = decode_sampled_frames(p, uniform_temporal_indices(fc, 1))[0]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    ev = eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std)(Image.fromarray(rgb))
    dump("08_normalization.json", {
        "bgr_shape": list(bgr.shape),
        "raw_min": float(raw.min()), "raw_max": float(raw.max()),
        "eval_min": float(ev.min()), "eval_max": float(ev.max()), "eval_mean": float(ev.mean()),
    })

    dump("09_preprocessing.json", {
        "train": [t.__class__.__name__ for t in train_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std).transforms],
        "eval": [t.__class__.__name__ for t in eval_spatial_transforms(cfg.image_size, cfg.imagenet_mean, cfg.imagenet_std).transforms],
    })

    m = build_model(cfg, pretrained=False)
    x = torch.randn(2, 16, 3, 224, 224)
    with torch.no_grad():
        logits = m(x)
    dump("10_forward_pass.json", {"logits_shape": list(logits.shape), "params": parameter_counts(m)})

    loader, _ = build_dataloader("train", cfg, batch_size=8, shuffle=True)
    b = next(iter(loader))
    dump("11_train_batch.json", {
        "shape": list(b["video"].shape),
        "unique_labels": len(set(b["label"].tolist())),
        "unique_files": len(set(b["video_file"])),
        "label_freq": dict(Counter(b["label"].tolist())),
        "tensor_min": float(b["video"].min()), "tensor_max": float(b["video"].max()), "tensor_mean": float(b["video"].mean()),
    })

    print("tiny overfit frozen...")
    dump("12_overfit_frozen.json", tiny_overfit(True, 12))
    print("tiny overfit unfrozen...")
    dump("13_overfit_unfrozen.json", tiny_overfit(False, 8))

    print("DONE", OUT)


if __name__ == "__main__":
    main()
