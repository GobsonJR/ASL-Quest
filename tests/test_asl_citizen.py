"""Tests for the isolated ASL Citizen 100-class pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asl_citizen.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
from src.asl_citizen.config import (
    ASL_CITIZEN_ROOT,
    ASL_CITIZEN_VIDEOS_DIR,
    NUM_CLASSES,
    NUM_FRAMES,
    default_config,
)
from src.asl_citizen.dataset import ASLCitizenDataset, build_dataloader
from src.asl_citizen.preprocessing import eval_spatial_transforms
from src.asl_citizen.manifest import build_manifests, verify_signer_leakage
from src.asl_citizen.model import ASLCitizenResNet18GRU, build_model, parameter_counts
from src.asl_citizen.preprocessing import uniform_temporal_indices
from src.asl_citizen.utils import load_class_mappings, load_manifest_rows, resolve_video_path


class ASLCitizenPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not ASL_CITIZEN_ROOT.exists():
            raise unittest.SkipTest(f"ASL Citizen dataset not found at {ASL_CITIZEN_ROOT}")
        cls.cfg = default_config()
        cls.report = build_manifests(cfg=cls.cfg)
        cls.class_to_idx, cls.idx_to_class = load_class_mappings(
            cls.cfg.class_to_idx_path,
            cls.cfg.idx_to_class_path,
        )

    def test_manifest_generation_counts(self) -> None:
        self.assertEqual(self.report["num_classes"], NUM_CLASSES)
        counts = self.report["split_counts"]
        self.assertGreater(counts["train"], 0)
        self.assertGreater(counts["val"], 0)
        self.assertGreater(counts["test"], 0)
        self.assertEqual(counts["total"], counts["train"] + counts["val"] + counts["test"])

    def test_class_mapping_consistency(self) -> None:
        self.assertEqual(len(self.class_to_idx), NUM_CLASSES)
        self.assertEqual(len(self.idx_to_class), NUM_CLASSES)
        for gloss, idx in self.class_to_idx.items():
            self.assertEqual(self.idx_to_class[str(idx)], gloss)
        indices = sorted(int(v) for v in self.class_to_idx.values())
        self.assertEqual(indices, list(range(NUM_CLASSES)))

    def test_no_class_missing_from_split(self) -> None:
        missing = self.report["class_balance"]["classes_missing_any_split"]
        self.assertEqual(missing, [])

    def test_signer_split_integrity(self) -> None:
        leakage = self.report["leakage_check"]
        self.assertTrue(leakage["signer_independent"])
        self.assertEqual(leakage["train_val_overlap"], [])
        self.assertEqual(leakage["train_test_overlap"], [])
        self.assertEqual(leakage["val_test_overlap"], [])
        self.assertEqual(leakage["cross_split_duplicate_filenames"], [])

    def test_video_path_resolution(self) -> None:
        rows = load_manifest_rows(self.cfg.train_manifest)
        path = resolve_video_path(ASL_CITIZEN_VIDEOS_DIR, rows[0]["video_file"])
        self.assertTrue(path.is_file())

    def test_dataset_length(self) -> None:
        train_rows = load_manifest_rows(self.cfg.train_manifest)
        dataset = ASLCitizenDataset(
            train_rows,
            video_root=ASL_CITIZEN_VIDEOS_DIR,
            class_to_idx=self.class_to_idx,
            num_frames=NUM_FRAMES,
            training=False,
        )
        self.assertEqual(len(dataset), len(train_rows))

    def test_one_sample_loading_and_shape(self) -> None:
        train_rows = load_manifest_rows(self.cfg.train_manifest)
        transform = eval_spatial_transforms(
            self.cfg.image_size,
            self.cfg.imagenet_mean,
            self.cfg.imagenet_std,
        )
        dataset = ASLCitizenDataset(
            train_rows[:1],
            video_root=ASL_CITIZEN_VIDEOS_DIR,
            class_to_idx=self.class_to_idx,
            num_frames=NUM_FRAMES,
            transform=transform,
            training=False,
        )
        sample = dataset[0]
        self.assertEqual(sample["video"].shape, (NUM_FRAMES, 3, 224, 224))
        self.assertGreaterEqual(sample["label"], 0)
        self.assertLess(sample["label"], NUM_CLASSES)

    def test_batch_shape(self) -> None:
        loader, _ = build_dataloader("train", self.cfg, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(batch["video"].shape[0], 2)
        self.assertEqual(batch["video"].shape[1], NUM_FRAMES)
        self.assertEqual(batch["video"].shape[2], 3)
        self.assertEqual(batch["video"].shape[3], 224)
        self.assertEqual(batch["video"].shape[4], 224)

    def test_model_forward_pass(self) -> None:
        model = build_model(self.cfg, pretrained=False)
        x = torch.randn(2, NUM_FRAMES, 3, 224, 224)
        logits = model(x)
        self.assertEqual(logits.shape, (2, NUM_CLASSES))

    def test_backward_pass(self) -> None:
        model = build_model(self.cfg, pretrained=False)
        x = torch.randn(1, NUM_FRAMES, 3, 224, 224)
        labels = torch.tensor([0])
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))

    def test_checkpoint_save_load(self) -> None:
        model = build_model(self.cfg, pretrained=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pth"
            payload = build_checkpoint(
                model,
                self.class_to_idx,
                self.idx_to_class,
                self.cfg,
                epoch=1,
                val_accuracy=0.5,
                val_top5=0.8,
            )
            save_checkpoint(payload, path)
            loaded, cfg, c2i, i2c = load_checkpoint(path, map_location="cpu")
            self.assertEqual(loaded["num_classes"], NUM_CLASSES)
            self.assertEqual(loaded["num_frames"], NUM_FRAMES)
            self.assertEqual(c2i, self.class_to_idx)
            self.assertEqual(i2c, self.idx_to_class)
            self.assertEqual(cfg.num_classes, NUM_CLASSES)

    def test_uniform_sampling_short_video(self) -> None:
        indices = uniform_temporal_indices(5, 16)
        self.assertEqual(len(indices), 16)
        self.assertTrue(all(0 <= i < 5 for i in indices))

    def test_leakage_check_function(self) -> None:
        rows = (
            load_manifest_rows(self.cfg.train_manifest)
            + load_manifest_rows(self.cfg.val_manifest)
            + load_manifest_rows(self.cfg.test_manifest)
        )
        result = verify_signer_leakage(rows)
        self.assertTrue(result["signer_independent"])

    def test_manifest_files_exist(self) -> None:
        for path in (
            self.cfg.data_dir / "classes.csv",
            self.cfg.train_manifest,
            self.cfg.val_manifest,
            self.cfg.test_manifest,
            self.cfg.class_to_idx_path,
            self.cfg.idx_to_class_path,
        ):
            self.assertTrue(path.exists(), f"Missing {path}")

    def test_class_mapping_json_sorted_deterministic(self) -> None:
        raw = json.loads(self.cfg.class_to_idx_path.read_text(encoding="utf-8"))
        glosses = list(raw.keys())
        self.assertEqual(glosses, sorted(glosses))

    def test_backbone_modes_trainability(self) -> None:
        frozen = build_model(default_config(backbone_train_mode="frozen"), pretrained=False)
        layer4 = build_model(default_config(backbone_train_mode="layer4"), pretrained=False)
        full = build_model(default_config(backbone_train_mode="full"), pretrained=False)
        self.assertFalse(any(p.requires_grad for p in frozen.encoder.parameters()))
        self.assertTrue(any(p.requires_grad for p in frozen.gru.parameters()))
        self.assertFalse(any(p.requires_grad for p in layer4.encoder.layer3.parameters()))
        self.assertTrue(any(p.requires_grad for p in layer4.encoder.layer4.parameters()))
        self.assertTrue(all(p.requires_grad for p in full.encoder.parameters()))

    def test_tiny_classifier_has_ten_outputs(self) -> None:
        model = ASLCitizenResNet18GRU(num_classes=10, pretrained=False, backbone_train_mode="layer4")
        logits = model(torch.randn(2, NUM_FRAMES, 3, 224, 224))
        self.assertEqual(logits.shape, (2, 10))

    def test_training_history_appends_each_epoch(self) -> None:
        from src.asl_citizen.train import persist_training_history

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_history.json"
            persist_training_history(path, {"epoch": 1, "train_accuracy": 0.1})
            persist_training_history(path, {"epoch": 2, "train_accuracy": 0.2})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["epochs"]), 2)
            self.assertEqual(payload["epochs"][1]["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
