from __future__ import annotations

import asyncio
import sys
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image
from starlette.datastructures import UploadFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import main as backend_main
from src.config import BEST_MODEL_PATH, DATASET_DIR, HAND_LANDMARKER_PATH
from src.data.dataset import sequential_split_indices
from src.data.preprocessing import eval_transforms, train_transforms
from src.inference.engine import Predictor
from src.inference.hands import HandDetector
from src.utils.checkpoint import load_checkpoint
from src.utils.smoothing import TemporalSmoother


class ProjectSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.class_names = [chr(ord("A") + index) for index in range(26)]

    def test_checkpoint_loads_with_class_mapping(self) -> None:
        payload = load_checkpoint(BEST_MODEL_PATH)
        self.assertEqual(payload["class_names"], self.class_names)
        self.assertEqual(payload["num_classes"], 26)
        self.assertEqual(tuple(payload["model_state_dict"]["fc.weight"].shape), (26, 512))

    def test_predictor_returns_one_image_prediction(self) -> None:
        predictor = Predictor(BEST_MODEL_PATH)
        result = predictor.predict_path(ROOT / "dataset" / "asl_alphabet_test" / "A_test.jpg")
        self.assertEqual(result["prediction"], "A")
        self.assertGreater(result["confidence"], 0.90)

    def test_dataset_class_mapping_is_stable(self) -> None:
        from torchvision import datasets

        dataset = datasets.ImageFolder(str(DATASET_DIR))
        self.assertEqual(dataset.classes, self.class_names)
        self.assertEqual(dataset.class_to_idx, {name: index for index, name in enumerate(self.class_names)})
        self.assertEqual(len(dataset.samples), 78000)
        train_idx, val_idx, test_idx = sequential_split_indices(dataset.samples)
        self.assertEqual(len(train_idx), 54600)
        self.assertEqual(len(val_idx), 11700)
        self.assertEqual(len(test_idx), 11700)

    def test_preprocessing_has_no_horizontal_flip(self) -> None:
        train_names = [step.__class__.__name__ for step in train_transforms().transforms]
        eval_names = [step.__class__.__name__ for step in eval_transforms().transforms]
        self.assertNotIn("RandomHorizontalFlip", train_names)
        self.assertNotIn("RandomHorizontalFlip", eval_names)
        self.assertIn("Normalize", train_names)
        self.assertIn("Normalize", eval_names)

    def test_temporal_smoother_resets_on_low_confidence(self) -> None:
        smoother = TemporalSmoother(window=5, threshold=0.70)
        label, _, status = smoother.update("A", 0.95)
        self.assertEqual(label, "A")
        self.assertEqual(status, "Updating")
        none_label, _, low_status = smoother.update("B", 0.10)
        self.assertIsNone(none_label)
        self.assertEqual(low_status, "Low confidence")
        self.assertEqual(len(smoother._labels), 0)

    def test_backend_health_and_prediction(self) -> None:
        backend_main.startup()
        health = backend_main.health()
        self.assertTrue(health["model_loaded"])
        self.assertEqual(health["classes"], self.class_names)
        self.assertTrue(health["hand_detection_available"], health.get("hand_detection_error"))

        payload = (ROOT / "dataset" / "asl_alphabet_test" / "A_test.jpg").read_bytes()
        upload = UploadFile(filename="A_test.jpg", file=BytesIO(payload))
        response = asyncio.run(backend_main.predict(upload))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"prediction":"A"', response.body)

        empty = UploadFile(filename="empty.jpg", file=BytesIO(b""))
        with self.assertRaises(Exception):
            asyncio.run(backend_main.predict(empty))

        invalid = UploadFile(filename="bad.txt", file=BytesIO(b"not-an-image"))
        with self.assertRaises(Exception):
            asyncio.run(backend_main.predict(invalid))

    def test_no_hand_state_on_blank_image(self) -> None:
        self.assertTrue(HAND_LANDMARKER_PATH.exists())
        detector = HandDetector(static_image_mode=True)
        self.addCleanup(detector.close)
        self.assertTrue(detector.available, detector.error)
        blank = Image.new("RGB", (224, 224), (255, 255, 255))
        crop = detector.crop_pil(blank)
        self.assertFalse(crop.detected)

        backend_main.startup()
        buffer = BytesIO()
        blank.save(buffer, format="JPEG")
        upload = UploadFile(filename="blank.jpg", file=BytesIO(buffer.getvalue()))
        response = asyncio.run(backend_main.predict(upload, require_hand=True))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"status":"no_hand_detected"', response.body)
        self.assertIn(b'"prediction":null', response.body)


if __name__ == "__main__":
    unittest.main()
