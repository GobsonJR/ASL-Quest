"""Tests for the development-only live-request capture (backend/services/live_diagnostics.py)."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from starlette.datastructures import UploadFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.routers import ml  # noqa: E402
from backend.services.live_diagnostics import LiveDiagnostics, tensor_to_image  # noqa: E402
from src.config import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.data.preprocessing import eval_transforms  # noqa: E402
from src.inference.hands import HandCrop  # noqa: E402

LETTERS = [chr(65 + i) for i in range(26)]


class StubPredictor:
    """Just enough of Predictor for the capture code, with a real (tiny) file standing in for the checkpoint."""

    class_names = LETTERS
    device = "cpu"

    def __init__(self, checkpoint: Path, label: str = "W", confidence: float = 0.23) -> None:
        self.model_path = checkpoint
        self.transform = eval_transforms()
        self.label, self.confidence = label, confidence

    def predict_pil(self, image, top_k: int = 3):
        others = [c for c in LETTERS if c != self.label][: top_k - 1]
        top = [{"label": self.label, "confidence": self.confidence}] + [
            {"label": c, "confidence": round((1 - self.confidence) / 25, 3)} for c in others
        ]
        return {
            "prediction": self.label,
            "confidence": self.confidence,
            "top_predictions": top,
            "low_confidence": self.confidence < 0.7,
            "diagnostics": {"tensor_shape": [1, 3, 224, 224]},
        }


class StubDetector:
    available = True
    error = None

    def crop_pil(self, image, padding: float = 0.30) -> HandCrop:
        window = (20, 10, 140, 130)
        return HandCrop(image=image.crop(window), box=window, detected=True, hand_box=(30.0, 20.0, 130.0, 120.0))


def jpeg_payload(size=(200, 160)) -> bytes:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=72)
    return buffer.getvalue()


class CaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "live"
        self.checkpoint = Path(self.tmp.name) / "model.pth"
        self.checkpoint.write_bytes(b"not a real checkpoint, only hashed")
        self.predictor = StubPredictor(self.checkpoint)

    def save(self, diag: LiveDiagnostics, payload: bytes | None = None, **kw):
        payload = payload or jpeg_payload()
        frame = Image.open(BytesIO(payload))
        frame.load()
        crop = frame.crop((20, 10, 140, 130))
        return diag.save(
            payload=payload,
            frame=frame,
            source_image=crop,
            hand_box=(30.0, 20.0, 130.0, 120.0),
            crop_window=(20, 10, 140, 130),
            predictor=self.predictor,
            result=self.predictor.predict_pil(crop, top_k=5),
            meta={"target": "C", "preview_mirrored": False, "video_size": "960x720"},
            **kw,
        )

    def test_disabled_by_default_writes_nothing(self) -> None:
        diag = LiveDiagnostics(self.root, enabled=False)
        self.assertIsNone(self.save(diag))
        self.assertFalse(self.root.exists())

    def test_one_bundle_holds_every_artifact_for_the_same_request(self) -> None:
        diag = LiveDiagnostics(self.root, enabled=True)
        payload = jpeg_payload()
        saved = self.save(diag, payload)
        folder = Path(saved["saved_to"])
        self.assertEqual(folder.name, saved["request_id"])
        self.assertEqual(
            sorted(p.name for p in folder.iterdir()),
            ["1_original_frame.jpg", "2_hand_box_overlay.png", "3_classifier_crop.png", "4_model_input_224.png", "result.json"],
        )
        # the original is byte-for-byte what the server received
        self.assertEqual((folder / "1_original_frame.jpg").read_bytes(), payload)
        self.assertEqual(Image.open(folder / "3_classifier_crop.png").size, (120, 120))
        self.assertEqual(Image.open(folder / "4_model_input_224.png").size, (224, 224))

        record = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(record["request_id"], saved["request_id"])
        self.assertEqual(record["frame_size"], [200, 160])
        self.assertEqual(record["classifier_crop_size"], [120, 120])
        self.assertEqual(record["model_input_shape"], [3, 224, 224])
        self.assertEqual(len(record["top_predictions"]), 5)
        self.assertEqual(record["prediction"], "W")
        self.assertEqual(record["client"], {"target": "C", "preview_mirrored": False, "video_size": "960x720"})
        self.assertTrue(record["class_order_is_a_to_z"])
        self.assertEqual(len(record["checkpoint_sha256"]), 64)
        self.assertTrue(record["checkpoint"].endswith("model.pth"))

    def test_model_input_image_is_the_normalised_tensor_converted_back(self) -> None:
        colour = (200, 80, 30)
        tensor = eval_transforms()(Image.new("RGB", (90, 60), colour))
        back = np.array(tensor_to_image(tensor, IMAGENET_MEAN, IMAGENET_STD))
        self.assertEqual(back.shape, (224, 224, 3))
        self.assertLessEqual(int(np.abs(back[100, 100].astype(int) - np.array(colour)).max()), 1)

    def test_rate_limit_and_force(self) -> None:
        diag = LiveDiagnostics(self.root, enabled=True, min_interval_s=60)
        self.assertIsNotNone(self.save(diag))
        self.assertIsNone(self.save(diag))  # too soon
        self.assertIsNotNone(self.save(diag, force=True))

    def test_only_the_newest_bundles_are_kept(self) -> None:
        diag = LiveDiagnostics(self.root, enabled=True, keep=3, min_interval_s=0)
        for _ in range(6):
            self.save(diag)
        self.assertEqual(len([p for p in self.root.iterdir() if p.is_dir()]), 3)


class EndpointCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        checkpoint = Path(self.tmp.name) / "m.pth"
        checkpoint.write_bytes(b"x")
        ml.set_runtime(StubPredictor(checkpoint), StubDetector(), None)
        self.original_diag = ml.live_diag

    def tearDown(self) -> None:
        ml.live_diag = self.original_diag
        ml.set_runtime(None, None, None)

    def call(self, **kwargs) -> dict:
        upload = UploadFile(filename="practice.jpg", file=BytesIO(jpeg_payload((320, 240))))
        return json.loads(asyncio.run(ml.predict(upload, **kwargs)).body)

    def test_capture_is_reported_off_when_not_enabled(self) -> None:
        ml.live_diag = LiveDiagnostics(Path(self.tmp.name) / "live", enabled=False)
        body = self.call(require_hand=True, debug=True, save_diagnostics=True, target="C")
        self.assertFalse(body["diagnostics"]["capture_enabled"])
        self.assertNotIn("request_id", body["diagnostics"])
        self.assertFalse((Path(self.tmp.name) / "live").exists())

    def test_enabled_capture_saves_the_request_that_produced_the_response(self) -> None:
        ml.live_diag = LiveDiagnostics(Path(self.tmp.name) / "live", enabled=True, min_interval_s=0)
        body = self.call(require_hand=True, debug=True, save_diagnostics=True, target="C", preview_mirrored=False, video_size="960x720")
        folder = Path(body["diagnostics"]["saved_to"])
        record = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(record["prediction"], body["prediction"])  # same request -> same answer
        self.assertEqual(record["landmark_box"], [30.0, 20.0, 130.0, 120.0])
        self.assertEqual(record["crop_window_clipped"], list(body["hand_box"]))
        self.assertEqual(record["client"]["target"], "C")
        self.assertEqual(len(record["top_predictions"]), 5)  # capture always records the top five
        self.assertEqual(Image.open(folder / "3_classifier_crop.png").size, (120, 120))

    def test_nothing_is_saved_unless_the_request_asks_for_it(self) -> None:
        ml.live_diag = LiveDiagnostics(Path(self.tmp.name) / "live", enabled=True, min_interval_s=0)
        self.call(require_hand=True, debug=True)
        self.assertFalse((Path(self.tmp.name) / "live").exists())


if __name__ == "__main__":
    unittest.main()
