"""Tests for the live webcam pipeline: crop geometry, preprocessing, label mapping, and the
/predict state separation (no hand vs low confidence vs error).

These run without a webcam: frames are synthetic arrays or dataset images, and the endpoint tests
use small fakes for the predictor / hand detector so each state can be forced deterministically.
They do NOT prove live webcam recognition quality -- that needs a physical camera.
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.routers import ml  # noqa: E402
from src.config import BEST_MODEL_PATH, CONFIDENCE_THRESHOLD, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.data.preprocessing import eval_transforms  # noqa: E402
from src.inference.hands import HandCrop, HandDetector, MIN_CROP_SIDE, crop_square, square_window  # noqa: E402

LETTERS = [chr(ord("A") + i) for i in range(26)]


def jpeg_bytes(size=(320, 240), color=(120, 120, 120)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def upload(payload: bytes | None = None, name: str = "frame.jpg") -> UploadFile:
    return UploadFile(filename=name, file=BytesIO(jpeg_bytes() if payload is None else payload))


def detector_with_landmarks(points):
    """A real HandDetector (no MediaPipe needed) whose landmark source is replaced by `points`."""
    detector = HandDetector.__new__(HandDetector)
    detector._legacy_hands = None
    detector._landmarker = None
    detector.error = None
    detector._landmarks = lambda rgb: points  # type: ignore[method-assign]
    return detector


class SquareCropTests(unittest.TestCase):
    def test_window_is_square_and_centred_for_a_wide_box(self) -> None:
        left, top, right, bottom = square_window((100, 100, 300, 160), padding=0.30)
        self.assertEqual(right - left, bottom - top)
        self.assertAlmostEqual((left + right) / 2, 200, delta=1)
        self.assertAlmostEqual((top + bottom) / 2, 130, delta=1)
        # larger side 200 * (1 + 2 * 0.30) = 320
        self.assertAlmostEqual(right - left, 320, delta=2)

    def test_window_is_square_for_a_tall_box(self) -> None:
        left, top, right, bottom = square_window((10, 10, 70, 210), padding=0.0)
        self.assertEqual(right - left, bottom - top)
        self.assertAlmostEqual(bottom - top, 200, delta=2)

    def test_degenerate_box_gets_a_minimum_size(self) -> None:
        left, top, right, bottom = square_window((50, 50, 50, 50), padding=0.3)
        self.assertGreaterEqual(right - left, MIN_CROP_SIDE)

    def test_crop_outside_frame_is_edge_filled_not_black(self) -> None:
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        image = crop_square(frame, (-40, -40, 60, 60))
        self.assertEqual(image.size, (100, 100))
        self.assertGreater(np.array(image).min(), 150)  # no black padding bars

    def test_crop_rgb_returns_square_image_and_clipped_box(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # a wide hand near the right edge: x 0.6-0.98, y 0.4-0.55
        points = [(0.6, 0.4), (0.98, 0.55), (0.8, 0.45)]
        crop = detector_with_landmarks(points).crop_rgb(frame)
        self.assertTrue(crop.detected)
        self.assertEqual(crop.image.width, crop.image.height)  # aspect preserved (square) -> no stretching later
        x1, y1, x2, y2 = crop.box
        self.assertTrue(0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480)

    def test_crop_rgb_reports_no_hand_without_landmarks(self) -> None:
        crop = detector_with_landmarks(None).crop_rgb(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertFalse(crop.detected)
        self.assertIsNone(crop.box)

    def test_hand_entirely_outside_the_frame_is_not_a_detection(self) -> None:
        crop = detector_with_landmarks([(3.0, 3.0), (3.1, 3.1)]).crop_rgb(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertFalse(crop.detected)


class PreprocessingTests(unittest.TestCase):
    def test_eval_transform_shape_dtype_and_normalisation(self) -> None:
        # solid colour, deliberately non-square, to confirm size handling and ImageNet normalisation
        tensor = eval_transforms()(Image.new("RGB", (400, 250), (255, 0, 128)))
        self.assertEqual(tuple(tensor.shape), (3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(tensor.dtype, torch.float32)
        expected = [(v / 255 - m) / s for v, m, s in zip((255, 0, 128), IMAGENET_MEAN, IMAGENET_STD)]
        for channel, value in enumerate(expected):
            self.assertAlmostEqual(float(tensor[channel, 10, 10]), value, places=3)

    def test_rgb_channel_order_is_preserved(self) -> None:
        # A pure-red image must have a large positive value in channel 0 (R), not channel 2 (B).
        tensor = eval_transforms()(Image.new("RGB", (64, 64), (255, 0, 0)))
        self.assertGreater(float(tensor[0].mean()), 1.0)
        self.assertLess(float(tensor[2].mean()), 0.0)


@unittest.skipUnless(BEST_MODEL_PATH.exists(), "trained model not present")
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.inference.engine import Predictor

        cls.predictor = Predictor(BEST_MODEL_PATH)

    def test_output_index_maps_to_the_matching_letter(self) -> None:
        self.assertEqual(self.predictor.class_names, LETTERS)
        image = Image.open(ROOT / "dataset" / "asl_alphabet_test" / "A_test.jpg")
        result = self.predictor.predict_pil(image)
        self.assertEqual(LETTERS[result["diagnostics"]["class_index"]], result["prediction"])
        self.assertEqual(result["prediction"], "A")

    def test_diagnostics_describe_the_real_input_tensor(self) -> None:
        result = self.predictor.predict_pil(Image.new("RGB", (300, 200), (90, 90, 90)))
        d = result["diagnostics"]
        self.assertEqual(d["tensor_shape"], [1, 3, IMAGE_SIZE, IMAGE_SIZE])
        self.assertEqual(d["tensor_dtype"], "float32")
        self.assertGreaterEqual(d["inference_ms"], 0)
        self.assertEqual(d["input_size"], [300, 200])


class FakePredictor:
    model_path = Path("fake.pth")
    class_names = LETTERS

    def __init__(self, label: str = "B", confidence: float = 0.95) -> None:
        self.label, self.confidence = label, confidence

    def predict_pil(self, image, top_k: int = 3):
        return {
            "prediction": self.label,
            "confidence": self.confidence,
            "top_predictions": [{"label": self.label, "confidence": self.confidence}],
            "low_confidence": self.confidence < CONFIDENCE_THRESHOLD,
            "diagnostics": {"tensor_shape": [1, 3, 224, 224], "class_index": LETTERS.index(self.label)},
        }


class FakeDetector:
    def __init__(self, detected: bool, available: bool = True) -> None:
        self.detected, self.available, self.error = detected, available, None if available else "no model"

    def crop_pil(self, image, padding: float = 0.30) -> HandCrop:
        box = (10, 10, 110, 110) if self.detected else None
        return HandCrop(image=image.crop((10, 10, 110, 110)) if self.detected else image, box=box, detected=self.detected)


class PredictStateTests(unittest.TestCase):
    def tearDown(self) -> None:
        ml.set_runtime(None, None, None)

    def call(self, **kwargs) -> dict:
        response = asyncio.run(ml.predict(upload(), **kwargs))
        return json.loads(response.body)

    def test_no_hand_is_reported_separately_from_a_prediction(self) -> None:
        ml.set_runtime(FakePredictor(), FakeDetector(detected=False), None)
        body = self.call(require_hand=True)
        self.assertEqual(body["status"], "no_hand_detected")
        self.assertIsNone(body["prediction"])
        self.assertFalse(body["hand_detected"])

    def test_low_confidence_still_means_a_hand_was_found(self) -> None:
        ml.set_runtime(FakePredictor("S", 0.31), FakeDetector(detected=True), None)
        body = self.call(require_hand=True)
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["hand_detected"])
        self.assertTrue(body["low_confidence"])
        self.assertEqual(body["prediction"], "S")  # best candidate is kept for diagnostics
        self.assertEqual(body["threshold"], CONFIDENCE_THRESHOLD)

    def test_confident_prediction(self) -> None:
        ml.set_runtime(FakePredictor("C", 0.97), FakeDetector(detected=True), None)
        body = self.call(require_hand=True)
        self.assertEqual((body["status"], body["prediction"], body["low_confidence"]), ("ok", "C", False))

    def test_detector_unavailable_is_not_reported_as_no_hand(self) -> None:
        ml.set_runtime(FakePredictor(), FakeDetector(detected=False, available=False), None)
        body = self.call(require_hand=True)
        self.assertEqual(body["status"], "hand_detector_unavailable")
        self.assertFalse(body["hand_detection_available"])

    def test_model_not_loaded_returns_503_with_reason(self) -> None:
        ml.set_runtime(None, None, "checkpoint missing")
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(ml.predict(upload()))
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("checkpoint missing", ctx.exception.detail)

    def test_invalid_and_empty_uploads_are_400(self) -> None:
        ml.set_runtime(FakePredictor(), FakeDetector(detected=True), None)
        for payload in (b"not-an-image", b""):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(ml.predict(upload(payload)))
            self.assertEqual(ctx.exception.status_code, 400)

    def test_diagnostics_only_when_requested(self) -> None:
        ml.set_runtime(FakePredictor(), FakeDetector(detected=True), None)
        self.assertNotIn("diagnostics", self.call(require_hand=True))
        debug = self.call(require_hand=True, debug=True)
        self.assertEqual(debug["diagnostics"]["tensor_shape"], [1, 3, 224, 224])
        self.assertIn("hand_detection_confidence", debug["diagnostics"])
        self.assertEqual(debug["frame_size"], [320, 240])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------------------------
# Letter-C regression tests (real checkpoint + real held-out images).
# Held-out = numeric filenames 2551-3000 of each class (sequential split, outputs/splits.json).
# ---------------------------------------------------------------------------------------------
def held_out(letter: str, step: int = 25, count: int | None = None) -> list[Path]:
    from src.config import DATASET_DIR

    paths = [DATASET_DIR / letter / f"{letter}{n}.jpg" for n in range(2551, 3001, step)]
    paths = [p for p in paths if p.exists()]
    return paths[:count] if count else paths


@unittest.skipUnless(BEST_MODEL_PATH.exists(), "trained model not present")
class LetterCRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.inference.engine import Predictor

        cls.predictor = Predictor(BEST_MODEL_PATH)

    def top1(self, image: Image.Image) -> tuple[str, float]:
        result = self.predictor.predict_pil(image)
        return result["prediction"], result["confidence"]

    def test_c_has_class_index_2_and_the_checkpoint_order_is_a_to_z(self) -> None:
        self.assertEqual(self.predictor.class_names, LETTERS)
        self.assertEqual(self.predictor.class_names.index("C"), 2)

    def test_predictor_preprocessing_is_exactly_the_evaluation_transform(self) -> None:
        image = Image.open(held_out("C")[0]).convert("RGB")
        self.assertTrue(torch.equal(self.predictor.transform(image), eval_transforms(self.predictor.image_size)(image)))

    def test_correctly_labelled_held_out_c_images_are_recognised_confidently(self) -> None:
        paths = held_out("C")
        self.assertGreaterEqual(len(paths), 10)
        for path in paths:
            label, confidence = self.top1(Image.open(path))
            self.assertEqual(label, "C", f"{path.name} read as {label} ({confidence:.2f})")
            self.assertGreater(confidence, CONFIDENCE_THRESHOLD, path.name)  # must clear the app's own acceptance threshold

    def test_c_and_w_are_not_confused_with_each_other(self) -> None:
        c_as_w = sum(self.top1(Image.open(p))[0] == "W" for p in held_out("C"))
        w_as_c = sum(self.top1(Image.open(p))[0] == "C" for p in held_out("W"))
        self.assertEqual((c_as_w, w_as_c), (0, 0))

    def test_c_survives_the_live_square_crop_when_a_hand_is_found(self) -> None:
        import numpy as np

        detector = HandDetector(static_image_mode=True)
        self.addCleanup(detector.close)
        if not detector.available:
            self.skipTest("hand detector unavailable")
        checked = 0
        for path in held_out("C", step=5):
            rgb = np.array(Image.open(path).convert("RGB"))
            box = detector.landmark_box(rgb)
            if box is None:
                continue
            checked += 1
            for pad in (0.30, 0.60):
                label, _ = self.top1(crop_square(rgb, square_window(box, pad)))
                self.assertEqual(label, "C", f"{path.name} pad {pad}")
        self.assertGreater(checked, 0, "no held-out C image had a detectable hand")

    def test_mirrored_c_is_still_read_as_c(self) -> None:
        # Guards the robustness the checkpoint was retrained for: a left-handed signer's C is the
        # mirror image. The earlier checkpoint read mirrored C as G/Q (0-7% correct).
        from PIL import ImageOps

        paths = held_out("C")
        hits = sum(self.top1(ImageOps.mirror(Image.open(p).convert("RGB")))[0] == "C" for p in paths)
        self.assertGreaterEqual(hits / len(paths), 0.8, f"only {hits}/{len(paths)} mirrored C images recognised")
