"""Recognizer switch (resnet default / landmarks opt-in) and the evaluation helpers in scripts/evaluate_webcam.py."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.routers import ml  # noqa: E402
from src.config import CONFIDENCE_THRESHOLD  # noqa: E402
from src.inference.hands import HandCrop, HandObservation  # noqa: E402

LETTERS = [chr(65 + i) for i in range(26)]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (320, 240), (120, 120, 120)).save(buffer, "JPEG")
    return buffer.getvalue()


class StubImageModel:
    model_path = Path("stub.pth")
    class_names = LETTERS

    def predict_pil(self, image, top_k: int = 3):
        return {"prediction": "B", "confidence": 0.9, "top_predictions": [{"label": "B", "confidence": 0.9}], "low_confidence": False,
                "diagnostics": {}}


class ExplodingImageModel(StubImageModel):
    def predict_pil(self, *a, **k):
        raise AssertionError("the image classifier must not run when the landmark recognizer is selected")


class StubDetector:
    available = True
    error = None

    def __init__(self, found: bool = True) -> None:
        self.found = found

    def observe(self, rgb):
        return HandObservation([(0.4 + 0.01 * i, 0.5, 0.0) for i in range(21)], "Left", 0.97) if self.found else None

    def crop_rgb(self, rgb, padding: float = 0.30, observation=None) -> HandCrop:
        return HandCrop(image=Image.fromarray(rgb[10:110, 10:110]), box=(10, 10, 110, 110), detected=True)

    def crop_pil(self, image, padding: float = 0.30) -> HandCrop:
        return HandCrop(image=image.crop((10, 10, 110, 110)), box=(10, 10, 110, 110), detected=True)


class StubLandmarkClassifier:
    def predict_observation(self, observation, top_k: int = 3):
        return {"prediction": "C", "confidence": 0.93, "top_predictions": [{"label": "C", "confidence": 0.93}], "low_confidence": False}


class RecognizerSwitchTests(unittest.TestCase):
    def tearDown(self) -> None:
        ml.set_recognizer("resnet", None, None)
        ml.set_runtime(None, None, None)

    def call(self, **kwargs):
        upload = UploadFile(filename="f.jpg", file=BytesIO(jpeg()))
        return json.loads(asyncio.run(ml.predict(upload, **kwargs)).body)

    def test_default_is_the_image_classifier_and_its_response_is_unchanged(self) -> None:
        ml.set_runtime(StubImageModel(), StubDetector(), None)
        body = self.call(require_hand=True)
        self.assertEqual((body["prediction"], body["status"]), ("B", "ok"))
        self.assertNotIn("recognizer", body)
        self.assertEqual(ml.health.__name__, "health")

    def test_landmarks_recognizer_answers_without_touching_the_image_model(self) -> None:
        ml.set_runtime(ExplodingImageModel(), StubDetector(), None)
        ml.set_recognizer("landmarks", StubLandmarkClassifier())
        body = self.call(debug=True)
        self.assertEqual((body["prediction"], body["recognizer"], body["hand_box"]), ("C", "landmarks", [10, 10, 110, 110]))
        self.assertEqual(body["diagnostics"]["actual_hand"], "right")  # MediaPipe "Left" is the real right hand
        self.assertEqual(body["threshold"], CONFIDENCE_THRESHOLD)

    def test_landmarks_recognizer_reports_no_hand_instead_of_guessing(self) -> None:
        ml.set_runtime(ExplodingImageModel(), StubDetector(found=False), None)
        ml.set_recognizer("landmarks", StubLandmarkClassifier())
        body = self.call()
        self.assertEqual((body["status"], body["prediction"], body["hand_detected"]), ("no_hand_detected", None, False))

    def test_landmarks_selected_but_not_loaded_fails_loudly(self) -> None:
        ml.set_runtime(StubImageModel(), StubDetector(), None)
        ml.set_recognizer("landmarks", None, "Could not load the landmark model: file not found")
        with self.assertRaises(HTTPException) as ctx:
            self.call()
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("file not found", ctx.exception.detail)

    def test_health_reports_which_recognizer_is_active(self) -> None:
        ml.set_runtime(StubImageModel(), StubDetector(), None)
        ml.set_recognizer("landmarks", None, "missing")
        health = ml.health()
        self.assertEqual((health["recognizer"], health["landmark_model_loaded"], health["landmark_model_error"]), ("landmarks", False, "missing"))


class EvaluationHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ev = load_script("evaluate_webcam")

    def row(self, letter: str, **kw):
        return {"letter": letter, "kind": "single", "sequence_id": None, "frame_index": None, "room": "r", "background": "b", "hand": "right", **kw}

    def test_metrics_on_a_hand_checked_example(self) -> None:
        rows = [self.row("C"), self.row("C"), self.row("C"), self.row("W")]
        preds = [[("C", 0.95)], [("C", 0.5)], [("W", 0.9)], [("W", 0.8)]]
        m = self.ev.metrics_for(rows, preds)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["per_letter"]["C"]["recall"], 2 / 3)
        self.assertAlmostEqual(m["per_letter"]["C"]["precision"], 1.0)
        self.assertAlmostEqual(m["per_letter"]["W"]["precision"], 0.5)
        self.assertEqual(m["rejected_uncertain"], 1)               # the 0.5-confidence C
        self.assertAlmostEqual(m["accuracy_among_accepted"], 2 / 3)  # C@.95 right, W@.9 wrong-for-C, W@.8 right
        self.assertEqual(m["per_letter"]["C"]["top_wrong"], {"W": 1})
        self.assertEqual(m["letters_present"], 2)

    def test_a_missing_prediction_counts_as_not_scored_never_as_correct(self) -> None:
        m = self.ev.metrics_for([self.row("C"), self.row("C")], [[("C", 0.9)], None])
        self.assertEqual((m["n_scored"], m["no_prediction"]), (1, 1))

    def test_grouped_accuracy_counts_missing_predictions_as_wrong(self) -> None:
        rows = [self.row("C", room="a"), self.row("C", room="a"), self.row("C", room="b")]
        got = self.ev.grouped_accuracy(rows, [[("C", 1.0)], None, [("W", 1.0)]], "room")
        self.assertEqual((got["a"]["accuracy"], got["a"]["n"], got["b"]["accuracy"]), (0.5, 2, 0.0))

    def hold(self, letter: str, labels: list[tuple[str, float]], seq: str):
        rows = [self.row(letter, kind="hold", sequence_id=seq, frame_index=i) for i in range(len(labels))]
        return rows, [[(l, c)] for l, c in labels]

    def test_stability_reports_correct_wrong_and_never_confirmed_sequences(self) -> None:
        good_rows, good = self.hold("C", [("C", 0.9)] * 5, "good")
        wrong_rows, wrong = self.hold("C", [("W", 0.9)] * 5, "wrong")
        weak_rows, weak = self.hold("C", [("C", 0.4)] * 5, "weak")
        result = self.ev.stability(good_rows + wrong_rows + weak_rows, good + wrong + weak)
        self.assertEqual(result["sequences"], 3)
        self.assertAlmostEqual(result["confirmed_correct"], 1 / 3)
        self.assertAlmostEqual(result["confirmed_wrong"], 1 / 3)   # a confidently wrong letter IS confirmed: this must be visible
        self.assertAlmostEqual(result["never_confirmed"], 1 / 3)
        self.assertEqual(result["median_frames_to_confirm"], 3)
        self.assertEqual(result["wrong_confirmations"], {"C->W": 1})  # string keys: tuple keys made the JSON report crash
        json.dumps(result)  # the whole result must be serialisable

    def test_thin_hold_keeps_singles_and_a_few_frames_per_sequence(self) -> None:
        rows = [self.row("C")] + self.hold("C", [("C", 0.9)] * 10, "s")[0]
        kept = self.ev.thin_hold(rows)
        self.assertEqual(len(kept), 1 + 3)


if __name__ == "__main__":
    unittest.main()
