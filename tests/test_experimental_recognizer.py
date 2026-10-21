"""Experimental landmark recognizer: safe loading, per-request switching, and failures that must not crash the app."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from fastapi import HTTPException
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from starlette.datastructures import UploadFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.routers import ml  # noqa: E402
from backend.services import experimental_recognizer as er  # noqa: E402
from src.inference.hands import HandCrop, HandObservation  # noqa: E402
from src.landmarks.classifier import LandmarkClassifier  # noqa: E402
from src.landmarks.features import FEATURE_DIM, LandmarkError  # noqa: E402

LETTERS = [chr(65 + i) for i in range(26)]
REAL_L2 = ROOT / "models" / "experiments" / "landmark_L2_s0.joblib"


def tiny_model(path: Path, *, classes=None, n_features: int = FEATURE_DIM) -> Path:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(52, n_features))
    y = np.repeat(np.arange(26), 2)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]).fit(x, y)
    joblib.dump({"pipeline": pipe, "classes": classes or LETTERS, "rotate": False, "model": "logreg",
                 "trained_on": {"rooms": ["a", "b"], "n_samples": 52}, "created_at": "now"}, path)
    return path


def hand(wobble: float = 0.0) -> HandObservation:
    pts = [(0.5, 0.8, 0.0)] + [(0.5 + 0.03 * ((i % 4) - 1.5) + wobble, 0.8 - 0.03 * i, 0.0) for i in range(1, 21)]
    return HandObservation(pts, "Right", 0.97)


def jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (320, 240), (120, 120, 120)).save(buffer, "JPEG")
    return buffer.getvalue()


class StubImageModel:
    model_path = Path("stub.pth")
    class_names = LETTERS

    def predict_pil(self, image, top_k: int = 3):
        return {"prediction": "B", "confidence": 0.9, "top_predictions": [{"label": "B", "confidence": 0.9}], "low_confidence": False, "diagnostics": {}}


class StubDetector:
    available = True
    error = None

    def __init__(self, observation=None, observe_raises: bool = False) -> None:
        self.observation, self.observe_raises = observation, observe_raises

    def observe(self, rgb):
        if self.observe_raises:
            raise RuntimeError("mediapipe exploded")
        return self.observation

    def crop_rgb(self, rgb, padding: float = 0.30, observation=None) -> HandCrop:
        return HandCrop(image=Image.fromarray(rgb[10:110, 10:110]), box=(10, 10, 110, 110), detected=True)

    def crop_pil(self, image, padding: float = 0.30) -> HandCrop:
        return HandCrop(image=image.crop((10, 10, 110, 110)), box=(10, 10, 110, 110), detected=True)


class StubLandmarks:
    path = Path("stub_L2.joblib")

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def predict_observation(self, observation, top_k: int = 3):
        if self.error:
            raise self.error
        return {"prediction": "F", "confidence": 0.95, "top_predictions": [{"label": "F", "confidence": 0.95}], "low_confidence": False}


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "models"
        (self.root / "experiments").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def load(self, text: str):
        return er.load_experimental_landmarks(text, allowed_root=self.root)

    def test_feature_is_off_when_not_configured(self) -> None:
        self.assertEqual(self.load(""), (None, None))
        self.assertEqual(self.load("   "), (None, None))

    def test_a_valid_model_loads_and_is_described(self) -> None:
        path = tiny_model(self.root / "experiments" / "ok.joblib")
        classifier, error = self.load(str(path))
        self.assertIsNone(error)
        info = er.describe(classifier, error)
        self.assertEqual((info["loaded"], info["model_file"], info["model"]), (True, "ok.joblib", "logreg"))

    def test_missing_file_is_an_error_message_not_an_exception(self) -> None:
        classifier, error = self.load(str(self.root / "experiments" / "nope.joblib"))
        self.assertIsNone(classifier)
        self.assertIn("not found", error)
        self.assertFalse(er.describe(classifier, error)["loaded"])

    def test_files_outside_models_and_wrong_extensions_are_refused(self) -> None:
        outside = tiny_model(Path(self.tmp.name) / "elsewhere.joblib")
        self.assertIn("must live under", self.load(str(outside))[1])
        self.assertIn(".joblib", self.load(str(self.root / "experiments" / "m.pkl"))[1])

    def test_corrupt_file_does_not_raise(self) -> None:
        bad = self.root / "experiments" / "bad.joblib"
        bad.write_bytes(b"this is not a joblib file")
        classifier, error = self.load(str(bad))
        self.assertIsNone(classifier)
        self.assertIn("could not load", error)

    def test_wrong_class_order_is_rejected(self) -> None:
        path = tiny_model(self.root / "experiments" / "order.joblib", classes=list(reversed(LETTERS)))
        classifier, error = self.load(str(path))
        self.assertIsNone(classifier)
        self.assertIn("class order", error)

    def test_a_model_with_a_different_feature_schema_is_rejected(self) -> None:
        path = tiny_model(self.root / "experiments" / "dim.joblib", n_features=FEATURE_DIM - 3)
        classifier, error = self.load(str(path))
        self.assertIsNone(classifier)
        self.assertIn(f"{FEATURE_DIM - 3} features", error)

    @unittest.skipUnless(REAL_L2.exists(), "the experimental L2 checkpoint is not on this machine")
    def test_the_real_l2_checkpoint_matches_the_live_feature_schema(self) -> None:
        classifier, error = er.load_experimental_landmarks(str(REAL_L2))
        self.assertIsNone(error)
        self.assertEqual(classifier.pipeline[0].n_features_in_, FEATURE_DIM)
        reply = classifier.predict_observation(hand(), top_k=3)
        self.assertIn(reply["prediction"], LETTERS)
        self.assertAlmostEqual(sum(t["confidence"] for t in classifier.predict_observation(hand(), top_k=26)["top_predictions"]), 1.0, places=5)


class PerRequestSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        ml.set_runtime(StubImageModel(), StubDetector(hand()), None)
        ml.set_recognizer("resnet", None, None)
        ml.set_experimental_landmarks(StubLandmarks(), None)

    def tearDown(self) -> None:
        ml.set_runtime(None, None, None)
        ml.set_recognizer("resnet", None, None)
        ml.set_experimental_landmarks(None, None)

    def call(self, **kwargs):
        upload = UploadFile(filename="f.jpg", file=BytesIO(jpeg()))
        return json.loads(asyncio.run(ml.predict(upload, **kwargs)).body)

    def test_resnet_stays_the_default_even_when_the_experimental_model_is_loaded(self) -> None:
        body = self.call(require_hand=True)
        self.assertEqual(body["prediction"], "B")
        self.assertNotIn("recognizer", body)  # the production response is unchanged

    def test_an_explicit_landmarks_request_uses_the_experimental_model(self) -> None:
        body = self.call(recognizer="landmarks")
        self.assertEqual((body["prediction"], body["recognizer"], body["landmark_model"], body["status"]), ("F", "landmarks", "stub_L2.joblib", "ok"))
        self.assertEqual(self.call(recognizer="resnet")["prediction"], "B")  # and back again, per request

    def test_explicit_resnet_overrides_a_server_wide_landmarks_setting(self) -> None:
        ml.set_recognizer("landmarks", StubLandmarks())
        self.assertEqual(self.call(recognizer="resnet")["prediction"], "B")

    def test_landmarks_requested_but_not_loaded_is_a_clear_503(self) -> None:
        ml.set_experimental_landmarks(None, "experimental landmark model not found: x")
        with self.assertRaises(HTTPException) as ctx:
            self.call(recognizer="landmarks")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("not found", ctx.exception.detail)
        self.assertEqual(self.call()["prediction"], "B")  # the production path is untouched

    def test_unknown_recognizer_value_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self.call(recognizer="resnet50")
        self.assertEqual(ctx.exception.status_code, 422)

    def test_no_hand_is_a_normal_answer_not_an_error(self) -> None:
        ml.set_runtime(StubImageModel(), StubDetector(None), None)
        body = self.call(recognizer="landmarks")
        self.assertEqual((body["status"], body["prediction"], body["hand_detected"]), ("no_hand_detected", None, False))

    def test_a_crashing_hand_detector_does_not_crash_the_request(self) -> None:
        ml.set_runtime(StubImageModel(), StubDetector(hand(), observe_raises=True), None)
        body = self.call(recognizer="landmarks")
        self.assertEqual((body["status"], body["prediction"]), ("hand_detector_unavailable", None))

    def test_unusable_landmarks_are_reported_as_no_hand(self) -> None:
        ml.set_experimental_landmarks(StubLandmarks(LandmarkError("degenerate hand (zero palm length)")), None)
        body = self.call(recognizer="landmarks")
        self.assertEqual((body["status"], body["prediction"]), ("no_hand_detected", None))
        self.assertIn("unusable", body["message"])

    def test_a_model_failure_is_a_503_not_an_unhandled_error(self) -> None:
        ml.set_experimental_landmarks(StubLandmarks(RuntimeError("sklearn blew up")), None)
        with self.assertRaises(HTTPException) as ctx:
            self.call(recognizer="landmarks")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertNotIn("sklearn", ctx.exception.detail)  # no internals leaked to the client

    def test_real_classifier_with_non_finite_landmarks_is_handled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            classifier = LandmarkClassifier(tiny_model(Path(tmp) / "m.joblib"))
        bad = HandObservation([(float("nan"), 0.5, 0.0)] * 21, "Right", 0.9)
        ml.set_runtime(StubImageModel(), StubDetector(bad), None)
        ml.set_experimental_landmarks(classifier, None)
        body = self.call(recognizer="landmarks")
        self.assertEqual((body["status"], body["prediction"]), ("no_hand_detected", None))

    def test_health_reports_the_experimental_model_without_changing_the_default(self) -> None:
        health = ml.health()
        self.assertEqual(health["recognizer"], "resnet")
        self.assertEqual((health["experimental_landmark"]["loaded"], health["experimental_landmark"]["model_file"]), (True, "stub_L2.joblib"))
        ml.set_experimental_landmarks(None, "boom")
        self.assertEqual(ml.health()["experimental_landmark"], {"loaded": False, "model_file": None, "model": None, "trained_on": None, "error": "boom"})


if __name__ == "__main__":
    unittest.main()
