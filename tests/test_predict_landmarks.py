"""/predict_landmarks: validation, handedness/mirroring, class order and feature dimension (no retraining, no real model needed)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from fastapi import HTTPException
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.routers import ml  # noqa: E402
from src.landmarks.classifier import LETTERS, LandmarkClassifier  # noqa: E402
from src.landmarks.features import FEATURE_DIM, N_LANDMARKS, canonicalize, feature_vector  # noqa: E402


def tiny_classifier(directory: Path) -> LandmarkClassifier:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(52, FEATURE_DIM))
    y = np.repeat(np.arange(26), 2)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]).fit(x, y)
    path = directory / "tiny.joblib"
    joblib.dump({"pipeline": pipe, "classes": LETTERS, "rotate": False, "model": "logreg", "trained_on": {}, "created_at": "now"}, path)
    return LandmarkClassifier(path)


def points(shift: float = 0.0) -> list[list[float]]:
    return [[0.5 + shift, 0.8, 0.0]] + [[0.5 + shift + 0.03 * ((i % 4) - 1.5), 0.8 - 0.03 * i, 0.0] for i in range(1, 21)]


class PredictLandmarksTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.classifier = tiny_classifier(Path(self._tmp.name))
        ml.set_experimental_landmarks(self.classifier, None)
        self.addCleanup(ml.set_experimental_landmarks, None, None)

    def call(self, **kwargs):
        import json

        return json.loads(ml.predict_landmarks(ml.LandmarkPayload(**kwargs)).body)

    def test_feature_dimension_and_class_order(self) -> None:
        self.assertEqual(len(feature_vector(points(), "right")), FEATURE_DIM)
        self.assertEqual(LETTERS, [chr(65 + i) for i in range(26)])
        self.assertEqual(self.classifier.pipeline[0].n_features_in_, FEATURE_DIM)

    def test_valid_request_returns_a_letter(self) -> None:
        reply = self.call(landmarks=points(), handedness="Right")
        self.assertEqual(reply["status"], "ok")
        self.assertIn(reply["prediction"], LETTERS)
        self.assertEqual(reply["actual_hand"], "left")  # MediaPipe "Right" = person's left hand in an unmirrored frame
        self.assertIn("classify", reply["timings_ms"])

    def test_matches_the_image_path_classifier_for_the_same_landmarks(self) -> None:
        from src.inference.hands import HandObservation

        direct = self.classifier.predict_observation(HandObservation([tuple(p) for p in points()], "Left", None))
        reply = self.call(landmarks=points(), handedness="Left")
        self.assertEqual(reply["prediction"], direct["prediction"])
        self.assertAlmostEqual(reply["confidence"], direct["confidence"])

    def test_mirrored_frame_equals_the_unmirrored_frame_it_came_from(self) -> None:
        unmirrored = self.call(landmarks=points(), handedness="Left")
        flipped = [[1.0 - x, y, z] for x, y, z in points()]
        mirrored = self.call(landmarks=flipped, handedness="Right", image_mirrored=True)
        self.assertEqual(unmirrored["prediction"], mirrored["prediction"])
        self.assertAlmostEqual(unmirrored["confidence"], mirrored["confidence"], places=6)

    def test_canonicalization_makes_both_hands_look_alike(self) -> None:
        right = np.array(points())
        left = right.copy()
        left[:, 0] = 1.0 - left[:, 0]
        np.testing.assert_allclose(canonicalize(right, "right"), canonicalize(left, "left"), atol=1e-9)

    def test_malformed_payloads_are_rejected(self) -> None:
        bad = {
            "too few points": dict(landmarks=points()[:20]),
            "too many points": dict(landmarks=points() + [[0.5, 0.5, 0.0]]),
            "wrong width": dict(landmarks=[[0.5, 0.5]] * N_LANDMARKS),
            "nan": dict(landmarks=[[float("nan"), 0.5, 0.0]] * N_LANDMARKS),
            "inf": dict(landmarks=[[float("inf"), 0.5, 0.0]] * N_LANDMARKS),
            "out of range": dict(landmarks=[[40.0, 0.5, 0.0]] * N_LANDMARKS),
            "bad handedness": dict(landmarks=points(), handedness="Middle"),
        }
        for name, kwargs in bad.items():
            with self.subTest(name):
                with self.assertRaises(HTTPException) as ctx:
                    self.call(**kwargs)
                self.assertEqual(ctx.exception.status_code, 422)

    def test_degenerate_hand_is_422_not_500(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self.call(landmarks=[[0.5, 0.5, 0.0]] * N_LANDMARKS, handedness="Right")
        self.assertEqual(ctx.exception.status_code, 422)

    def test_503_when_no_model_is_loaded(self) -> None:
        ml.set_experimental_landmarks(None, "not configured")
        saved = ml.landmark_classifier
        ml.landmark_classifier = None
        self.addCleanup(setattr, ml, "landmark_classifier", saved)
        with self.assertRaises(HTTPException) as ctx:
            self.call(landmarks=points(), handedness="Right")
        self.assertEqual(ctx.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
