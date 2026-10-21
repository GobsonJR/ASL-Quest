"""Tests for the landmark-based classifier (src/landmarks). Synthetic hands test the CODE (invariances, selection
rules, round trip); they say nothing about real-world accuracy, which only real webcam data can show."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.hands import HandObservation  # noqa: E402
from src.landmarks.classifier import LETTERS, LandmarkClassifier, make_xy, save_model, select_and_train  # noqa: E402
from src.landmarks.features import FEATURE_DIM, LandmarkError, augment_landmarks, canonicalize, feature_vector  # noqa: E402

# curl (0 = straight, 1 = fully bent) for thumb, index, middle, ring, pinky: a different pattern per class
PATTERNS = {
    "A": (0.1, 1.0, 1.0, 1.0, 1.0),
    "B": (0.9, 0.0, 0.0, 0.0, 0.0),
    "C": (0.4, 0.5, 0.5, 0.5, 0.5),
    "L": (0.0, 0.0, 1.0, 1.0, 1.0),
    "V": (0.9, 0.0, 0.0, 1.0, 1.0),
    "Y": (0.0, 1.0, 1.0, 1.0, 0.0),
}


def make_hand(curls, rng: np.random.Generator, *, hand: str = "right", angle: float = 0.0, scale: float = 1.0, shift=(0.5, 0.6), noise: float = 0.004) -> np.ndarray:
    """21 landmarks of a toy hand: palm 0.2 long (image-plane), fingers bent by `curls`."""
    pts = np.zeros((21, 3))
    bases = {1: (-0.06, -0.04), 5: (-0.045, -0.2), 9: (-0.005, -0.21), 13: (0.035, -0.2), 17: (0.07, -0.17)}
    for finger, (first, curl) in enumerate(zip((1, 5, 9, 13, 17), curls)):
        bx, by = bases[first]
        direction = np.array([0.0, -1.0]) if finger else np.array([-0.7, -0.7])
        pos = np.array([bx, by])
        pts[first, :2] = pos
        heading = np.arctan2(direction[0], -direction[1])
        for seg in range(1, 4):
            heading += curl * 1.1 * (1 if finger else -1)
            step = 0.07 * np.array([np.sin(heading), -np.cos(heading)])
            pos = pos + step
            pts[first + seg, :2] = pos
    c, s = np.cos(angle), np.sin(angle)
    pts[:, :2] = pts[:, :2] @ np.array([[c, s], [-s, c]])
    pts[:, :2] *= scale
    if hand == "left":
        pts[:, 0] *= -1
    pts[:, :2] += np.asarray(shift)
    return pts + rng.normal(0, noise, pts.shape)


def rows_for(rooms: dict[str, dict], n: int, rng: np.random.Generator) -> list[dict]:
    rows = []
    for room, style in rooms.items():
        for letter, curls in PATTERNS.items():
            for _ in range(n):
                hand = rng.choice(["left", "right"])
                pts = make_hand(curls, rng, hand=hand, angle=rng.uniform(-0.3, 0.3), scale=style["scale"] * rng.uniform(0.9, 1.1), shift=style["shift"] + rng.uniform(-0.05, 0.05, 2))
                rows.append({"landmarks": pts.tolist(), "hand": hand, "letter": letter, "room": room})
    return rows


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        self.hand = make_hand(PATTERNS["C"], self.rng, noise=0.0)

    def test_feature_vector_has_a_fixed_length(self) -> None:
        self.assertEqual(len(feature_vector(self.hand)), FEATURE_DIM)
        self.assertEqual(len(feature_vector(make_hand(PATTERNS["V"], self.rng))), FEATURE_DIM)

    def test_translation_and_scale_do_not_change_the_features(self) -> None:
        base = feature_vector(self.hand)
        moved = feature_vector(self.hand * np.array([1.0, 1.0, 1.0]) + np.array([0.3, -0.2, 0.0]))
        resized = feature_vector((self.hand - self.hand[0]) * 2.7 + self.hand[0])
        np.testing.assert_allclose(base, moved, atol=1e-9)
        np.testing.assert_allclose(base, resized, atol=1e-9)

    def test_in_plane_rotation_is_removed_only_when_asked(self) -> None:
        c, s = np.cos(0.5), np.sin(0.5)
        rotated = self.hand.copy()
        rel = rotated[:, :2] - rotated[0, :2]
        rotated[:, :2] = rel @ np.array([[c, s], [-s, c]]) + rotated[0, :2]
        np.testing.assert_allclose(feature_vector(self.hand, rotate=True), feature_vector(rotated, rotate=True), atol=1e-9)
        self.assertGreater(np.abs(feature_vector(self.hand, rotate=False) - feature_vector(rotated, rotate=False)).max(), 0.1)

    def test_a_left_hand_is_mapped_onto_the_matching_right_hand(self) -> None:
        mirrored = self.hand.copy()
        mirrored[:, 0] = 2 * mirrored[0, 0] - mirrored[:, 0]
        np.testing.assert_allclose(feature_vector(self.hand, "right"), feature_vector(mirrored, "left"), atol=1e-9)

    def test_different_handshapes_give_different_features(self) -> None:
        a = feature_vector(make_hand(PATTERNS["A"], self.rng, noise=0))
        b = feature_vector(make_hand(PATTERNS["B"], self.rng, noise=0))
        self.assertGreater(np.linalg.norm(a - b), 1.0)

    def test_bad_input_is_rejected(self) -> None:
        with self.assertRaises(LandmarkError):
            feature_vector(np.zeros((20, 3)))
        with self.assertRaises(LandmarkError):
            feature_vector(np.full((21, 3), np.nan))
        with self.assertRaises(LandmarkError):
            feature_vector(np.zeros((21, 3)))  # zero palm length
        self.assertEqual(canonicalize(self.hand, None).shape, (21, 3))  # unknown hand is tolerated

    def test_augmentation_keeps_the_shape_close(self) -> None:
        jittered = augment_landmarks(self.hand, np.random.default_rng(1))
        self.assertEqual(jittered.shape, (21, 3))
        self.assertLess(np.linalg.norm(feature_vector(jittered) - feature_vector(self.hand)), 3.0)

    def test_augmented_copies_are_added_only_when_requested(self) -> None:
        rows = rows_for({"r": {"scale": 1.0, "shift": np.array([0.5, 0.6])}}, 2, self.rng)
        self.assertEqual(len(make_xy(rows, rotate=True)[0]), len(rows))
        self.assertEqual(len(make_xy(rows, rotate=True, augment=3)[0]), len(rows) * 4)


class SelectionAndRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(3)
        styles = {
            "kitchen": {"scale": 1.0, "shift": np.array([0.5, 0.6])},
            "hall": {"scale": 1.4, "shift": np.array([0.4, 0.5])},
            "office": {"scale": 0.8, "shift": np.array([0.6, 0.7])},
            "attic": {"scale": 1.2, "shift": np.array([0.55, 0.45])},
        }
        self.train = rows_for({k: styles[k] for k in ("kitchen", "hall")}, 12, self.rng)
        self.val = rows_for({"office": styles["office"]}, 12, self.rng)
        self.test = rows_for({"attic": styles["attic"]}, 12, self.rng)

    def test_refuses_to_select_a_model_without_unseen_data(self) -> None:
        one_room = [r for r in self.train if r["room"] == "kitchen"]
        with self.assertRaises(ValueError) as ctx:
            select_and_train(one_room, [], augment=1)
        self.assertIn("unseen-room", str(ctx.exception))

    def test_selects_on_unseen_room_validation_and_generalises_to_another_room(self) -> None:
        model, report = select_and_train(self.train, self.val, augment=2, seed=0)
        self.assertEqual(report["selection_data"], "unseen-room validation")
        self.assertIn(report["selected"], report["selection_scores_macro_f1"])
        self.assertEqual(set(model["trained_on"]["rooms"]), {"kitchen", "hall", "office"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lm.joblib"
            save_model(model, path)
            clf = LandmarkClassifier(path)
            correct = 0
            for row in self.test:
                reply = clf.predict_points(row["landmarks"], row["hand"])
                correct += reply["prediction"] == row["letter"]
            self.assertGreater(correct / len(self.test), 0.9)  # toy shapes in an unseen "room"

    def test_leave_room_out_cross_validation_when_no_validation_room(self) -> None:
        _, report = select_and_train(self.train, [], augment=1, seed=0)
        self.assertEqual(report["selection_data"], "leave-room-out cross-validation")

    def test_prediction_format_matches_the_image_classifier_and_flags_uncertainty(self) -> None:
        model, _ = select_and_train(self.train, self.val, augment=1, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lm.joblib"
            save_model(model, path)
            clf = LandmarkClassifier(path)
            obs = HandObservation([tuple(p) for p in self.test[0]["landmarks"]], "Left", 0.99)  # MediaPipe "Left" = real right hand
            reply = clf.predict_observation(obs, top_k=3)
            self.assertEqual(set(reply), {"prediction", "confidence", "top_predictions", "low_confidence"})
            self.assertEqual(len(reply["top_predictions"]), 3)
            self.assertAlmostEqual(sum(t["confidence"] for t in clf.predict_points(self.test[0]["landmarks"], "right", top_k=26)["top_predictions"]), 1.0, places=5)
            self.assertIsNone(clf.predict_observation(None))  # no hand -> no prediction (never a guess)
            garbage = clf.predict_points(np.random.default_rng(9).random((21, 3)), "right")
            self.assertIn(garbage["prediction"], LETTERS)

    def test_a_checkpoint_with_a_wrong_class_order_is_refused(self) -> None:
        model, _ = select_and_train(self.train, self.val, augment=1, seed=0)
        model["classes"] = list(reversed(LETTERS))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.joblib"
            save_model(model, path)
            with self.assertRaises(ValueError):
                LandmarkClassifier(path)


if __name__ == "__main__":
    unittest.main()


class HandSourceTests(unittest.TestCase):
    def test_training_rows_use_the_hand_the_live_pipeline_will_use(self) -> None:
        from src.landmarks.classifier import with_deployed_hand

        rows = [
            {"letter": "C", "hand": "right", "handedness_raw": "Right"},   # the person says right, MediaPipe says (mirrored) Right = real left
            {"letter": "C", "hand": "left", "handedness_raw": "Left"},     # MediaPipe "Left" = real right
            {"letter": "C", "hand": "left", "handedness_raw": None},       # no MediaPipe label: keep the label
        ]
        self.assertEqual([r["hand"] for r in with_deployed_hand(rows)], ["left", "right", "left"])
        self.assertEqual([r["hand"] for r in with_deployed_hand(rows, "label")], ["right", "left", "left"])
        self.assertEqual(rows[0]["hand"], "right")  # input rows are not modified
