"""Pins the handedness / mirroring conventions of the landmark pipeline (see outputs/recognition_investigation/REPORT.md).

These tests describe CURRENT behaviour and the invariants it relies on. They deliberately do not "fix" anything: the
investigation measured that MediaPipe's raw label equals the hand a person says they used in 257/259 real webcam frames
(so the Left->right mapping is mis-named for those frames) while the pipeline stays self-consistent because training and
live use the same mapping. Changing the mapping would silently mirror every trained model's input; if it ever has to change,
these tests must be changed on purpose, together with a retrain that is reported and approved.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.hands import HandObservation  # noqa: E402
from src.landmarks.classifier import MEDIAPIPE_TO_ACTUAL_HAND, with_deployed_hand  # noqa: E402
from src.landmarks.features import canonicalize, feature_vector  # noqa: E402


def _hand(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 0.05, (21, 3)) + np.array([0.5, 0.6, 0.0])
    pts[0] = [0.5, 0.8, 0.0]
    pts[9] = [0.5, 0.5, -0.02]  # middle MCP above the wrist: a non-degenerate palm
    pts[:, 2] = rng.normal(0, 0.02, 21)
    return pts


def _mirror_image(points: np.ndarray) -> np.ndarray:
    """What flipping the image left-right does to normalised landmarks: x -> 1 - x, y and depth unchanged."""
    out = points.copy()
    out[:, 0] = 1.0 - out[:, 0]
    return out


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MappingIsSingleSourceOfTruth(unittest.TestCase):
    def test_observation_actual_hand_uses_the_training_mapping(self) -> None:
        for raw, expected in MEDIAPIPE_TO_ACTUAL_HAND.items():
            self.assertEqual(HandObservation([(0.0, 0.0, 0.0)] * 21, raw, 0.9).actual_hand, expected)
        self.assertIsNone(HandObservation([(0.0, 0.0, 0.0)] * 21, None, None).actual_hand)

    def test_current_mapping_is_pinned(self) -> None:
        # Measured on real webcam frames: the raw label matched the person's stated hand in 257/259 frames, i.e. this
        # table is inverted relative to the person's own words. It is still applied identically in training and live, so
        # it is a naming problem, not a runtime error. Do not edit without retraining and re-evaluating.
        self.assertEqual(MEDIAPIPE_TO_ACTUAL_HAND, {"Left": "right", "Right": "left"})

    def test_training_and_live_resolve_the_hand_identically(self) -> None:
        for raw in ("Left", "Right"):
            trained = with_deployed_hand([{"letter": "A", "hand": "x", "handedness_raw": raw}])[0]["hand"]
            live = HandObservation([(0.0, 0.0, 0.0)] * 21, raw, 0.9).actual_hand
            self.assertEqual(trained, live)


class MirrorInvariance(unittest.TestCase):
    """A hand and its mirror image must give identical features once each carries the opposite hand flag."""

    def test_canonical_form_ignores_mirroring_for_both_rotation_modes(self) -> None:
        pts = _hand()
        for rotate in (True, False):
            np.testing.assert_allclose(
                canonicalize(pts, "right", rotate=rotate), canonicalize(_mirror_image(pts), "left", rotate=rotate), atol=1e-9
            )

    def test_depth_is_not_negated_by_mirroring(self) -> None:
        # a mirror reflection in x leaves depth alone, so the hand flag must flip x only
        canon = canonicalize(_mirror_image(_hand()), "left", rotate=False)
        np.testing.assert_allclose(canon[:, 2], canonicalize(_hand(), "right", rotate=False)[:, 2], atol=1e-9)

    def test_feature_vector_is_mirror_invariant_with_opposite_flags(self) -> None:
        pts = _hand()
        for rotate in (True, False):
            np.testing.assert_allclose(
                feature_vector(pts, "right", rotate=rotate), feature_vector(_mirror_image(pts), "left", rotate=rotate), atol=1e-9
            )

    def test_a_wrong_hand_flag_changes_the_features(self) -> None:
        # this is the failure mode a mislabelled hand causes: same landmarks, opposite flag, different input to the model
        pts = _hand()
        self.assertGreater(np.abs(feature_vector(pts, "right", rotate=False) - feature_vector(pts, "left", rotate=False)).max(), 0.05)


class InvestigationHelpers(unittest.TestCase):
    def test_chirality_flips_under_mirroring(self) -> None:
        module = _load_script("investigate_handedness")
        pts = _hand()
        pts[5], pts[17] = [0.55, 0.6, 0.0], [0.42, 0.65, 0.0]
        self.assertEqual(module.chirality(pts), -module.chirality(_mirror_image(pts)))

    def test_convention_table(self) -> None:
        module = _load_script("investigate_handedness")
        self.assertEqual(module.convention_hand("Left", "as_deployed"), "right")
        self.assertEqual(module.convention_hand("Right", "as_deployed"), "left")
        self.assertEqual(module.convention_hand("Left", "never_flip"), "right")
        self.assertEqual(module.convention_hand("Right", "always_flip"), "left")
        self.assertIsNone(module.convention_hand(None, "as_deployed"))


if __name__ == "__main__":
    unittest.main()
