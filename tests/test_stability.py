"""Behaviour of the Python mirror of the frontend tracker (and the hand-moved reset)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.stability import (  # noqa: E402
    HAND_MOVED_IOU,
    LOW_CONFIDENCE,
    LOW_CONFIDENCE_RESET_FRAMES,
    MIN_STABLE_VOTES,
    NO_HAND,
    NO_HAND_GRACE_FRAMES,
    RECOGNIZED,
    StableLetterTracker,
    iou,
    majority,
)

BOX = (100.0, 100.0, 300.0, 300.0)


class TrackerTests(unittest.TestCase):
    def test_needs_several_agreeing_confident_frames(self) -> None:
        t = StableLetterTracker()
        for _ in range(MIN_STABLE_VOTES - 1):
            self.assertIsNone(t.update(RECOGNIZED, "C", BOX).stable)
        self.assertEqual(t.update(RECOGNIZED, "C", BOX).stable, "C")

    def test_a_single_dropped_detection_does_not_erase_a_good_run(self) -> None:
        t = StableLetterTracker()
        for _ in range(2):
            t.update(RECOGNIZED, "C", BOX)
        t.update(NO_HAND, None)
        self.assertEqual(t.update(RECOGNIZED, "C", BOX).stable, "C")

    def test_the_hand_really_leaving_clears_the_votes(self) -> None:
        t = StableLetterTracker()
        for _ in range(3):
            t.update(RECOGNIZED, "C", BOX)
        for _ in range(NO_HAND_GRACE_FRAMES):
            t.update(NO_HAND, None)
        self.assertIsNone(t.update(RECOGNIZED, "C", BOX).stable)

    def test_weak_frames_neither_vote_nor_wipe_immediately_but_a_long_streak_resets(self) -> None:
        t = StableLetterTracker()
        for _ in range(2):
            t.update(RECOGNIZED, "C", BOX)
        self.assertIsNone(t.update(LOW_CONFIDENCE, "S", BOX).stable)
        self.assertEqual(t.update(RECOGNIZED, "C", BOX).stable, "C")
        t = StableLetterTracker()
        for _ in range(2):
            t.update(RECOGNIZED, "C", BOX)
        for _ in range(LOW_CONFIDENCE_RESET_FRAMES):
            t.update(LOW_CONFIDENCE, "S", BOX)
        self.assertIsNone(t.update(RECOGNIZED, "C", BOX).stable)

    def test_a_flickering_reading_is_never_confirmed(self) -> None:
        t = StableLetterTracker()
        self.assertTrue(all(t.update(RECOGNIZED, label, BOX).stable is None for label in "ABABCBA"))

    def test_votes_for_the_old_pose_do_not_count_after_the_hand_moves_substantially(self) -> None:
        t = StableLetterTracker()
        for _ in range(2):
            t.update(RECOGNIZED, "W", BOX)
        far = (600.0, 400.0, 800.0, 600.0)
        self.assertLess(iou(BOX, far), HAND_MOVED_IOU)
        # without the reset this third W vote would confirm W for a different hand placement
        self.assertIsNone(t.update(RECOGNIZED, "W", far).stable)

    def test_small_jitter_keeps_the_votes(self) -> None:
        t = StableLetterTracker()
        t.update(RECOGNIZED, "C", BOX)
        t.update(RECOGNIZED, "C", (104.0, 98.0, 305.0, 302.0))
        self.assertEqual(t.update(RECOGNIZED, "C", (95.0, 103.0, 297.0, 299.0)).stable, "C")

    def test_helpers(self) -> None:
        self.assertEqual(iou(BOX, BOX), 1.0)
        self.assertEqual(iou(BOX, (400.0, 400.0, 500.0, 500.0)), 0.0)
        self.assertIsNone(majority(["A", "A"]))
        self.assertEqual(majority(["A", "A", "A", "B"]), "A")


if __name__ == "__main__":
    unittest.main()
