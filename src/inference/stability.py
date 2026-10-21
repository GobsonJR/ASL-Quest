"""Python mirror of the live frontend's StableLetterTracker (frontend/src/shared/recognition.ts).

The app decides "this letter is confirmed" in the browser. To MEASURE how long that takes and how often it confirms a
wrong letter on recorded frames, the same rules are reproduced here; keep the constants in sync with the TypeScript file
(tests/test_stability.py pins the shared behaviour).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

VOTE_WINDOW = 7
MIN_STABLE_VOTES = 3
NO_HAND_GRACE_FRAMES = 2
LOW_CONFIDENCE_RESET_FRAMES = 6
# A hand box that overlaps its previous position by less than this (IoU) means the hand moved/changed substantially,
# so votes for the old pose must not count towards the new one.
HAND_MOVED_IOU = 0.35

NO_HAND, LOW_CONFIDENCE, RECOGNIZED, DETECTOR_UNAVAILABLE = "no_hand", "low_confidence", "recognized", "detector_unavailable"

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def majority(labels: list[str]) -> str | None:
    if len(labels) < MIN_STABLE_VOTES:
        return None
    label, count = Counter(labels).most_common(1)[0]
    return label if count >= max(MIN_STABLE_VOTES, len(labels) // 2 + 1) else None


@dataclass
class TrackerResult:
    stable: str | None
    phase: str
    frames_in_phase: int


class StableLetterTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.votes: list[str] = []
        self.miss_streak = 0
        self.weak_streak = 0
        self.last_phase: str | None = None
        self.phase_frames = 0
        self.last_box: Box | None = None

    def update(self, state: str, label: str | None, box: Box | None = None) -> TrackerResult:
        stable = None
        if state == DETECTOR_UNAVAILABLE:
            self.votes = []
            phase = DETECTOR_UNAVAILABLE
        elif state == NO_HAND:
            self.miss_streak += 1
            self.weak_streak = 0
            if self.miss_streak >= NO_HAND_GRACE_FRAMES:
                self.votes = []
                self.last_box = None
            phase = NO_HAND
        else:
            self.miss_streak = 0
            if box is not None:
                if self.last_box is not None and iou(self.last_box, box) < HAND_MOVED_IOU:
                    self.votes = []  # the hand moved to a different place/size: start over
                    self.weak_streak = 0
                self.last_box = box
            if state == LOW_CONFIDENCE:
                self.weak_streak += 1
                if self.weak_streak >= LOW_CONFIDENCE_RESET_FRAMES:
                    self.votes = []
                phase = LOW_CONFIDENCE
            else:
                self.weak_streak = 0
                if label:
                    self.votes = (self.votes + [label])[-VOTE_WINDOW:]
                stable = majority(self.votes)
                phase = "stable" if stable else "hand_waiting"
        self.phase_frames = self.phase_frames + 1 if phase == self.last_phase else 1
        self.last_phase = phase
        return TrackerResult(stable, phase, self.phase_frames)
