"""Tests for src/data/webcam_export.py: room-separated splits, hold handling, label integrity."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.webcam_export import (  # noqa: E402
    ExportError,
    assign_by_session,
    assign_splits,
    evenly_spaced,
    export_dataset,
    leakage_report,
    load_sessions,
)
from src.inference.hands import HandCrop, HandObservation  # noqa: E402


class FakeDetector:
    """Detects a hand in every frame except completely black ones."""

    def crop_rgb(self, rgb, padding: float = 0.30) -> HandCrop:
        if float(rgb.mean()) < 3:
            return HandCrop(image=Image.fromarray(rgb), box=None, detected=False)
        return HandCrop(image=Image.fromarray(rgb[10:110, 10:110]), box=(10, 10, 110, 110), detected=True)

    def observe(self, rgb) -> HandObservation:
        points = [(0.5 + 0.01 * i, 0.5, 0.0) for i in range(21)]
        return HandObservation(points, "Left", 0.99)  # MediaPipe "Left" == the person's right hand in an unmirrored frame


def write_session(root: Path, session_id: str, room: str, letters: dict[str, int], *, hold: int = 0, blank: int = 0, hand: str = "right") -> None:
    folder = root / session_id
    folder.mkdir(parents=True)
    meta = {"session_id": session_id, "room": room, "lighting": "lamp", "background": "shelves", "face_visible": True, "glasses": False}
    (folder / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    rng = np.random.default_rng(abs(hash(session_id)) % 2**32)

    def add(letter: str, n: int, *, mode: str = "single", seq: str | None = None, idx: int | None = None, black: bool = False) -> None:
        sid = f"{letter}_{n}_{abs(hash((session_id, letter, n, seq, idx))) % 65536:04x}"
        (folder / letter).mkdir(exist_ok=True)
        pixels = rng.integers(1, 255, size=(120, 160, 3), dtype=np.uint8)
        if black:
            pixels[:] = 0
        Image.fromarray(pixels).save(folder / letter / f"{sid}.jpg")
        record = {
            "sample_id": sid, "session_id": session_id, "letter": letter, "label_source": "person_confirmed",
            "image": f"{sid}.jpg", "hand": hand, "distance": "medium", "mode": mode, "sequence_id": seq, "frame_index": idx,
            "lighting": "lamp", "background": "shelves", "face_visible": True, "glasses": False,
        }
        (folder / letter / f"{sid}.json").write_text(json.dumps(record), encoding="utf-8")

    for letter, count in letters.items():
        for n in range(count):
            add(letter, n)
        for n in range(blank):
            add(letter, 900 + n, black=True)
        if hold:
            for i in range(hold):
                add(letter, 500 + i, mode="hold", seq=f"h-{session_id}-{letter}", idx=i)


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = Path(self.tmp.name) / "webcam"
        for room in ("kitchen", "bedroom", "hall", "office"):
            for k in range(2):
                write_session(self.src, f"s20261019_1{k}0000_{room}", room, {"C": 4, "W": 4})
        self.sessions = load_sessions(self.src)

    def test_a_whole_room_never_straddles_splits(self) -> None:
        assignment, _ = assign_splits(self.sessions, auto=True, seed=1)
        by_room: dict[str, set[str]] = {}
        for sid, split in assignment.items():
            by_room.setdefault(self.sessions[sid]["room"], set()).add(split)
        self.assertTrue(all(len(v) == 1 for v in by_room.values()), by_room)
        self.assertIn("train", assignment.values())
        self.assertIn("test", assignment.values())
        self.assertTrue(leakage_report(assignment, self.sessions)["ok_room_separated"])

    def test_auto_split_is_deterministic(self) -> None:
        self.assertEqual(assign_splits(self.sessions, auto=True, seed=3)[0], assign_splits(self.sessions, auto=True, seed=3)[0])

    def test_explicit_rooms_are_honoured_and_validated(self) -> None:
        assignment, _ = assign_splits(self.sessions, test_rooms=["kitchen"], val_rooms=["hall"])
        self.assertEqual({sid: s for sid, s in assignment.items() if "kitchen" in sid}, {sid: "test" for sid in assignment if "kitchen" in sid})
        self.assertEqual({s for sid, s in assignment.items() if "hall" in sid}, {"val"})
        with self.assertRaises(ExportError):
            assign_splits(self.sessions, test_rooms=["attic"])
        with self.assertRaises(ExportError):
            assign_splits(self.sessions, test_rooms=["hall"], val_rooms=["hall"])

    def test_a_single_room_cannot_produce_an_unseen_room_test(self) -> None:
        one = {sid: e for sid, e in self.sessions.items() if e["room"] == "kitchen"}
        with self.assertRaises(ExportError) as ctx:
            assign_splits(one, auto=True)
        self.assertIn("at least 2 rooms", str(ctx.exception))

    def test_session_level_fallback_is_flagged_as_not_a_generalisation_test(self) -> None:
        one = {sid: e for sid, e in self.sessions.items() if e["room"] == "kitchen"}
        sids = sorted(one)
        assignment, notes = assign_by_session(one, test_sessions=[sids[1]], val_sessions=[])
        self.assertEqual(assignment[sids[1]], "test")
        self.assertIn("WARNING", notes[0])
        self.assertFalse(leakage_report(assignment, one)["ok_room_separated"])

    def test_evenly_spaced_keeps_ends_and_count(self) -> None:
        self.assertEqual(len(evenly_spaced(list(range(10)), 3)), 3)
        self.assertEqual(evenly_spaced(list(range(10)), 3)[0], 0)
        self.assertEqual(evenly_spaced(list(range(10)), 3)[-1], 9)
        self.assertEqual(evenly_spaced([1, 2], 5), [1, 2])


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = Path(self.tmp.name) / "webcam"
        self.out = Path(self.tmp.name) / "export"
        write_session(self.src, "s20261019_100000_kitchen", "kitchen", {"C": 3}, hold=10, blank=2)
        write_session(self.src, "s20261019_110000_hall", "hall", {"C": 3}, hold=10, hand="left")
        self.sessions = load_sessions(self.src)
        self.assignment = {"s20261019_100000_kitchen": "train", "s20261019_110000_hall": "test"}
        self.report = export_dataset(self.src, self.out, FakeDetector(), assignment=self.assignment, sessions=self.sessions)
        self.rows = [json.loads(line) for line in (self.out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_frames_without_a_hand_are_skipped_and_counted(self) -> None:
        self.assertEqual(self.report["skipped"], {"no hand detected": 2})
        self.assertEqual(len(self.rows), self.report["exported"])

    def test_hold_frames_are_thinned_for_training_but_kept_for_testing(self) -> None:
        train_hold = [r for r in self.rows if r["split"] == "train" and r["kind"] == "hold"]
        test_hold = [r for r in self.rows if r["split"] == "test" and r["kind"] == "hold"]
        self.assertEqual(len(train_hold), 3)
        self.assertEqual(len(test_hold), 10)
        self.assertEqual(sorted(r["frame_index"] for r in test_hold), list(range(10)))

    def test_labels_come_from_the_confirmed_metadata_and_files_land_in_split_folders(self) -> None:
        for row in self.rows:
            self.assertEqual(row["label_source"], "person_confirmed")
            self.assertTrue((self.out / row["crop"]).exists())
            self.assertTrue(row["crop"].startswith(f"{row['split']}/{row['letter']}/"))
        self.assertEqual({r["split"] for r in self.rows}, {"train", "test"})
        self.assertEqual({r["room"] for r in self.rows if r["split"] == "test"}, {"hall"})

    def test_landmarks_and_handedness_are_recorded_and_cross_checked(self) -> None:
        row = self.rows[0]
        self.assertEqual(len(row["landmarks"]), 21)
        self.assertEqual(row["handedness_raw"], "Left")
        check = self.report["handedness_vs_mediapipe"]
        # kitchen was declared "right" (agrees with the fake), hall was declared "left" (disagrees)
        self.assertGreater(check["agree"], 0)
        self.assertGreater(check["disagree"], 0)

    def test_report_documents_coverage_and_leakage(self) -> None:
        self.assertEqual(self.report["missing_letters"]["val"], list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.assertTrue(self.report["leakage"]["ok_room_separated"])
        self.assertEqual(self.report["per_split_per_letter"]["train"]["C"], 6)  # 3 singles + 3 thinned hold frames
        self.assertEqual(Counter(r["split"] for r in self.rows)["test"], 13)


if __name__ == "__main__":
    unittest.main()
