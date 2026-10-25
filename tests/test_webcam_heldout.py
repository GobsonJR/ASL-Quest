"""Tests for scripts/evaluate_webcam_heldout.py: eligibility, the coverage gate, metric definitions, and read-only behaviour."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("evaluate_webcam_heldout", ROOT / "scripts" / "evaluate_webcam_heldout.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def make_session(root: Path, sid: str, person: str, letters: dict[str, int], *, hold: int = 0, negatives: dict[str, int] | None = None) -> None:
    folder = root / sid
    folder.mkdir(parents=True)
    (folder / "session.json").write_text(json.dumps({"session_id": sid, "person": person, "lighting": "lamp", "background": "shelves",
                                                      "glasses": False, "face_visible": True}), encoding="utf-8")
    n = 0
    for letter, count in letters.items():
        (folder / letter).mkdir()
        for i in range(count):
            name = f"{letter}_{n}_{i:04x}"
            Image.new("RGB", (32, 32), (i * 5 % 255, 10, 10)).save(folder / letter / f"{name}.jpg")
            seq = f"h_{letter}" if i < hold else None
            (folder / letter / f"{name}.json").write_text(json.dumps({
                "sample_id": name, "letter": letter, "image": f"{name}.jpg", "hand": "right", "distance": "medium",
                "mode": "hold" if seq else "single", "sequence_id": seq, "frame_index": i if seq else None,
                "quality": {"warnings": ["hand touches the frame edge (fingers may be cut off)"] if i == 0 else []}}), encoding="utf-8")
            n += 1
    for kind, count in (negatives or {}).items():
        neg = folder / "negatives" / kind
        neg.mkdir(parents=True)
        for i in range(count):
            name = f"N_{kind}_{i}_{i:04x}"
            Image.new("RGB", (32, 32), (i * 7 % 255, 0, 0)).save(neg / f"{name}.jpg")
            (neg / f"{name}.json").write_text(json.dumps({"sample_id": name, "kind": kind, "image": f"{name}.jpg", "touches_edge": kind == "partial_hand"}), encoding="utf-8")


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


class EligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        make_session(self.root, "s20261019_100000_plain-wall", "ann", {"A": 3})      # a model trained on this room
        make_session(self.root, "s20261020_100000_kitchen", "ann", {"B": 3})          # same PERSON as training
        make_session(self.root, "s20261020_110000_hall", "ben", {"C": 4}, hold=4, negatives={"no_hand": 2, "partial_hand": 1})
        make_session(self.root, "s20261020_120000_porch", "", {"D": 2})               # no person recorded

    def test_training_rooms_and_training_people_are_excluded_with_reasons(self) -> None:
        items, audit = ev.build_items(self.root, {"plain-wall"})
        self.assertEqual(sorted(audit["excluded_sessions"]), ["s20261019_100000_plain-wall", "s20261020_100000_kitchen"])
        self.assertIn("a model trained on this room", audit["excluded_sessions"]["s20261019_100000_plain-wall"])
        self.assertIn("person 'ann'", audit["excluded_sessions"]["s20261020_100000_kitchen"])
        self.assertEqual({i["letter"] for i in items}, {"C", "D", "NONE"})

    def test_a_declared_training_person_is_excluded_everywhere(self) -> None:
        items, audit = ev.build_items(self.root, {"plain-wall"}, declared_training_persons={"ben"})
        self.assertEqual({i["letter"] for i in items}, {"D"})  # ben's hall session is gone; the person-less porch session stays
        self.assertIn("person 'ben'", audit["excluded_sessions"]["s20261020_110000_hall"])

    def test_hold_sequences_are_thinned_and_negatives_carry_no_letter(self) -> None:
        items, _ = ev.build_items(self.root, {"plain-wall"})
        self.assertEqual(sum(1 for i in items if i["letter"] == "C"), ev.HOLD_KEEP)
        negatives = [i for i in items if i["letter"] == "NONE"]
        self.assertEqual(sorted(i["meta"]["kind"] for i in negatives), ["no_hand", "no_hand", "partial_hand"])
        self.assertTrue(all(i["meta"]["person"] == "ben" for i in negatives))

    def test_edge_touching_positives_are_flagged_from_the_collector_warning(self) -> None:
        items, _ = ev.build_items(self.root, {"plain-wall"})
        self.assertTrue(any(i["meta"]["touches_edge"] for i in items if i["letter"] == "D"))

    def test_nothing_on_disk_is_modified(self) -> None:
        before = tree_digest(self.root)
        ev.build_items(self.root, {"plain-wall"})
        self.assertEqual(before, tree_digest(self.root))

    def test_training_footprint_comes_from_model_metadata_and_ignores_dataset_blocks(self) -> None:
        model = Path(self.tmp.name) / "m.joblib"
        joblib.dump({"trained_on": {"rooms": ["original-block0", "original-block3", "plain-wall"]}}, model)
        self.assertEqual(ev.training_footprint([model]), {"rooms": ["plain-wall"]})


class CoverageGate(unittest.TestCase):
    @staticmethod
    def item(letter, person="p1", room="r1", lighting="lamp", distance="medium", hand="right", kind="single", edge=False):
        return {"letter": letter, "meta": {"person": person, "room": room, "lighting": lighting, "distance": distance, "hand": hand, "kind": kind, "touches_edge": edge}}

    def test_thin_data_never_passes(self) -> None:
        cov = ev.coverage([self.item("A")] * 50)
        self.assertFalse(cov["met_all"])
        unmet = {c["requirement"] for c in cov["checks"] if not c["met"]}
        self.assertTrue(any("J and Z" in r for r in unmet) and any("named people" in r for r in unmet))

    def test_complete_collection_passes_and_each_requirement_is_individually_necessary(self) -> None:
        def full(skip=None):
            items = []
            for l_i, letter in enumerate(ev.LETTERS):
                for k in range(36):
                    person = f"p{k % 3}"
                    items.append(self.item(letter, person=person, room=f"r{k % 3}", lighting=("lamp", "dim", "daylight")[k % 3],
                                           distance=("near", "medium", "far")[(k // 3) % 3], hand="left" if k < 8 else "right", edge=(l_i * 36 + k) % 15 == 0))
            items += [self.item("NONE", kind="no_hand")] * 100 + [self.item("NONE", kind="partial_hand")] * 100
            return items

        self.assertTrue(ev.coverage(full())["met_all"], [c for c in ev.coverage(full())["checks"] if not c["met"]])
        no_negatives = [i for i in full() if i["letter"] != "NONE"]
        self.assertFalse(ev.coverage(no_negatives)["met_all"])
        no_j = [i for i in full() if i["letter"] != "J"]
        self.assertFalse(ev.coverage(no_j)["met_all"])
        one_person = [{**i, "meta": {**i["meta"], "person": "p0"}} for i in full()]
        self.assertFalse(ev.coverage(one_person)["met_all"])

    def test_frames_with_no_person_do_not_count_as_a_person(self) -> None:
        cov = ev.coverage([self.item("A", person="")] * 10)
        people = next(c for c in cov["checks"] if "named people" in c["requirement"])
        self.assertFalse(people["met"])
        self.assertIn("0 named", people["have"])


def rec(letter, detected=True, answers=None, kind="single"):
    out = {"id": f"x{id(object())}{letter}", "letter": letter, "detected": detected, "meta": {"kind": kind, "person": "p"}, "brightness": 120.0}
    for system, ans in (answers or {}).items():
        out[system] = {"top": [ans], "ms": 1.0} if ans else None
    return out


class MetricDefinitions(unittest.TestCase):
    def test_end_to_end_counts_misses_while_conditional_does_not(self) -> None:
        records = [rec("A", answers={"s": ("A", 0.9)}), rec("A", answers={"s": ("B", 0.9)}), rec("A", detected=False, answers={"s": None}), rec("B", detected=False, answers={"s": None})]
        m = ev.metrics(records, "s")
        self.assertEqual((m["correct"], m["answered"], m["n_positive"]), (1, 2, 4))
        self.assertEqual((m["detected"],), (2,))

    def test_threshold_splits_accepted_wrong_and_rejected_correct(self) -> None:
        records = [rec("A", answers={"s": ("A", 0.95)}), rec("A", answers={"s": ("A", 0.40)}), rec("A", answers={"s": ("C", 0.80)}), rec("A", answers={"s": ("C", 0.30)})]
        m = ev.metrics(records, "s", 0.70)
        self.assertEqual((m["accepted"], m["accepted_correct"], m["accepted_wrong"], m["rejected_but_correct"]), (2, 1, 1, 1))
        sweep = {row["threshold"]: row for row in ev.threshold_sweep(records, "s", (0.2, 0.9))}
        self.assertEqual(sweep[0.2]["accepted_share"], 1.0)
        self.assertEqual(sweep[0.9]["accepted_share"], 0.25)

    def test_precision_and_recall_per_letter(self) -> None:
        records = [rec("A", answers={"s": ("A", 0.9)}), rec("A", answers={"s": ("B", 0.9)}), rec("B", answers={"s": ("B", 0.9)}), rec("B", detected=False, answers={"s": None})]
        pl = ev.metrics(records, "s")["per_letter"]
        self.assertEqual((pl["A"]["recall"], pl["A"]["precision"]), (0.5, 1.0))
        self.assertEqual((pl["B"]["recall"], pl["B"]["precision"]), (0.5, 0.5))  # the missed hand lowers recall, not precision
        self.assertIsNone(pl["C"]["recall"])

    def test_negatives_are_scored_separately_and_never_mixed_into_accuracy(self) -> None:
        records = [rec("A", answers={"s": ("A", 0.9)}), rec("NONE", kind="no_hand", answers={"s": ("F", 0.9)}), rec("NONE", kind="no_hand", detected=False, answers={"s": None}),
                   rec("NONE", kind="partial_hand", answers={"s": ("F", 0.2)})]
        m = ev.metrics(records, "s")
        self.assertEqual((m["n_positive"], m["correct"]), (1, 1))
        self.assertEqual(m["negatives"]["no_hand"], {"n": 2, "answered": 1, "accepted": 1, "detected": 1})
        self.assertEqual(m["negatives"]["partial_hand"], {"n": 1, "answered": 1, "accepted": 0, "detected": 1})

    def test_wilson_interval_is_sane(self) -> None:
        lo, hi = ev.wilson(5, 10)
        self.assertTrue(0.2 < lo < 0.5 < hi < 0.8)
        self.assertEqual(ev.fmt(0, 0), "n/a")


if __name__ == "__main__":
    unittest.main()
