"""The promotion gate and rollback (scripts/promote_checkpoint.py), run against a temp models folder."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("promote_checkpoint", ROOT / "scripts" / "promote_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["promote_checkpoint"] = module
    spec.loader.exec_module(module)
    return module


def evaluation(macro_f1: float, accepted: float = 0.9, wrong_confirm: float | None = 0.0, rooms=("kitchen",)) -> dict:
    return {"rooms": list(rooms), "split": "test", "overall": {"macro_f1_over_present_letters": macro_f1, "accuracy_among_accepted": accepted},
            "stability": {} if wrong_confirm is None else {"confirmed_wrong": wrong_confirm}}


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p = load()

    def test_a_clear_improvement_passes(self) -> None:
        self.assertEqual(self.p.gate(evaluation(0.80), evaluation(0.70)), [])

    def test_a_worse_model_is_refused(self) -> None:
        self.assertTrue(any("macro-F1" in r for r in self.p.gate(evaluation(0.60), evaluation(0.70))))

    def test_min_gain_is_enforced(self) -> None:
        self.assertTrue(self.p.gate(evaluation(0.71), evaluation(0.70), min_gain=0.05))
        self.assertEqual(self.p.gate(evaluation(0.76), evaluation(0.70), min_gain=0.05), [])

    def test_a_model_that_confirms_wrong_letters_more_often_is_refused_even_if_f1_is_higher(self) -> None:
        reasons = self.p.gate(evaluation(0.80, wrong_confirm=0.15), evaluation(0.70, wrong_confirm=0.05))
        self.assertTrue(any("wrong letter" in r for r in reasons))

    def test_lower_accuracy_among_accepted_is_refused(self) -> None:
        self.assertTrue(any("accepted" in r for r in self.p.gate(evaluation(0.80, accepted=0.80), evaluation(0.70, accepted=0.92))))

    def test_different_test_rooms_are_not_comparable(self) -> None:
        reasons = self.p.gate(evaluation(0.9, rooms=("attic",)), evaluation(0.7, rooms=("kitchen",)))
        self.assertEqual(len(reasons), 1)
        self.assertIn("not the same test", reasons[0])


class PromoteAndRollbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p = load()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.p.BACKUP = root / "backup"
        self.current, self.candidate = root / "best_model.pth", root / "candidate.pth"
        self.current.write_bytes(b"CURRENT-MODEL")
        self.candidate.write_bytes(b"CANDIDATE-MODEL")
        self.ev_good, self.ev_cur = root / "good.json", root / "cur.json"
        self.ev_good.write_text(json.dumps(evaluation(0.9)), encoding="utf-8")
        self.ev_cur.write_text(json.dumps(evaluation(0.7)), encoding="utf-8")

    def test_promotion_backs_up_first_then_replaces_and_logs(self) -> None:
        message = self.p.promote(self.candidate, self.current, self.ev_good, self.ev_cur, min_gain=0.0, force=False)
        self.assertEqual(self.current.read_bytes(), b"CANDIDATE-MODEL")
        backups = list(self.p.BACKUP.glob("best_model_before_promotion_*.pth"))
        self.assertEqual([b.read_bytes() for b in backups], [b"CURRENT-MODEL"])
        self.assertIn("gate passed", (self.p.BACKUP / "PROMOTION_LOG.md").read_text(encoding="utf-8"))
        self.assertIn("rollback", message)

    def test_refused_promotion_changes_nothing(self) -> None:
        worse = self.ev_good.parent / "worse.json"
        worse.write_text(json.dumps(evaluation(0.5)), encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            self.p.promote(self.candidate, self.current, worse, self.ev_cur, min_gain=0.0, force=False)
        self.assertIn("NOT promoted", str(ctx.exception))
        self.assertEqual(self.current.read_bytes(), b"CURRENT-MODEL")
        self.assertFalse(self.p.BACKUP.exists())

    def test_promotion_without_a_measured_comparison_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self.p.promote(self.candidate, self.current, None, None, min_gain=0.0, force=False)
        self.assertEqual(self.current.read_bytes(), b"CURRENT-MODEL")

    def test_force_overrides_but_is_logged(self) -> None:
        worse = self.ev_good.parent / "worse.json"
        worse.write_text(json.dumps(evaluation(0.5)), encoding="utf-8")
        self.p.promote(self.candidate, self.current, worse, self.ev_cur, min_gain=0.0, force=True)
        self.assertEqual(self.current.read_bytes(), b"CANDIDATE-MODEL")
        self.assertIn("WITH --force", (self.p.BACKUP / "PROMOTION_LOG.md").read_text(encoding="utf-8"))

    def test_rollback_restores_the_previous_model_and_keeps_the_replaced_one(self) -> None:
        self.p.promote(self.candidate, self.current, self.ev_good, self.ev_cur, min_gain=0.0, force=False)
        self.p.rollback(self.current, None)
        self.assertEqual(self.current.read_bytes(), b"CURRENT-MODEL")
        saved = list(self.p.BACKUP.glob("best_model_before_rollback_*.pth"))
        self.assertEqual([s.read_bytes() for s in saved], [b"CANDIDATE-MODEL"])

    def test_rollback_with_no_backup_is_an_error(self) -> None:
        with self.assertRaises(SystemExit):
            self.p.rollback(self.current, None)
        self.assertEqual(self.current.read_bytes(), b"CURRENT-MODEL")


if __name__ == "__main__":
    unittest.main()
