"""scripts/check_webcam_sessions.py is read-only and flags the 'frames went to the old session' pattern."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_webcam_sessions", ROOT / "scripts" / "check_webcam_sessions.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def make_session(root: Path, sid: str, created: str, room: str, lighting="lamp", background="plain_wall") -> Path:
    folder = root / sid
    folder.mkdir(parents=True)
    (folder / "session.json").write_text(
        json.dumps({"session_id": sid, "created_at": created, "room": room, "lighting": lighting, "background": background, "glasses": False, "face_visible": True}),
        encoding="utf-8",
    )
    return folder


def add_sample(folder: Path, letter: str, captured: str, n: int, label: str | None = None, **override) -> dict:
    """`letter` is the folder; `label` (default: the same) is what the JSON claims."""
    sid = folder.name
    rec = {"sample_id": f"{letter}_{n}_aaaa", "session_id": sid, "letter": label or letter, "label_source": "person_confirmed",
           "captured_at": captured, "image": f"{letter}_{n}_aaaa.jpg", "mode": "single", "lighting": "lamp", "background": "plain_wall"}
    rec.update(override)
    (folder / letter).mkdir(exist_ok=True)
    (folder / letter / f"{letter}_{n}_aaaa.json").write_text(json.dumps(rec), encoding="utf-8")
    (folder / letter / rec["image"]).write_bytes(b"x")
    return rec


def run(root: Path) -> tuple[int, str]:
    out = io.StringIO()
    argv = sys.argv
    sys.argv = ["check", "--src", str(root)]
    try:
        with redirect_stdout(out):
            code = checker.main()
    finally:
        sys.argv = argv
    return code, out.getvalue()


def snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


class CheckWebcamSessionsTests(unittest.TestCase):
    def test_flags_samples_saved_into_the_old_session_after_a_newer_one_was_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = make_session(root, "s20261019_100000_plain-wall", "2026-10-19T10:00:00+00:00", "plain-wall")
            make_session(root, "s20261019_110000_bedroom", "2026-10-19T11:00:00+00:00", "bedroom")
            add_sample(old, "F", "2026-10-19T10:30:00+00:00", 1)
            add_sample(old, "F", "2026-10-19T11:30:00+00:00", 2)  # saved after the bedroom session existed
            code, text = run(root)
            self.assertEqual(code, 0)  # warnings only
            self.assertIn("saved AFTER the newer session s20261019_110000_bedroom", text)
            self.assertIn("empty session", text)

    def test_reports_counts_and_changes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = make_session(root, "s20261019_100000_room", "2026-10-19T10:00:00+00:00", "room")
            add_sample(folder, "F", "2026-10-19T10:01:00+00:00", 1)
            add_sample(folder, "F", "2026-10-19T10:02:00+00:00", 2, mode="hold")
            before = snapshot(root)
            code, text = run(root)
            self.assertEqual(code, 0)
            self.assertIn("F:1+1", text)
            self.assertEqual(snapshot(root), before)

    def test_errors_for_wrong_label_wrong_session_missing_image_and_unconfirmed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = make_session(root, "s20261019_100000_room", "2026-10-19T10:00:00+00:00", "room")
            add_sample(folder, "F", "2026-10-19T10:01:00+00:00", 1, label="B")  # label differs from its folder
            add_sample(folder, "C", "2026-10-19T10:02:00+00:00", 2, session_id="s20261019_999999_other")
            add_sample(folder, "D", "2026-10-19T10:03:00+00:00", 3, label_source="model_prediction")
            add_sample(folder, "E", "2026-10-19T10:04:00+00:00", 4, background="shelves")
            (folder / "E" / "E_4_aaaa.jpg").unlink()
            code, text = run(root)
            self.assertEqual(code, 1)
            for expected in ("label 'B' != folder 'F'", "records session_id='s20261019_999999_other'", "label_source='model_prediction'", "missing", "differs from session"):
                self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
