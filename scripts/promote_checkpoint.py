"""Promote a candidate checkpoint to production, or roll back, with a mandatory backup and a written log.

    # promote (refuses unless the candidate wins on the SAME unseen-room evaluation)
    venv311\\Scripts\\python.exe scripts\\promote_checkpoint.py --candidate models\\experiments\\az_webcam_best.pth ^
        --eval-candidate outputs\\evaluation_webcam\\resnet_webcam.json --eval-current outputs\\evaluation_webcam\\resnet_current.json

    # roll back to the most recent backup
    venv311\\Scripts\\python.exe scripts\\promote_checkpoint.py --rollback

Every promotion first copies the current models/best_model.pth to models/backup/ (timestamp + hash in the name) and appends
to models/backup/PROMOTION_LOG.md. Restart the backend afterwards (it loads the checkpoint at startup).
Gate (override only with --force, which is logged): same test rooms, macro-F1 not lower than the current model's (plus
--min-gain), accuracy among accepted predictions not more than 1 point lower, and no increase in the share of hold
sequences confirmed as a WRONG letter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
BACKUP = MODELS / "backup"


def sha12(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def gate(candidate: dict, current: dict, min_gain: float = 0.0) -> list[str]:
    """Reasons to refuse (empty list = the candidate may be promoted)."""
    problems = []
    if candidate["rooms"] != current["rooms"] or candidate["split"] != current["split"]:
        return [f"the evaluations are not the same test (rooms {candidate['rooms']} vs {current['rooms']}, split {candidate['split']} vs {current['split']})"]
    c, k = candidate["overall"], current["overall"]
    if (c["macro_f1_over_present_letters"] or 0) < (k["macro_f1_over_present_letters"] or 0) + min_gain:
        problems.append(f"macro-F1 {100 * (c['macro_f1_over_present_letters'] or 0):.1f}% is not above the current {100 * (k['macro_f1_over_present_letters'] or 0):.1f}%" + (f" + {100 * min_gain:.1f} required" if min_gain else ""))
    if (c["accuracy_among_accepted"] or 0) < (k["accuracy_among_accepted"] or 0) - 0.01:
        problems.append("accuracy among accepted predictions dropped by more than 1 point")
    cw, kw = candidate.get("stability", {}).get("confirmed_wrong"), current.get("stability", {}).get("confirmed_wrong")
    if cw is not None and kw is not None and cw > kw:
        problems.append(f"it confirms a wrong letter more often on held signs ({100 * cw:.0f}% vs {100 * kw:.0f}%)")
    return problems


def log(message: str) -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    with (BACKUP / "PROMOTION_LOG.md").open("a", encoding="utf-8") as handle:
        handle.write(f"- {datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}\n")


def backup_current(current: Path, reason: str) -> Path:
    BACKUP.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = BACKUP / f"best_model_{reason}_{stamp}_{sha12(current)}.pth"
    shutil.copy2(current, target)
    return target


def promote(candidate: Path, current: Path, eval_candidate: Path | None, eval_current: Path | None, *, min_gain: float, force: bool) -> str:
    if not candidate.exists():
        raise SystemExit(f"candidate not found: {candidate}")
    if eval_candidate and eval_current:
        problems = gate(json.loads(eval_candidate.read_text(encoding="utf-8")), json.loads(eval_current.read_text(encoding="utf-8")), min_gain)
    else:
        problems = ["no evaluation files given (--eval-candidate and --eval-current): promotion needs a measured comparison"]
    if problems and not force:
        raise SystemExit("NOT promoted:\n  - " + "\n  - ".join(problems) + "\n(--force overrides this and is logged)")
    backup = backup_current(current, "before_promotion")
    shutil.copy2(candidate, current)
    note = f"PROMOTED {candidate.name} (sha {sha12(candidate)}) over {backup.name}" + (f" WITH --force despite: {problems}" if problems else " (gate passed)")
    log(note)
    return f"{note}\nrollback: python scripts/promote_checkpoint.py --rollback\nrestart the backend to load it"


def rollback(current: Path, to: Path | None) -> str:
    candidates = sorted(BACKUP.glob("best_model_*.pth"), key=lambda p: p.stat().st_mtime)
    source = to or (candidates[-1] if candidates else None)
    if source is None or not source.exists():
        raise SystemExit("no backup to roll back to")
    saved = backup_current(current, "before_rollback") if current.exists() else None
    shutil.copy2(source, current)
    note = f"ROLLED BACK to {source.name}" + (f" (the replaced model was saved as {saved.name})" if saved else "")
    log(note)
    return note + "\nrestart the backend to load it"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--current", type=Path, default=MODELS / "best_model.pth")
    ap.add_argument("--eval-candidate", type=Path)
    ap.add_argument("--eval-current", type=Path)
    ap.add_argument("--min-gain", type=float, default=0.0, help="required macro-F1 improvement (0.02 = 2 points)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--to", type=Path, help="with --rollback: a specific backup file")
    args = ap.parse_args()
    if args.rollback:
        print(rollback(args.current, args.to))
    elif args.candidate:
        print(promote(args.candidate, args.current, args.eval_candidate, args.eval_current, min_gain=args.min_gain, force=args.force))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
