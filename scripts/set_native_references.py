"""Attach reference-video links to native signs from a JSON file.

    venv311\\Scripts\\python.exe scripts\\set_native_references.py data\\native_reference_urls.json          (dry run)
    venv311\\Scripts\\python.exe scripts\\set_native_references.py data\\native_reference_urls.json --apply

The file maps a gloss to {"url": "https://...", "license_note": "who owns it / why linking is allowed"}. See
data/native_reference_urls.example.json. Entries with an empty url are skipped. Back up data/asl_quest.db first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.database import SessionLocal  # noqa: E402
from backend.services.native_references import ReferenceError_, attach_references, source_type_for, validate_entry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--apply", action="store_true", help="write to the database (default is a dry run)")
    args = ap.parse_args()
    mapping = {g.upper(): e for g, e in json.loads(args.file.read_text(encoding="utf-8")).items() if (e.get("url") or "").strip()}
    if not mapping:
        sys.exit("no entries with a url in the file")
    try:
        for gloss, entry in mapping.items():
            url, _ = validate_entry(gloss, entry)
            print(f"{gloss:10s} {source_type_for(url):8s} {url}")
        if not args.apply:
            print("dry run only; add --apply to write")
            return
        db = SessionLocal()
        try:
            print("attached:", attach_references(db, mapping))
        finally:
            db.close()
    except ReferenceError_ as error:
        sys.exit(f"refused: {error}")


if __name__ == "__main__":
    main()
