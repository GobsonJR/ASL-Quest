"""Insert native_signs catalog rows for the asl_citizen_native_10 vocabulary.

Idempotent: skips any gloss that already has a native_signs row. Only inserts into
native_signs (created empty by migration 0002); touches no other table and no other
experiment's data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import SessionLocal
from backend.models import NativeSign

DATA_DIR = ROOT / "data" / "asl_citizen_native_10"

# display_name / meaning / category / difficulty are catalog metadata (not present in the
# ASL Citizen source CSVs), written in the same style as backend/words/catalog.py.
CATALOG: list[dict[str, str]] = [
    {
        "gloss": "HELLO",
        "display_name": "Hello",
        "meaning": "A greeting used when meeting someone.",
        "category": "Greetings",
        "difficulty": "Easy",
        "description": "The sign for HELLO, a common first sign for new learners.",
        "example_text": "Hello, nice to meet you.",
    },
    {
        "gloss": "THANKYOU",
        "display_name": "Thank You",
        "meaning": "An expression of gratitude.",
        "category": "Courtesy",
        "difficulty": "Easy",
        "description": "The sign for THANK YOU.",
        "example_text": "Thank you for your help.",
    },
    {
        "gloss": "PLEASE",
        "display_name": "Please",
        "meaning": "A polite word used to make a request.",
        "category": "Courtesy",
        "difficulty": "Easy",
        "description": "The sign for PLEASE.",
        "example_text": "Please pass the book.",
    },
    {
        "gloss": "YES",
        "display_name": "Yes",
        "meaning": "An affirmative response.",
        "category": "Responses",
        "difficulty": "Easy",
        "description": "The sign for YES, a basic affirmative response.",
        "example_text": "Yes, that is correct.",
    },
    {
        "gloss": "NO",
        "display_name": "No",
        "meaning": "A negative response.",
        "category": "Responses",
        "difficulty": "Easy",
        "description": "The sign for NO, a basic negative response.",
        "example_text": "No, that is not right.",
    },
    {
        "gloss": "MOTHER",
        "display_name": "Mother",
        "meaning": "A female parent.",
        "category": "Family",
        "difficulty": "Easy",
        "description": "The sign for MOTHER.",
        "example_text": "My mother is a teacher.",
    },
    {
        "gloss": "WATER",
        "display_name": "Water",
        "meaning": "A clear liquid essential for life.",
        "category": "Everyday",
        "difficulty": "Easy",
        "description": "The sign for WATER.",
        "example_text": "I would like a glass of water.",
    },
    {
        "gloss": "EAT1",
        "display_name": "Eat",
        "meaning": "To consume food.",
        "category": "Actions",
        "difficulty": "Easy",
        "description": "The sign for EAT.",
        "example_text": "Let's eat dinner.",
    },
    {
        "gloss": "HELP",
        "display_name": "Help",
        "meaning": "To assist someone.",
        "category": "Actions",
        "difficulty": "Easy",
        "description": "The sign for HELP.",
        "example_text": "Can you help me?",
    },
    {
        "gloss": "BOOK",
        "display_name": "Book",
        "meaning": "A written or printed work.",
        "category": "Everyday",
        "difficulty": "Easy",
        "description": "The sign for BOOK.",
        "example_text": "I am reading a good book.",
    },
]


def main() -> dict:
    class_to_idx = json.loads((DATA_DIR / "class_to_idx.json").read_text(encoding="utf-8"))
    expected_glosses = set(class_to_idx.keys())
    catalog_glosses = {entry["gloss"] for entry in CATALOG}
    if catalog_glosses != expected_glosses:
        raise RuntimeError(
            f"Catalog/manifest gloss mismatch. Manifest: {sorted(expected_glosses)}, "
            f"catalog: {sorted(catalog_glosses)}"
        )

    db = SessionLocal()
    created: list[str] = []
    skipped: list[str] = []
    try:
        for entry in CATALOG:
            existing = db.query(NativeSign).filter(NativeSign.gloss == entry["gloss"]).first()
            if existing is not None:
                skipped.append(entry["gloss"])
                continue
            db.add(
                NativeSign(
                    gloss=entry["gloss"],
                    display_name=entry["display_name"],
                    meaning=entry["meaning"],
                    category=entry["category"],
                    difficulty=entry["difficulty"],
                    description=entry["description"],
                    example_text=entry["example_text"],
                    dataset_available=True,
                    model_available=False,
                    active=True,
                )
            )
            created.append(entry["gloss"])
        db.commit()

        rows = db.query(NativeSign).filter(NativeSign.gloss.in_(catalog_glosses)).all()
        result = {
            "created": created,
            "skipped_existing": skipped,
            "total_rows_now": len(rows),
            "rows": [
                {
                    "id": row.id,
                    "gloss": row.gloss,
                    "display_name": row.display_name,
                    "category": row.category,
                    "dataset_available": row.dataset_available,
                    "model_available": row.model_available,
                    "active": row.active,
                }
                for row in sorted(rows, key=lambda r: r.gloss)
            ],
        }
        return result
    finally:
        db.close()


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
